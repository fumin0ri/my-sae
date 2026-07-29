# Offset-conditioned Transition JEPA-SAE

Frozen autoregressive LLMの可変長residual trajectoryから、現在位置で
すでに予測可能な未来の内部状態を疎な辞書として抽出する研究コードです。
SAEとpredictorの学習にはThe Pileの公式22-subcorpus mixtureを使い、
MMLUは学習に混ぜずlocked評価だけに使います。

```text
h₀ ─ online Top-K SAE ─ z₀ ─┐
                              ├─ offset-conditioned MLP ─ softplus ─ ẑₖ
offset embedding(k) ──────────┘

hₖ ─ EMA target SAE ─ stopgrad(zₖ),  k = 1,...,W-1
```

`z₀`とoffset `k`だけから、`z₁...z_{W-1}`を個別に予測します。targetの平均化や
Transformer predictorは使いません。主張する対象は完全な未来状態ではなく、
データ分布の下で`z₀`から予測可能な成分です。

```text
P(z₀, k) ≈ E[zₖ | z₀, k]
```

後続tokenを入力していないため、これは決定論的な状態遷移ではありません。

## 一発実行

CUDA 12.1、`torch==2.5.1+cu121`、単一RTX 4090向けの標準設定です。
Hugging FaceからはSafeTensors revisionだけを読み込みます。

```bash
git clone https://github.com/fumin0ri/my-sae.git
cd my-sae
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --upgrade -e .
bash scripts/transition_jepa_quickstart.sh
```

既存cloneを更新する場合:

```bash
cd ~/my-sae
git pull origin main
python -m pip install --upgrade -e .
bash scripts/transition_jepa_quickstart.sh
```

標準設定:

| 項目 | 設定 |
|---|---:|
| frozen LLM | `EleutherAI/pythia-6.9b-deduped` |
| residual layer | 16 / 32 |
| residual width | 4,096 |
| SAE dictionary | 32,768 features |
| sparsity | Top-K 64 |
| predictor | width 512, offset-conditioned MLP |
| residual window | 10 positions（`WINDOW_SIZE`で変更可能） |
| Pile train sample | 5,242,880 token positions（W=10では524,288 windows） |
| Pile validation | 163,840 token positions（W=10では16,384 windows） |
| activation storage | BF16 shards、実データ約41 GiB |
| standard SAE | 12,000 steps |
| JEPA conditions | 各8,000 steps |
| arithmetic | BF16 autocast + TF32 + fused AdamW |
| evaluation | MMLU test最大14,042問（実token長 ≥ W） |

結果は次の自己完結HTMLに出ます。

```text
runs/transition-jepa-pile/report/index.html
```

## 軽量smoke test

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 \
D_SAE=2048 \
K=32 \
PREDICTOR_WIDTH=64 \
PILE_TRAIN_WINDOWS=4096 \
PILE_VALIDATION_WINDOWS=512 \
PILE_SHARD_WINDOWS=512 \
STANDARD_STEPS=300 \
FORECAST_STEPS=300 \
PREDICTOR_WARMUP_STEPS=50 \
MMLU_MAX_QUESTIONS=320 \
PILE_EXTRACT_BATCH_SIZE=32 \
EVAL_EXTRACT_BATCH_SIZE=32 \
RUN_CAUSAL=0 \
bash scripts/transition_jepa_quickstart.sh
```

## Window size

既定値は10ですが、2以上の任意のwindow lengthへ変更できます。Pile extraction、
standard SAE、JEPA predictor、MMLU評価、causal intervention、可視化のすべてに
同じ値が伝播します。

```bash
WINDOW_SIZE=16 bash scripts/transition_jepa_quickstart.sh
```

`PILE_SEQUENCE_LENGTH`を明示する場合は`WINDOW_SIZE`の倍数にしてください。
省略時は約320 tokenを超えない最大の倍数へ自動調整されます。異なるwindow
sizeのrunは別の`RUN_DIR`へ保存することを推奨します。

既定の抽出量とshardサイズはwindow数ではなくtoken position数で固定されます。
したがって`WINDOW_SIZE=128`ではtrain 40,960 windows、validation 1,280
windows、1 shard 320 windowsとなり、W=10とほぼ同じ約41 GiBに収まります。
比較する独立window数を明示的に固定したい場合だけ`PILE_TRAIN_WINDOWS`、
`PILE_VALIDATION_WINDOWS`、`PILE_SHARD_WINDOWS`で上書きしてください。この場合は
必要容量がwindow sizeに比例します。
`BATCH_SIZE`と`EVAL_BATCH_SIZE`も既定では約320 positions/batchになるよう調整され、
W=128では2 windowsになります。VRAMに余裕がある場合は明示的に上書きできます。

完了済みstageがあるrunは`START_STAGE`で既存artifactから再開できます。たとえば
stage 9の評価だけをやり直してreportまで生成する場合:

```bash
WINDOW_SIZE=32 \
START_STAGE=9 \
RUN_CAUSAL=0 \
RUN_DIR=runs/transition-jepa-pile \
bash scripts/transition_jepa_quickstart.sh
```

`END_STAGE`を指定すると単一stageだけを安全に再実行できます。Wを変更した旧runで
stage 9に`base-model MMLU results and activation rows`の不一致が出た場合は、
checkpointとactivationを作り直さずstage 4だけを再採点してから再開します。

```bash
WINDOW_SIZE=128 LAYER=16 RUN_DIR=runs/l16_win128 \
START_STAGE=4 END_STAGE=4 \
bash scripts/transition_jepa_quickstart.sh

WINDOW_SIZE=128 LAYER=16 RUN_DIR=runs/l16_win128 \
START_STAGE=9 \
bash scripts/transition_jepa_quickstart.sh
```

stage 9はoffsetごとのdense codeを全件保持せず、batch内でscalar statisticsへ
集約します。大きいwindowでも、最終offsetのfeature解析に必要なcodeだけを
保持します。

## The Pile training data

`EleutherAI/the_pile_deduplicated`の`default` configurationをstreamingで読みます。
これはThe Pileの公式22-subcorpus mixtureへexact/near deduplicationを施した
Parquet版です。上流のpreweighted mixtureを保ったまま、10,000-document
bufferで追加shuffleします。datasetは再現性のためcommit
`fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb`へ固定しています。

このParquet releaseはdocumentごとのsubcorpus labelを公開していません。
したがってmanifestには公式のtarget mixtureと「source metadataなし」を記録し、
実測のsubcorpus比率を装って報告しません。ラベル付き旧releaseを監査目的で使う
場合は、次のように明示的に切り替えられます。ただし旧releaseは外部配布hostと
Hugging Face dataset scriptに依存します。

```bash
PILE_DATASET=EleutherAI/pile \
PILE_DATASET_REVISION=148e1d5e8349977c76f673190424a2faf6980a1d \
PILE_DATASET_TRUST_REMOTE_CODE=1 \
PILE_REQUIRE_ALL_DOMAINS=1 \
bash scripts/transition_jepa_quickstart.sh
```

documentは決定論的hashでtrain/validationへ分離するため、同じdocument由来の
windowが両方へ入ることはありません。抽出結果は単一巨大tensorではなく、既定で
40,960-position単位（約320 MiB）のBF16 shardとして保存されます。320 token未満の
短いdocumentも右paddingして実トークン部分だけを保存するため、短文中心の
subcorpusを捨てません。

抽出前には推定容量、filesystem空き容量、5 GiBのreserveを検査します。各shardは
`.partial`へ書いてから原子的に確定するため、disk fullやquota超過で壊れた`.pt`を
正式artifactとして残しません。stage 1で失敗した出力は自動再開しません。容量を
確認して失敗した`pile-activations` directoryを削除するか、新しい`RUN_DIR`を
指定してstage 1から再実行してください。quotaを別途確認済みの場合だけ
`PILE_SKIP_DISK_SPACE_CHECK=1`で事前検査を無効化できます。

```text
runs/transition-jepa-pile/pile-activations/
  manifest.json
  train/shard-*.pt
  validation/shard-*.pt
```

`manifest.json`には、利用可能なら実際のsubcorpus別window数、公式target
mixture、モデルrevision、layer、normalization statistics、データfingerprint
が記録されます。3条件は同じfingerprintを持つstandard SAE checkpointからしか
開始できません。

The Pileにはsubcorpusごとに異なるライセンスと利用条件があります。利用者は
各componentの条件を確認してください。

## 実験条件

すべて同じPile学習済みreconstruction-only standard SAE checkpointから
開始します。

- `joint`: predictor warm-up後、predictor・online encoder・decoderを共同学習
- `fixed`: standard SAEを固定し、同じpredictorだけ学習
- `k_only`: `z₀`を遮断し、offsetだけから予測
- shuffled context: locked testで別MMLU questionの`z₀`へ交換

online SAEはwindow内の全位置を再構成します。EMA target encoderはjoint条件だけで
更新されます。predictor出力は学習時にはdense non-negative softplusとし、
support評価・residual decoding・因果介入でだけTop-Kを適用します。

主損失:

```text
L = L_reconstruction
  + λ_prediction mean_k[
      1 - cosine(ẑₖ, zₖ)
      + 0.25 normalized_MSE(ẑₖ, zₖ)
      + λ_residual FVU(decode(TopK(ẑₖ)), hₖ)
    ]
  + λ_variance L_variance
```

## 評価と可視化

Pileと独立なMMLU question-grouped locked testで次を出力します。

- offset 1...`WINDOW_SIZE-1`のcode cosine、normalized MSE、support precision/recall/Jaccard
- true-context minus shuffled-context
- joint minus fixedのquestion-group bootstrap 95% CI
- residual prediction FVUとinnovation energy
- semantics accuracy: balanced option permutation後の正答A/B/C/D
- context accuracy: 公式MMLU大分類（STEM/humanities/social sciences/other）
- syntax accuracy: 内容と独立に均衡割付した4種類のprompt形式
- base LLMのzero-shot MMLU accuracy、collapse診断
- top forecastable featureと活性化例
- forecastable componentだけのpatch・ablation・norm-matched random対照
- PNG、PDF、CSV、JSON、自己完結HTML

因果patchは実際の未来code全体を置換せず、予測可能成分だけを編集します。

```text
Δhₖ = D(TopK(P(z₀_source,k)) - TopK(P(z₀_target,k)))
```

sourceとtargetの両方が`WINDOW_SIZE`以上の実tokenを持つpairだけを使います。
既定では必要な128 pairの16倍（2,048候補）を決定論的に生成し、適格な先頭128
pairをpatch・ablation・random対照で共通利用します。候補数は`PAIR_POOL_SIZE`、
実行pair数は`PAIRS`で変更できます。

旧runのstage 10で`prefix is shorter than checkpoint window`が出た場合、学習や
stage 9をやり直す必要はありません。stage 2でpair poolだけ再生成してからstage
10へ進みます。

```bash
WINDOW_SIZE=128 LAYER=16 RUN_DIR=runs/l16_win128 \
START_STAGE=2 END_STAGE=2 \
bash scripts/transition_jepa_quickstart.sh

WINDOW_SIZE=128 LAYER=16 RUN_DIR=runs/l16_win128 \
START_STAGE=10 \
bash scripts/transition_jepa_quickstart.sh
```

MMLUは`cais/mmlu` commit
`c30699e8356da336a370243923dbaf21066bb9fe`へ固定しています。既定ではtest
14,042問から、人工paddingを避けるため実token長が`WINDOW_SIZE`以上のquestionを
使います。activation抽出とbase LLM accuracyは同じquestion ID集合で検証されます。
W=10では通常ほぼ全問、Wが大きいほど短いpromptが除外されます。短い確認実験だけ
`MMLU_MAX_QUESTIONS`を設定してください。base LLM scoreは表層形式を統制した
zero-shot評価であり、公式leaderboardの5-shot protocolとは区別して報告します。

## Replication

model、layer、seed、出力先を環境変数で分離できます。

```bash
MODEL=EleutherAI/pythia-2.8b-deduped \
LAYER=16 \
SEED=1 \
SPLIT_SEED=1 \
RUN_DIR=runs/transition-jepa-pile-mmlu-seed1 \
bash scripts/transition_jepa_quickstart.sh
```

モデル・層・data split seed・feature seedを独立したreplication unitとして
扱ってください。

## コード構成

```text
src/shared_residual/
  pile_extract.py              Pile streaming・residual shard生成
  mmlu_data.py                 MMLU均衡prompt・causal pair生成
  mmlu_score.py                base LLM MMLU accuracy
  activation_store.py          shard-aware training iterator
  standard_sae.py              reconstruction-only初期SAE
  transition_jepa_sae.py       JEPA-SAE学習と3条件
  transition_jepa_eval.py      locked-test評価
  transition_jepa_intervene.py 因果patch/ablation
  transition_jepa_visualize.py PNG/PDF/HTML生成
  training.py                  split・AMP・optimizer補助
  evaluation.py                bootstrap・probe・診断
  intervention_utils.py        feature/offset選択

scripts/
  transition_jepa_quickstart.sh

docs/
  TRANSITION_JEPA_PROTOCOL.md
```

主仮説、confirmatory comparison、棄却条件、replication方針は
[`docs/TRANSITION_JEPA_PROTOCOL.md`](docs/TRANSITION_JEPA_PROTOCOL.md)に
固定しています。
