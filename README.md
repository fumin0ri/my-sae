# Predictive sparse residual streams

LLMの「近接tokenに共通する内部状態」を、単なる平均や低ランク相関ではなく、**離れた文脈から予測可能な疎なresidual-space feature**として同定する研究コードです。

中心仮説は次です。

> ある中間表現がLLM内で維持される計算状態なら、target token自体を見なくても、手前のcontext residualからgapを越えてその表現を予測できる。その予測可能部分を除いた残差は、新規入力・局所token・surpriseを表すinnovationになる。

## 一発で動かす

LinuxのGPUマシン（SSH接続先を想定）で以下を実行します。

```bash
git clone https://github.com/fumin0ri/my-sae.git
cd my-sae
python -m venv .venv
source .venv/bin/activate
pip install -U pip
python -m pip install --upgrade -e .
bash scripts/research_quickstart.sh
```

### PyTorch 2.6以上が必要

公式PythiaチェックポイントはSafeTensorsではなくPyTorch `.bin` 形式です。
CVE-2025-32434への対策として、TransformersはTorch 2.6未満でこの形式の
読み込みを拒否します。`HF_HUB_DISABLE_TORCH_SECURITY_CHECK` では安全に回避
できないため、次のコマンドで現在の環境を更新してください。

```bash
# RTX 4090 / CUDA 12.4向けの再現性を重視した推奨構成
python -m pip install --upgrade \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --upgrade -e .
python -c "import torch; print(torch.__version__, torch.version.cuda)"
bash scripts/research_quickstart.sh
```

クイックスタートは13.9GBのモデルをダウンロードする前にTorchのバージョンを
検査し、不足している場合はアップグレードコマンドを表示して停止します。

デフォルトは**単一RTX 4090（24GB）向けprofile**です。

| 項目 | 設定 |
|---|---:|
| frozen LLM | `EleutherAI/pythia-6.9b-deduped` |
| residual layer | 16 / 32 |
| residual width | 4,096 |
| SAE dictionary | 32,768 features（8× expansion） |
| sparsity | Top-K 64 |
| JEPA predictor | width 256、8 heads、3 layers |
| mask | context 32、gap 2/4/8、target 2/4/8/16 |
| training | BF16 autocast + TF32 + fused AdamW |
| batch | micro 16 × accumulation 2 = effective 32 |
| optimizer steps | 12,000 |

Pythia-12BはBF16重みだけで24GB級となり、因果介入時にSAEを同じGPUへ置けません。6.9Bは32層・hidden 4,096で、8× SAEと同居できる最大寄りの構成です。

次を一括実行します。

1. paraphraseをproblem groupとして管理したcontrolled benchmarkを生成
2. frozen LLMから64-token residual windowをBF16で抽出
3. JEPA-regularized SAEを学習
4. standard SAE + frozen post-hoc predictorを同じsplitで学習
5. 未使用のlocked testを一度だけ評価
6. 従来のlow-rank random-effects法をbaselineとしてfit
7. predictable featureのpatch、ablation、norm-matched random ablation
8. HTML、PNG、PDFの研究レポートを生成

結果はここに出ます。

```text
runs/predictive-research/report/index.html
```

同じrun directoryにcode commit、`pip freeze`、GPU名・VRAM・driverも保存されます。

SSH先でレポートを見る場合:

```bash
python -m http.server 8000 --directory runs/predictive-research/report
```

手元PCから:

```bash
ssh -L 8000:localhost:8000 <user>@<ssh-host>
```

その後 `http://localhost:8000` を開きます。

最初に軽く動作確認する場合:

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 \
D_SAE=2048 \
PREDICTOR_WIDTH=128 \
PREDICTOR_HEADS=4 \
PREDICTOR_LAYERS=2 \
STEPS=300 \
PROBLEMS=40 \
EXTRACT_BATCH_SIZE=32 \
RUN_CAUSAL=0 \
bash scripts/research_quickstart.sh
```

OOMになった場合は、まずmicro batchだけを下げてeffective batchを維持します。

```bash
BATCH_SIZE=8 \
GRADIENT_ACCUMULATION=4 \
EXTRACT_BATCH_SIZE=4 \
bash scripts/research_quickstart.sh
```

それでも不足する場合のみ `D_SAE=16384` に下げてください。学習レポートにはGPU名、AMP/TF32/fused optimizerの状態、effective batch、peak allocated/reserved VRAMが保存されます。

## モデル

frozen LLMのlayer `l` におけるresidualを `h_t` とします。target span `B` の手前からgapを空けてcontext `C` を作ります。

```text
h_C ── online sparse encoder ── context transformer ── predictor ── ẑ_B
                                                                         │
h_B ── EMA target sparse encoder ──────────────────────────────── z_B     │
                                                                         ▼
original residual space ◀──────────────────────────────────────── decoder D
```

学習する主な量は次です。

```text
z_t       = TopK(ReLU(E_online(h_t - b)))
ĥ_t       = b + D z_t
z_target  = stopgrad(TopK(ReLU(E_EMA(h_B - b))))
z_predict = TopK(ReLU(P(E_online(h_C), position(B))))

L = L_reconstruct(h, ĥ)
  + α L_predict(z_predict, z_target)
  + β L_residual-predict(b + D z_predict, h_B)
```

decoderの各行は元のLLM residual spaceに存在します。したがってfeatureを解釈するだけでなく、元LLMへそのままpatchまたはablateできます。

分解は次です。

```text
predictable_B = b + D z_predict
innovation_B  = h_B - predictable_B
```

`predictable` は「contextから予測できる持続状態」の候補、`innovation` は予測できなかった局所更新の候補です。これは解釈であり、評価と因果介入を通らない段階では結論ではありません。

## decoder-only LLMでのmask

主実験は必ず次のcausal maskを使います。

```text
[ context C ][ unused gap ][ target B ]
```

decoder-only LLMではtargetより右のresidualは既にtarget tokenへattentionしています。左右のcontextからtargetを予測すると情報漏洩になるため、`--context-mode retrospective` は明示的なnegative-control / upper-bound ablationとしてのみ実装しています。主結果には使わないでください。

target spanとgapは学習時に複数のスケールからsampleします。

```bash
--target-sizes 2,4,8 --gaps 2,4,8
```

これにより、単なる隣接token copyingと、phrase程度の時間幅を持つ状態を分離します。

## 比較条件

同じarchitecture、dictionary size、Top-K、データsplitで次を比較します。

- `joint`: 提案法。SAE reconstructionとmasked predictionを共同学習
- `posthoc`: 通常SAEを先に学習して固定し、predictorだけを後付け学習
- `low-rank-baseline`: 従来のwindow random-effects generalized eigenspace
- raw target residual
- innovation residual

`posthoc`との比較で「通常SAEに既に予測可能featureが存在するだけか」と「予測目的が辞書自体を改善したか」を分けます。従来のlow-rank法、shared/private group SAE、SAE Lens解析のCLIも削除せずbaselineとして残しています。

## locked-test評価

splitはtoken window単位ではなく `group_id` 単位です。同じ問題のparaphraseは必ず同じsplitに入ります。

```text
60% train / 20% validation / 20% locked test
```

locked testで自動生成する主な結果:

- target sparse codeのgap越し予測cosine / NRMSE
- residual prediction FVU
- joint vs post-hocのtask-state probeとgroup bootstrap 95% CI
- same-problem paraphrase / same-state別問題 / different-stateのcosine
- predictable codeとinnovation residualの情報分離
- effective rank、dead feature率、L0によるcollapse診断
- gap × target-span heatmap
- top predictable featureと上位activation例
- original residualでのpatch / ablation / norm-matched random control
- paired bootstrap、sign-flip test、Cohen's `d_z`

詳しいconfirmatory hypothesis、falsification条件、replication matrixは [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) に固定しています。

Pythia 1.4B / 2.8B / 6.9B × 3 layer × 3 task family × 3 feature seedのprespecified replicationを回す場合:

```bash
bash scripts/replication_matrix.sh
```

各runのHTMLに加え、forest plot、CSV、JSONを次に集約します。

```text
runs/predictive-replication/summary/index.html
runs/predictive-replication/summary/replication_summary.csv
```

計算量が大きいため、最初はmatrixを縮めて確認できます。

```bash
MODEL_SPECS='EleutherAI/pythia-70m-deduped|1,3,5' \
TASK_FAMILIES=fsm \
SEEDS=0 \
STEPS=300 \
PROBLEMS=40 \
bash scripts/replication_matrix.sh
```

## 個別コマンド

学習:

```bash
sr-train-predictive-sae \
  --activations runs/layer-016.pt \
  --output-dir runs/joint \
  --objective joint \
  --d-sae 32768 \
  --k 64 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 3 \
  --context-width 32 \
  --target-sizes 2,4,8,16 \
  --gaps 2,4,8 \
  --steps 12000 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --amp-dtype bfloat16
```

通常SAE control:

```bash
sr-train-predictive-sae \
  --activations runs/layer-016.pt \
  --output-dir runs/posthoc \
  --objective posthoc \
  --d-sae 32768 \
  --k 64 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 3 \
  --context-width 32 \
  --target-sizes 2,4,8,16 \
  --gaps 2,4,8 \
  --steps 12000 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --amp-dtype bfloat16
```

locked evaluation:

```bash
sr-evaluate-predictive-sae \
  --activations runs/layer-016.pt \
  --joint-checkpoint runs/joint/predictive_sae.pt \
  --baseline-checkpoint runs/posthoc/predictive_sae.pt \
  --output-dir runs/analysis
```

featureを限定した因果介入:

```bash
sr-intervene-predictive-sae \
  --model EleutherAI/pythia-6.9b-deduped \
  --pairs data/research/pairs.jsonl \
  --checkpoint runs/joint/predictive_sae.pt \
  --output runs/analysis/feature-ablation.jsonl \
  --layer 16 \
  --mode ablate \
  --feature-ids 17,92,301 \
  --target-size 4 \
  --gap 4
```

可視化:

```bash
sr-visualize-predictive-sae --run-dir runs/predictive-research
```

## 研究としての注意

- 高いprediction scoreだけでは、token identity、template、absolute positionを予測している可能性があります。
- 高いprobe精度だけでは因果性を示しません。
- SAE featureは一意とは限りません。seed、dictionary size、Top-Kを変えた再現性が必要です。
- EMAだけをcollapse対策とみなさず、reconstruction、effective rank、feature variance、dead-feature率を必ず監視します。
- model/layer/hook/revisionがcheckpointと異なる因果介入はデフォルトで拒否します。
- layerやhyperparameterをlocked testで選ばないでください。
- 本研究の主張単位は単一promptではなく、model × task family × seedです。

## 関連する一次資料

- [Pythia: Interpreting Transformers Across Time and Scale](https://github.com/EleutherAI/pythia)
- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [Joint Embedding Predictive Architectures Focus on Slow Features](https://arxiv.org/abs/2211.10831)
- [LLM-JEPA: Large Language Models Meet Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
- [SparseJEPA](https://arxiv.org/abs/2504.16140)
- [C-JEPA: Contrastive-JEPA for Avoiding Representation Collapse](https://arxiv.org/abs/2410.19560)

このrepositoryの目的はJEPAをLLMの新しいtraining objectiveとして使うことではなく、**frozen LLMのresidual dynamicsから予測可能な計算状態を同定し、疎なresidual-space directionとして因果検証すること**です。
