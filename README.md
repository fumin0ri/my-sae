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
pip install -e .
bash scripts/research_quickstart.sh
```

デフォルトではPythia-70Mのlayer 3を使い、次を一括実行します。

1. paraphraseをproblem groupとして管理したcontrolled benchmarkを生成
2. frozen LLMから48-token residual windowを抽出
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
STEPS=300 D_SAE=512 RUN_CAUSAL=0 bash scripts/research_quickstart.sh
```

GPUメモリに合わせて変更する場合:

```bash
MODEL=EleutherAI/pythia-410m-deduped \
LAYER=12 \
D_SAE=8192 \
K=64 \
BATCH_SIZE=16 \
STEPS=10000 \
bash scripts/research_quickstart.sh
```

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

3 model size × 3 layer × 3 seedのprespecified replicationを回す場合:

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
SEEDS=0 \
STEPS=300 \
PROBLEMS=40 \
bash scripts/replication_matrix.sh
```

## 個別コマンド

学習:

```bash
sr-train-predictive-sae \
  --activations runs/layer-003.pt \
  --output-dir runs/joint \
  --objective joint \
  --d-sae 8192 \
  --k 64 \
  --context-width 32 \
  --target-sizes 2,4,8 \
  --gaps 2,4,8 \
  --steps 20000
```

通常SAE control:

```bash
sr-train-predictive-sae \
  --activations runs/layer-003.pt \
  --output-dir runs/posthoc \
  --objective posthoc \
  --d-sae 8192 \
  --k 64 \
  --context-width 32 \
  --target-sizes 2,4,8 \
  --gaps 2,4,8 \
  --steps 20000
```

locked evaluation:

```bash
sr-evaluate-predictive-sae \
  --activations runs/layer-003.pt \
  --joint-checkpoint runs/joint/predictive_sae.pt \
  --baseline-checkpoint runs/posthoc/predictive_sae.pt \
  --output-dir runs/analysis
```

featureを限定した因果介入:

```bash
sr-intervene-predictive-sae \
  --model EleutherAI/pythia-70m-deduped \
  --pairs data/research/pairs.jsonl \
  --checkpoint runs/joint/predictive_sae.pt \
  --output runs/analysis/feature-ablation.jsonl \
  --layer 3 \
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

- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [Joint Embedding Predictive Architectures Focus on Slow Features](https://arxiv.org/abs/2211.10831)
- [LLM-JEPA: Large Language Models Meet Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
- [SparseJEPA](https://arxiv.org/abs/2504.16140)
- [C-JEPA: Contrastive-JEPA for Avoiding Representation Collapse](https://arxiv.org/abs/2410.19560)

このrepositoryの目的はJEPAをLLMの新しいtraining objectiveとして使うことではなく、**frozen LLMのresidual dynamicsから予測可能な計算状態を同定し、疎なresidual-space directionとして因果検証すること**です。
