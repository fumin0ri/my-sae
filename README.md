# Predictor-free Rectified LpJEPA-SAE

LLMのresidual streamから、近接token位置で共有されるhigh-level sparse
featureと、位置固有のlow-level sparse featureを分けて学習する研究用pipelineです。
predictor、teacher encoder、in-batch negativesは使いません。

```text
(h_a, h_b) from one random span
        |          |
        +-- shared high/low encoder --+
                   |
        dense high:  ReLU + invariance + random/axis RDMReg
        sparse high: ReLU + Top-K + reconstruction/evaluation
        sparse low:  ReLU + Top-K + reconstruction
```

Dense highはLpJEPA損失専用の候補表現です。decoder、通常SAE評価、SAEBench、
swapでは常にTop-K後のsparse highを使用するため、最終SAEのhigh L0は
`HIGH_K`で制御できます。SAE warmupはなく、invarianceとRDMRegは同じ
regularization rampで最初から立ち上がります。

## Quick start

対象環境はCUDA 12.1、PyTorch 2.5.1、単一RTX 4090です。

```bash
git clone https://github.com/fumin0ri/my-sae.git
cd my-sae
conda create -n sae python=3.11 -y
conda activate sae
bash scripts/install_cuda121.sh
HF_TOKEN=... bash scripts/rectified_lpjepa_quickstart.sh
```

`install_cuda121.sh`はPyTorch 2.5.1+cu121に加え、CUDA 12.1と両立する
SAEBench 0.5.0、SAE Lens 6.5.0、TransformerLens 2.15.4を固定します。
SAEBench 0.6.0は依存先のTransformerLensがPyTorch 2.6以上を要求するため、
このCUDA 12.1環境では使用しません。

SAEBench追加後に`CUDA initialization: driver ... too old`が出る既存環境も、
次のコマンドで修復できます。

```bash
conda activate sae
cd my-sae
git pull
bash scripts/install_cuda121.sh
```

主な既定値はPythia 6.9B、layer 16、maximum span 32、SAE width 32768、
high fraction 0.2、high Top-K 128、low Top-K 64、1024 random projections、
512 axis coordinates、12000 optimizer stepsです。実行後は次を開きます。

```text
runs/rectified-lpjepa-pile/report/index.html
```

主要設定の変更例:

```bash
WINDOW_SIZE=4 \
LAYER=16 \
HIGH_FRACTION=0.5 \
HIGH_K=256 \
LOW_K=64 \
HIGH_RECONSTRUCTION_WEIGHT=0.5 \
TARGET_ACTIVE_FRACTION=0.025 \
AXIS_RDM_FEATURES=512 \
AXIS_RDM_WEIGHT=1 \
RUN_DIR=runs/l16-win4-dual-high \
bash scripts/rectified_lpjepa_quickstart.sh
```

`TARGET_ACTIVE_FRACTION * d_high`はdense候補の期待L0です。Top-Kを安定して
満たすには、この値を`HIGH_K`より十分大きくしてください。既定の
`d_high=16384, target=0.025, HIGH_K=256`では期待候補数は約410です。

軽量smoke test:

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 WINDOW_SIZE=8 D_SAE=1024 HIGH_K=4 LOW_K=16 \
PILE_TRAIN_POSITIONS=8192 PILE_VALIDATION_POSITIONS=2048 \
TRAIN_STEPS=20 REGULARIZATION_RAMP_STEPS=5 TARGET_ACTIVE_FRACTION=0.05 \
BATCH_SIZE=16 GRADIENT_ACCUMULATION=1 \
RDM_PROJECTIONS=32 RDM_PROJECTION_CHUNK_SIZE=16 AXIS_RDM_FEATURES=32 \
RUN_LOSS_RECOVERED=0 SAEBENCH_CORE_RECONSTRUCTION_BATCHES=2 \
SAEBENCH_CORE_SPARSITY_BATCHES=4 \
RUN_DIR=runs/smoke \
bash scripts/rectified_lpjepa_quickstart.sh
```

既存activationから再開する場合:

```bash
ACTIVATION_MANIFEST=runs/existing/pile-activations/manifest.json \
START_STAGE=2 RUN_DIR=runs/new-dual-code-run \
bash scripts/rectified_lpjepa_quickstart.sh
```

既に学習済みの`RUN_DIR`へSAEBenchと可視化だけを追加する場合:

```bash
START_STAGE=4 END_STAGE=5 RUN_DIR=runs/l16-win4-dual-high \
bash scripts/rectified_lpjepa_quickstart.sh
```

同じ設定の公式Core結果が存在すれば再利用します。評価条件を変更して再計算する
場合は`SAEBENCH_FORCE_RERUN=1`を付けてください。

## Objective

```text
a_i^H = ReLU(E_H(h_i))
z_i^H = TopK(a_i^H, K_high)
z_i^L = TopK(ReLU(E_L(h_i)), K_low)

L = (1-lambda_H) L_full-rec(z^H, z^L)
  + lambda_H L_high-rec(z^H)
  + lambda_inv L_invariance(a_a^H, a_b^H)
  + lambda_rdm (L_random-RDM(a^H) + lambda_axis L_axis-RDM(a^H))
```

## Evaluation

quickstartは以下を一括で評価・可視化します。

1. Top-K SAE指標: FVU、FVE、cosine、L0、dead feature、loss recovered
2. sparse/dense shared-view validity: same-span cosine、shuffled null、距離別margin
3. dense-to-sparse保持率: energy retained、cosine、Top-K saturation fraction
4. high/low分解: ordinary reconstruction、same-span swap、shuffled swap
5. RGG整合: random-projection RDM、axis-aligned RDM、dense active fraction
6. SAEBench Core: explained variance、L0、KL/CE preservation、feature density
7. 任意のSAEBench Sparse Probing（`SAEBENCH_EVALS=core,sparse_probing`）

複数のSAEBench評価は評価ごとに独立したPython processで実行します。これにより、
Pythia-6.9BのCore評価後にGPU上へモデル参照が残っても、次のSparse Probing
開始前にprocess終了によって確実にVRAMを解放します。保存済みの公式結果は
再利用されます。Sparse Probingのresidual activationは各LLM batchの直後に
CPUへ退避し、SAE encode時だけmicrobatchをGPUへ戻します。4000/1000サンプルを
減らさず、24 GiB VRAMで評価できます。

SAEBench公式の推奨どおり、`HIGH_K=64,128,256`など複数のL0で学習した
checkpointを同一設定で比較してください。辞書幅32768で二乗計算量になる
weight-based類似度は既定で無効です。必要な場合のみ
`SAEBENCH_COMPUTE_WEIGHT_METRICS=1`を使用します。無効時の公式出力では、
未計算の`average_max_{encoder,decoder}_cosine_sim`を`-1`と記録し、
feature density、alive fraction、L0などは通常どおり集計します。

swap FVUは距離ごとに `sum(squared error) / sum(centered residual energy)` で
集計します。詳細は
[`docs/RECTIFIED_LPJEPA_PROTOCOL.md`](docs/RECTIFIED_LPJEPA_PROTOCOL.md)を参照してください。

## Artifacts

```text
RUN_DIR/
  pile-activations/manifest.json
  model/rectified_lpjepa_sae.pt
  model/training_report.json
  analysis/rectified_lpjepa_report.json
  analysis/distance_metrics.csv
  saebench/saebench_summary.json
  saebench/core/*.json
  report/index.html
  report/visualization_summary.json
```
