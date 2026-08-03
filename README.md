# Predictor-free Rectified LpJEPA-SAE

LLMのresidual streamから、近接token位置で共有されるhigh-level sparse
featureと、位置固有のlow-level sparse featureを分けて学習する研究用pipelineです。
predictor、horizon embedding、teacher encoder、in-batch negativesは使いません。

```text
(h_a, h_b) from one random span
        |          |
        +-- shared high/low SAE --+
                   |
        high: direct invariance + random/axis RDMReg
        low:  position-specific reconstruction
        full: residual reconstruction
```

high codeはshifted ReLU、low codeはReLU + Top-Kです。high codeの経験分布を
Rectified Generalized Gaussian (RGG) targetへ合わせます。既存のrandom-projection
sliced 2-Wassersteinに加え、座標軸上の分布を直接合わせるaxis-aligned RDMRegを
使用します。

## Quick start

対象環境はCUDA 12.1、PyTorch 2.5.1、単一RTX 4090です。

```bash
git clone https://github.com/fumin0ri/my-sae.git
cd my-sae
conda create -n sae python=3.11 -y
conda activate sae
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision torchaudio
pip install -e .
HF_TOKEN=... bash scripts/rectified_lpjepa_quickstart.sh
```

主な既定値はPythia 6.9B、layer 16、maximum span 32、SAE width 32768、
high fraction 0.2、low Top-K 64、1024 random projections、512 axis coordinates、
12000 optimizer stepsです。実行後は次を開きます。

```text
runs/rectified-lpjepa-pile/report/index.html
```

主要設定の変更例:

```bash
WINDOW_SIZE=8 \
LAYER=16 \
HIGH_FRACTION=0.5 \
HIGH_RECONSTRUCTION_WEIGHT=0.1 \
TARGET_ACTIVE_FRACTION=0.025 \
AXIS_RDM_FEATURES=512 \
AXIS_RDM_WEIGHT=1 \
RUN_DIR=runs/l16-win8-hf05-rgg-laplace \
bash scripts/rectified_lpjepa_quickstart.sh
```

`AXIS_RDM_WEIGHT=0`でaxis-aligned項だけを無効化できます。

軽量smoke test:

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 WINDOW_SIZE=8 D_SAE=1024 LOW_K=16 \
PILE_TRAIN_POSITIONS=8192 PILE_VALIDATION_POSITIONS=2048 \
TRAIN_STEPS=20 SAE_WARMUP_STEPS=5 REGULARIZATION_RAMP_STEPS=5 \
BATCH_SIZE=16 GRADIENT_ACCUMULATION=1 \
RDM_PROJECTIONS=32 RDM_PROJECTION_CHUNK_SIZE=16 AXIS_RDM_FEATURES=32 \
MMLU_MAX_QUESTIONS=64 PAIRS=8 RUN_LOSS_RECOVERED=0 RUN_CAUSAL=0 \
RUN_DIR=runs/smoke \
bash scripts/rectified_lpjepa_quickstart.sh
```

既存activationから再開する場合:

```bash
ACTIVATION_MANIFEST=runs/existing/pile-activations/manifest.json \
START_STAGE=2 RUN_DIR=runs/new-rgg-run \
bash scripts/rectified_lpjepa_quickstart.sh
```

## Objective

```text
(z_a^H, z_a^L) = E(h_a)
(z_b^H, z_b^L) = E(h_b)

L = (1-lambda_H) L_full-rec
  + lambda_H L_high-rec
  + lambda_inv L_invariance
  + lambda_rdm (L_random-RDM + lambda_axis L_axis-RDM)
```

- `L_full-rec`: high + lowによる両viewのFVU
- `L_high-rec`: highだけによる両viewのFVU
- `L_invariance`: high code間のtarget-second-moment正規化MSE
- `L_random-RDM`: random projections上の正規化sliced 2-Wasserstein
- `L_axis-RDM`: サンプルしたhigh座標上の正規化1次元2-Wasserstein

## Evaluation

quickstartは以下を一括で評価・可視化します。

1. 通常のSAE指標: FVU、FVE、cosine、L0、dead feature、loss recovered
2. shared-view validity: same-span cosine、shuffled null、距離別marginとCI
3. high/low分解: ordinary reconstruction、same-span swap、shuffled swap
4. RGG整合: random-projection RDM、axis-aligned RDM、active fraction
5. MMLU: semantics/context/syntaxのquestion-grouped locked probes
6. causal validity: high codeのpatch、ablation、norm-matched random ablation

swap FVUは距離ごとに `sum(squared error) / sum(centered residual energy)` で
集計します。詳細な事前登録と判定基準は
[`docs/RECTIFIED_LPJEPA_PROTOCOL.md`](docs/RECTIFIED_LPJEPA_PROTOCOL.md)を参照してください。

## Artifacts

```text
RUN_DIR/
  pile-activations/manifest.json
  model/
    rectified_lpjepa_sae.pt
    training_report.json
  analysis/
    rectified_lpjepa_report.json
    distance_metrics.csv
    mmlu_probe_accuracy.csv
    evaluation_embeddings.pt
    intervention-*.jsonl
  report/
    index.html
    visualization_summary.json
    figures/*.png
    figures/*.pdf
```
