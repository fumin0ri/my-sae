# Predictor-free Rectified LpJEPA-SAE

LLM residual streamから、局所系列内で共有されるhigh-level sparse featureと、
位置固有のlow-level sparse featureを分離して学習する研究パイプラインです。

予測器、horizon embedding、EMA teacher target、in-batch negativesは使いません。
同じランダムspanから異なる2位置を交換可能なviewとしてサンプリングし、同じ
online encoderへ通します。

```text
(h_a, h_b) from one random span
        |          |
        +-- shared online high/low SAE --+
                   |
        high: direct invariance + RDMReg
        low:  position-specific reconstruction
        full: residual reconstruction
```

high codeはshifted ReLU、low codeはReLU + Top-Kです。high codeの分布は、
Rectified Generalized Gaussian (RGG)

```text
ReLU(GN_p(mu, sigma))
```

の独立な積分布へ、random projection上の二標本sliced 2-Wasserstein distanceで
整合させます。これがcollapseを防ぎ、high codeの期待active fractionを制御します。
EMAは損失のteacherではなく、最終的に評価・介入で使うSAE全体の移動平均です。

## Quick start

対象はCUDA 12.1、PyTorch 2.5.1、単一RTX 4090です。

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

既定の本実験:

| item | value |
|---|---:|
| LLM | `EleutherAI/pythia-6.9b-deduped` |
| layer | 16 |
| maximum span | 32 tokens |
| SAE width | 32768 |
| high fraction | 0.2 |
| low Top-K | 64 |
| RGG | Rectified Laplace (`p=1`) |
| target high active fraction | 0.025 |
| RDM random projections | 1024 |
| training | 12000 optimizer steps |
| effective pair batch | 320 |

実行後は次を開きます。

```text
runs/rectified-lpjepa-pile/report/index.html
```

## よく変更する設定

```bash
WINDOW_SIZE=8 \
LAYER=16 \
HIGH_FRACTION=0.5 \
HIGH_RECONSTRUCTION_WEIGHT=0.1 \
TARGET_ACTIVE_FRACTION=0.025 \
RUN_DIR=runs/l16-win8-hf05-rgg-laplace \
bash scripts/rectified_lpjepa_quickstart.sh
```

Rectified Gaussianとの比較:

```bash
RGG_P=2 RUN_DIR=runs/rgg-gaussian \
bash scripts/rectified_lpjepa_quickstart.sh
```

高速スモーク実験:

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 WINDOW_SIZE=8 D_SAE=1024 LOW_K=16 \
PILE_TRAIN_POSITIONS=8192 PILE_VALIDATION_POSITIONS=2048 \
TRAIN_STEPS=20 SAE_WARMUP_STEPS=5 REGULARIZATION_RAMP_STEPS=5 \
BATCH_SIZE=16 GRADIENT_ACCUMULATION=1 \
RDM_PROJECTIONS=32 RDM_PROJECTION_CHUNK_SIZE=16 \
MMLU_MAX_QUESTIONS=64 PAIRS=8 RUN_LOSS_RECOVERED=0 RUN_CAUSAL=0 \
RUN_DIR=runs/smoke \
bash scripts/rectified_lpjepa_quickstart.sh
```

既存activationを再利用してstage 2から再開できます。

```bash
ACTIVATION_MANIFEST=runs/existing/pile-activations/manifest.json \
START_STAGE=2 RUN_DIR=runs/new-rgg-run \
bash scripts/rectified_lpjepa_quickstart.sh
```

## Objective

2 viewを

```text
(z_a^H, z_a^L) = E_online(h_a)
(z_b^H, z_b^L) = E_online(h_b)
```

とすると、損失は

```text
L = (1-lambda_H) L_full-rec
  + lambda_H L_high-rec
  + lambda_inv L_invariance
  + lambda_rdm L_RDMReg
```

です。

- `L_full-rec`: high + lowによる両viewのFVU
- `L_high-rec`: highだけによる両viewのFVU
- `L_invariance`: high code間のtarget-second-moment正規化MSE
- `L_RDMReg`: RGG targetへの正規化sliced 2-Wasserstein

high-only reconstructionは、RDMRegを満たすだけでdecoderから無視される「装飾的な
high code」を防ぎます。low側にはinvarianceもRDMRegも掛けません。

## Evaluation

quickstartは以下を一括で評価・可視化します。

1. 通常のSAE品質
   - online/EMA FVU、FVE、cosine、L0、dead feature
   - high-only/low-only reconstruction
   - next-token loss recovered
2. shared-view validity
   - 同一span high cosine
   - 異なるvalidation sequenceへshuffleしたnull
   - token distance別のpositive-minus-shuffled marginとbootstrap CI
3. high/low分解
   - same-span high codeを交換したswap reconstruction
   - shuffled high codeを交換したnull
4. RGG整合
   - held-out RDMReg
   - high active fractionと理論target active fraction
5. MMLU
   - semantics/context/syntaxのquestion-grouped locked probes
   - high window mean、endpoint high、low、fullの比較
6. causal validity
   - EMA high codeのpatch、ablation、norm-matched random ablation

MMLU表現は二段階で処理し、development splitで分散の大きい最大
`PROBE_MAX_DIM`次元だけをCPUへ保存します。大きな`WINDOW_SIZE`や`D_SAE`でも、
全表現を同時にRAMへ置かない設計です。

詳細な事前登録と判定基準は
[`docs/RECTIFIED_LPJEPA_PROTOCOL.md`](docs/RECTIFIED_LPJEPA_PROTOCOL.md)を参照してください。

## Artifacts

```text
RUN_DIR/
  pile-activations/
    manifest.json
    train/*.pt
    validation/*.pt
  model/
    rectified_lpjepa_sae.pt
    ema_sae.pt
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
