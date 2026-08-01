# High/low fixed-endpoint JEPA-SAE

LLMの同一系列にあるresidual trajectoryから、将来endpointを予測できる
high-level sparse featuresと、再構成の細部を補うlow-level sparse featuresを
分けて学習します。

```text
residual h_k
    │
    ├─ E_high ─ z_k^high ─ predictor(z_k^high, k) ─ z_T^high
    │                         endpoint JEPA supervision
    │
    └─ E_low  ─ z_k^low
              reconstruction detail only

h_T ≈ bias + D_high(z_T^high) + D_low(z_T^low)
```

high/low分割は
[AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)
のTemporal Matryoshka SAEを参考にしています。辞書の20%をhigh、80%をlowへ
割り当て、独立したTop-K budgetを使います。

現在の実験にはunsplit SAE、fixed SAE、position-only、MMLU probe、独自の
因果介入評価は含まれません。学習・評価対象はhigh/low SAEだけです。

## Quickstart

CUDA 12.1、`torch==2.5.1+cu121`、単一RTX 4090向けです。

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

既存cloneでは次だけで実行できます。

```bash
git pull origin main
python -m pip install --upgrade -e .
bash scripts/transition_jepa_quickstart.sh
```

既定設定:

| 項目 | 値 |
|---|---:|
| frozen LLM | `EleutherAI/pythia-6.9b-deduped` |
| layer | 16 |
| evaluation-compatible window | 128 positions |
| dictionary | 32,768 features |
| total sparsity | Top-K 64 |
| high / low | 20% / 80% |
| training | 12,000 steps |
| SAE-only warm-up | 4,000 steps |
| arithmetic | BF16 autocast + TF32 + fused AdamW |

`WINDOW_SIZE`は変更できます。ただしT-SAE repoの既定評価contextと揃えるため、
本repoの既定値も128です。

```bash
WINDOW_SIZE=32 RUN_DIR=runs/w32 \
bash scripts/transition_jepa_quickstart.sh
```

## Pipeline

quickstartは4 stageだけです。

1. document-disjointなPile train/validation residual windowを抽出
2. high/low full-EMA endpoint JEPA-SAEを直接学習
3. T-SAE互換指標とLLM loss recoveredを評価
4. PNG、PDF、CSV、JSON、HTMLを生成

完了済みartifactから再開できます。

```bash
# 評価から再開
START_STAGE=3 RUN_DIR=runs/high-low-jepa-pile \
bash scripts/transition_jepa_quickstart.sh

# 可視化だけ再生成
START_STAGE=4 END_STAGE=4 RUN_DIR=runs/high-low-jepa-pile \
bash scripts/transition_jepa_quickstart.sh
```

旧versionの`joint/`, `fixed/`, `k_only/`, `standard/` checkpointは新しい
architectureと互換性がありません。Pile activation shardだけはwindow、model、
layerが同じなら再利用できます。新しいmodelはstage 2から学習してください。

## Training objective

online SAE全体を勾配更新し、encoder、decoder、biasをEMA SAEへ更新します。
評価と最終artifactにはEMA SAEだけを使います。

```text
L_rec = alpha * FVU(D_high(z_T^high), h_T)
      + (1-alpha) * FVU(
          D_high(z_T^high) + D_low(z_T^low), h_T
        )

L = L_rec
  + lambda_prediction * mean_k<T [
      1 - cosine(z_hat_T^high(k), z_T,EMA^high)
      + 0.25 * normalized_MSE
      + lambda_residual * FVU(
          D_high,EMA(TopK(z_hat_T^high(k))), h_T
        )
    ]
```

最初の`SAE_WARMUP_STEPS`では`lambda_prediction=0`です。その後JEPA lossを
徐々に立ち上げます。unsplit standard SAEによる事前学習は行いません。

## T-SAE-compatible evaluation

評価式は上流の
[`dictionary_learning/evaluation.py`](https://github.com/AI4LIFE-GROUP/temporal-saes/blob/main/dictionary_learning/dictionary_learning/evaluation.py)
を移植しています。最終EMA SAEについて次を計算します。

- `l2_loss`, `l1_loss`, per-position `l0`, `sequence_l0`
- high/low total variation
- total/high/low Lipschitz continuity
- total/high/low FFT high/low frequency energy ratio
- total/high/low wavelet detail/approximation ratio
- total/high/low multiscale fine/coarse variation ratio
- total/high-only/low-only fraction of variance explained
- reconstruction cosine、L2 ratio、relative reconstruction bias
- alive feature fraction
- original / SAE-reconstructed / zero-ablated LLM loss
- fraction of loss recovered

activation指標はdocument-disjoint Pile validation shardで計算します。
loss recoveredは上流と同じ`monology/pile-uncopyrighted`を既定datasetとして、
LLM residualをEMA SAE再構成で置換して測定します。

loss recoveredを一時的に省略する場合:

```bash
RUN_LOSS_RECOVERED=0 START_STAGE=3 \
bash scripts/transition_jepa_quickstart.sh
```

評価数は変更できます。

```bash
LOSS_RECOVERED_INPUTS=128 \
LOSS_RECOVERED_CONTEXT_LENGTH=2048 \
START_STAGE=3 \
bash scripts/transition_jepa_quickstart.sh
```

## Outputs

```text
runs/high-low-jepa-pile/
  pile-activations/
    manifest.json
    train/shard-*.pt
    validation/shard-*.pt
  model/
    transition_jepa_sae.pt
    ema_sae.pt
    training_report.json
  analysis/
    temporal_sae_eval.json
    temporal_sae_metrics.csv
  report/
    index.html
    figures/*.png
    figures/*.pdf
```

最終レポート:

```text
runs/high-low-jepa-pile/report/index.html
```

## Smoke test

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 \
WINDOW_SIZE=16 \
D_SAE=2048 \
K=32 \
PREDICTOR_WIDTH=64 \
PILE_TRAIN_WINDOWS=4096 \
PILE_VALIDATION_WINDOWS=512 \
PILE_SHARD_WINDOWS=512 \
TRAIN_STEPS=300 \
SAE_WARMUP_STEPS=100 \
LOSS_RECOVERED_INPUTS=2 \
PILE_EXTRACT_BATCH_SIZE=32 \
RUN_DIR=runs/smoke \
bash scripts/transition_jepa_quickstart.sh
```

研究プロトコルは
[`docs/TRANSITION_JEPA_PROTOCOL.md`](docs/TRANSITION_JEPA_PROTOCOL.md)
に固定しています。
