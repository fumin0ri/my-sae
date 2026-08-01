# High/low fixed-endpoint JEPA-SAE

LLMの同一系列にあるresidual trajectoryから、endpointを予測できるhigh-level
sparse featuresと、再構成の細部を補うlow-level sparse featuresを分けて学習します。

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

学習architectureはhigh/low版だけです。unsplit SAE、fixed SAE、position-only学習
モデルはありません。position-onlyとshuffled-contextは、同じ学習済みpredictorへ
入力を変えて作る評価時null controlです。

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

既存cloneでは:

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
| window | 32 token |
| dictionary | 32,768 features |
| total sparsity | Top-K 64 |
| high / low | 20% / 80% |
| training | 12,000 steps |
| SAE-only warm-up | 4,000 steps |
| MMLU | 4,096 questions、question-locked split |
| arithmetic | BF16 autocast + TF32 + fused AdamW |

`WINDOW_SIZE`は任意に変更できます。

```bash
WINDOW_SIZE=128 RUN_DIR=runs/l16-win128 \
bash scripts/transition_jepa_quickstart.sh
```

## 8-stage pipeline

1. document-disjointなPile train/validation residualを抽出
2. high/low full-EMA endpoint JEPA-SAEを学習
3. balanced MMLU probeと因果pairを生成
4. MMLU residual trajectoryを抽出
5. frozen LLMのzero-shot MMLU accuracyを測定
6. 通常のSAE品質、forecast null、semantics/context/syntax probeを評価
7. forecastable high featuresをpatch、ablate、norm-matched random ablate
8. PNG、PDF、CSV、JSON、HTMLレポートを生成

既存のhigh/low checkpointとPile activationはそのまま再利用できます。評価だけ
やり直す場合はstage 3から実行してください。

```bash
START_STAGE=3 RUN_DIR=runs/high-low-jepa-pile \
bash scripts/transition_jepa_quickstart.sh
```

因果評価を省略する場合:

```bash
RUN_CAUSAL=0 START_STAGE=3 \
bash scripts/transition_jepa_quickstart.sh
```

小規模確認:

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 WINDOW_SIZE=16 D_SAE=2048 K=32 PREDICTOR_WIDTH=64 \
PILE_TRAIN_WINDOWS=4096 PILE_VALIDATION_WINDOWS=512 \
PILE_SHARD_WINDOWS=512 TRAIN_STEPS=300 SAE_WARMUP_STEPS=100 \
MMLU_MAX_QUESTIONS=256 PAIRS=8 PAIR_POOL_SIZE=128 \
LOSS_RECOVERED_INPUTS=2 RUN_DIR=runs/smoke \
bash scripts/transition_jepa_quickstart.sh
```

## 評価1: 通常のSAE性能

document-disjoint Pile validation residualについて、最終EMA SAEを評価します。

- reconstruction FVU / fraction of variance explained
- reconstruction cosine、平均L2誤差
- L1、per-position L0、high/low L0
- alive/dead feature fraction
- high-only、low-only、full reconstruction FVE
- original、SAE reconstructed、zero-ablated LLM loss
- fraction of loss recovered

これにより、予測しやすさのためにSAEがresidual情報を捨てていないかを確認します。

## 評価2: 提案手法は本当にcontextを使うか

各context位置 `k<T` からEMA endpoint high codeを予測し、距離別に以下を比較します。

- learned context predictor
- 別MMLU問題からcontext codeを入れるshuffled-context null
- context projectionをゼロにするposition-only null
- predictorなしのraw context-to-endpoint cosine

`learned - shuffled`と`learned - position-only`にはquestion-group bootstrap 95% CIを
付けます。最長horizonで両方のCI下限が0より大きいことを主要な有効性判定とします。

## 評価3: MMLU semantics / context / syntax

MMLU option順とprompt templateを決定論的に均衡化し、問題単位でdevelopmentとlocked
testを分離します。linear probeのaccuracyとbalanced accuracyを次の表現で表示します。

- context high
- predicted endpoint high
- actual endpoint high
- context low
- endpoint low
- endpoint full

軸は以下です。

- `semantics`: 正解選択肢 A/B/C/D
- `context`: STEM / humanities / social sciences / other
- `syntax`: 4種類のprompt template

high表現がcontext/semanticsを保持し、low表現との役割差が生じているかを測ります。

## 評価4: 因果介入

同じcontext category・syntaxで正解だけが異なるMMLU pairを使います。

- `patch`: sourceから予測したhigh codeとtarget予測codeの差をendpointへ追加
- `ablate`: targetの予測可能high componentを除去
- `random_ablate`:同じL2 normのランダム方向を除去

patchのsource-vs-target answer log-prob差、ablationのtarget answer log-prob低下、
norm-matched randomとの差を可視化します。

## Outputs

```text
runs/high-low-jepa-pile/
  pile-activations/
  model/
    transition_jepa_sae.pt
    ema_sae.pt
    training_report.json
  evaluation-data/
    mmlu-prompts.jsonl
    mmlu-causal-pairs.jsonl
  activations/
  analysis/
    transition_jepa_report.json
    transition_horizon_metrics.csv
    mmlu_probe_accuracy.csv
    mmlu_model_accuracy.json
    intervention-*.jsonl
  report/
    index.html
    figures/*.png
    figures/*.pdf
```

最終レポートは`runs/high-low-jepa-pile/report/index.html`です。

研究プロトコルは[`docs/TRANSITION_JEPA_PROTOCOL.md`](docs/TRANSITION_JEPA_PROTOCOL.md)
に固定しています。
