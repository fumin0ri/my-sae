# Offset-conditioned Transition JEPA-SAE

Frozen autoregressive LLMの10-token residual trajectoryから、現在位置で
すでに予測可能な未来の内部状態を疎な辞書として抽出する研究コードです。

```text
h₀ ─ online Top-K SAE ─ z₀ ─┐
                              ├─ offset-conditioned MLP ─ softplus ─ ẑₖ
offset embedding(k) ──────────┘

hₖ ─ EMA target SAE ─ stopgrad(zₖ),  k = 1,...,9
```

`z₀`とoffset `k`だけから、`z₁...z₉`を個別に予測します。targetの平均化や
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
| standard SAE | 12,000 steps |
| JEPA conditions | 各8,000 steps |
| arithmetic | BF16 autocast + TF32 + fused AdamW |

結果は次の自己完結HTMLに出ます。

```text
runs/transition-jepa-research/report/index.html
```

## 軽量smoke test

```bash
MODEL=EleutherAI/pythia-70m-deduped \
LAYER=3 \
D_SAE=2048 \
K=32 \
PREDICTOR_WIDTH=64 \
STANDARD_STEPS=300 \
FORECAST_STEPS=300 \
PREDICTOR_WARMUP_STEPS=50 \
PROBLEMS=40 \
EXTRACT_BATCH_SIZE=32 \
RUN_CAUSAL=0 \
bash scripts/transition_jepa_quickstart.sh
```

## 実験条件

すべて同じreconstruction-only standard SAE checkpointから開始します。

- `joint`: predictor warm-up後、predictor・online encoder・decoderを共同学習
- `fixed`: standard SAEを固定し、同じpredictorだけ学習
- `k_only`: `z₀`を遮断し、offsetだけから予測
- shuffled context: locked testで別problem groupの`z₀`へ交換

online SAEは10位置すべてを再構成します。EMA target encoderはjoint条件だけで
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

locked problem-group testで次を出力します。

- offset 1...9のcode cosine、normalized MSE、support precision/recall/Jaccard
- true-context minus shuffled-context
- joint minus fixedのproblem-group bootstrap 95% CI
- residual prediction FVUとinnovation energy
- task-state probe、paraphrase invariance、collapse診断
- top forecastable featureと活性化例
- forecastable componentだけのpatch・ablation・norm-matched random対照
- PNG、PDF、CSV、JSON、自己完結HTML

因果patchは実際の未来code全体を置換せず、予測可能成分だけを編集します。

```text
Δhₖ = D(TopK(P(z₀_source,k)) - TopK(P(z₀_target,k)))
```

## Replication

task family、seed、出力先を環境変数で分離できます。

```bash
TASK_FAMILY=logic \
SEED=1 \
SPLIT_SEED=1 \
RUN_DIR=runs/transition-jepa-logic-seed1 \
bash scripts/transition_jepa_quickstart.sh
```

`TASK_FAMILY`は`fsm`、`arithmetic`、`logic`を選べます。モデル・層・task
family・feature seedを独立したreplication unitとして扱ってください。

## コード構成

```text
src/shared_residual/
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
  make_research_data.py

docs/
  TRANSITION_JEPA_PROTOCOL.md
```

主仮説、confirmatory comparison、棄却条件、replication方針は
[`docs/TRANSITION_JEPA_PROTOCOL.md`](docs/TRANSITION_JEPA_PROTOCOL.md)に
固定しています。
