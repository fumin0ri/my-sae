# Token-shared residual stream

連続する複数トークン位置の residual stream に共通する内部状態を、統計的に抽出し、
SAE で記述し、因果介入で検証するための実験コードです。

結論を先に言うと、最初に試すべき対象は単純な10ベクトルの平均ではなく、
「同じ窓の位置間では再現し、別の窓では変化する低ランク部分空間」です。
平均はその状態の window ごとの座標として使います。

## クイックスタート

以下をそのまま実行すると、公開されている小型モデル
`EleutherAI/pythia-70m-deduped` を使って、データ生成から residual 抽出、
共有部分空間の推定、保持データ評価、permutation control まで実行します。

```bash
git clone https://github.com/fumin0ri/my-sae.git
cd my-sae
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
bash scripts/quickstart.sh
```

GPUがない場合も `sr-fit` は自動的にCPUへfallbackします。明示する場合:

```bash
FIT_DEVICE=cpu bash scripts/quickstart.sh
```

生成物は次に保存されます。

```text
data/quickstart.jsonl
runs/quickstart/activations.pt
runs/quickstart/shared/subspace.pt
runs/quickstart/shared/codes.pt
runs/quickstart/shared/report.json
```

このquickstartは実行経路と統計量を確認する smoke test です。デモデータでは
4種類の working-memory state が文章中に明示されています。したがって、ここで
共有部分空間やprobe精度が得られても「LLMの思考の本質」の証拠にはなりません。
本実験では後述する minimal pair、paraphrase分離、因果patchを使ってください。

## 仮説と推定量

layer `l`、window `w`、相対 token 位置 `t` の residual を

```text
x[w,t] = global_mean + relative_position[t] + z[w] + eps[w,t]
```

と分けます。`z[w]` が10位置に共有された候補状態、`eps[w,t]` が
token 固有成分です。学習データ上で

```text
Sigma_eps = pooled covariance of x[w,t] - mean_t(x[w,t])
Sigma_z   = covariance(mean_t(x[w,t])) - Sigma_eps / T
```

を推定し、次を最大化する一般化固有ベクトルを取ります。

```text
v' Sigma_z v / v' (Sigma_eps + ridge I) v
```

これは「window 間では変わるが、同じ window の10位置には共通する」方向を優先します。
保持データで intraclass correlation (ICC)、位置ごとに別 window へ崩す permutation
control、split-half の部分空間角を計測します。

ただし、この条件だけでは topic、話者、書式、頻出 token の影響も `z[w]` になり得ます。
「LLM が考えていることの本質」と呼ぶには、最低でも次を順に通してください。

1. 異なる prompt・言い換え・window offset でも保持データ上で再現する。
2. label probe が raw mean や last token より汎化する。
3. task state だけを変えた minimal pair 間で subspace patching すると回答確率が動く。
4. 同 rank の random-subspace intervention では同じ効果が出ない。
5. token 順序を window 間で崩した permutation control では信号が消える。

## インストール

上のquickstartを実行済みなら、この節は不要です。

```bash
git clone <this-directory-or-copy-it>
cd shared-residual
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

SAE Lens と TransformerLens も使う場合:

```bash
pip install -e '.[sae]'
```

テスト:

```bash
pip install -e '.[dev]'
pytest -q
```

## 1. データ

一行一 prompt の JSONL を用意します。

```json
{"id":"ex-0001","label":"arithmetic","text":"Question: ... reasoning prefix ..."}
```

`label` は任意です。特定 token span のみを対象にしたい場合は、model tokenizer
で数えた半開区間 `start_token`, `end_token` を追加できます。既定では各 prompt
末尾の10 tokenを1 windowにします。

同じ長文から sliding window を大量に取ると train/test leakage が起きます。
本実装の split は window 単位なので、sliding を使う場合は入力をあらかじめ
文書単位で train/test に分け、別々に実行してください。主要実験は一 prompt
一 window、最低数百、望ましくは数千 window を推奨します。

## 2. Residual stream の抽出

Hugging Face backend:

```bash
sr-extract \
  --model meta-llama/Llama-3.2-3B \
  --data data/prompts.jsonl \
  --output runs/layer12.pt \
  --layer 12 \
  --hook-point post \
  --window-size 10 \
  --batch-size 8
```

`pre` は block 入力、`post` は block 出力です。層を比較するときは同じ hook point
を使います。まず全層の 25%, 50%, 75% 付近を調べ、信号が強い帯を細かく走査するのが
効率的です。

既存 SAE と厳密に合わせる場合は、その SAE を学習した model、hook、dtype、
TransformerLens の設定をそろえてください。

```bash
sr-extract \
  --backend transformer_lens \
  --model gpt2-small \
  --hook-name blocks.6.hook_resid_post \
  --data data/prompts.jsonl \
  --output runs/gpt2-l6.pt \
  --window-size 10
```

## 3. 共通部分空間

```bash
sr-fit \
  --activations runs/layer12.pt \
  --output-dir runs/layer12-shared \
  --rank 32 \
  --ridge 1e-3 \
  --permutations 200 \
  --label-key label
```

出力:

- `subspace.pt`: 直交基底、global mean、相対位置効果。
- `codes.pt`: 各 window の共有座標、raw mean、last-token baseline。
- `report.json`: train/test ICC、permutation p 値、split-half 安定性、probe。

rank は 8, 16, 32, 64 程度を保持データで比較してください。rank を test set に
合わせて選ばず、validation set で選んだ後に未使用 test set を一度だけ評価します。
`--keep-relative-position` は位置効果を残す ablation であり、主解析には勧めません。

## 4. SAE による解釈

### 既存 SAE: token-first を主解析にする

```bash
sr-sae \
  --activations runs/gpt2-l6.pt \
  --subspace runs/gpt2-l6-shared/subspace.pt \
  --release <SAE-Lens release> \
  --sae-id <matching SAE id> \
  --output-dir runs/gpt2-l6-sae \
  --min-active-fraction 0.7 \
  --quantile 0.25
```

二つの順序を同時に保存します。

- **token-first**: 10本を個別に SAE encode し、7割以上の位置で active な feature
  の25 percentile activationを共有 codeとする。
- **mean-first**: 10本を平均してから SAE encode する。

主解析は token-first です。標準 SAE は token residual の分布で学習されるため、
平均 residual は off-manifold になり得ます。`sae_report.json` の token と mean の
reconstruction FVU を必ず比較します。`--subspace` を渡すと、
「反復 activation × SAE decoder direction の共有部分空間への overlap」で
featureを順位付けします。

### 専用の shared/private SAE

```bash
sr-train-group-sae \
  --activations runs/layer12.pt \
  --output-dir runs/layer12-group-sae \
  --d-shared 8192 \
  --d-private 8192 \
  --k-shared 32 \
  --k-private 32 \
  --steps 20000 \
  --batch-size 32
```

これは直接

```text
x[w,t] ≈ bias + D_shared z_shared[w] + D_private z_private[w,t]
```

を学習します。`z_shared` は10位置に一つだけです。private decoder の window 平均に
penalty をかけ、共通情報を private feature が10回重複して持つ解を抑えます。
これは探索的モデルなので、標準 SAE より強い同定仮定を置きます。結果は必ず
通常 SAE と線形 random-effects subspace の双方に照合してください。

## 5. 因果介入

pair JSONL:

```json
{
  "id": "pair-1",
  "source_text": "state A を表す完全な prefix",
  "target_text": "state B を表す完全な prefix",
  "answer": " target B に対する正解"
}
```

source と target の末尾10 tokenから共有座標差を計算し、target の同じ10位置へ
同じ residual shift を加えます。

```bash
sr-intervene \
  --model meta-llama/Llama-3.2-3B \
  --pairs data/pairs.jsonl \
  --subspace runs/layer12-shared/subspace.pt \
  --output runs/patch.jsonl \
  --layer 12 \
  --hook-point post \
  --mode patch
```

Ablation と rank・介入 norm を揃えた random control:

```bash
sr-intervene ... --mode ablate --output runs/ablate.jsonl
sr-intervene ... --mode random_ablate --output runs/random-ablate.jsonl
```

`delta_answer_logprob` が主要指標です。patch 実験では、source state と整合する回答と
target state と整合する回答を別々の pair 群で測ります。効果量は intervention の
L2 norm と rank を random control に一致させて比較してください。

## 推奨する最小実験

最初は自然文の自由な chain-of-thought ではなく、内部状態が明確な課題を使います。

- 同じ surface form で変数値だけが違う modular arithmetic。
- factual question の entity/country だけを置換した minimal pair。
- true/false の仮定を途中で反転する短い deduction。
- 同一問題の paraphrase を train/test で完全分離。

各 prompt は「答えを出す直前」の10 tokenを対象にし、`label` を latent state
（例: modulo class、entity、truth value）にします。そこで再現・probe・patch が
成立してから、自然な reasoning traceへ広げるのが安全です。

## 既知の限界

- 共通**方向**が同じでも位置ごとに係数が違う表現は、完全な additive `z[w]`
  では過小評価します。次段階は `x[w,t] = A[t] z[w] + eps` という multi-view CCA
  または shared response model です。
- attention sink、改行、引用符、system prompt は強い共有信号になります。
  token/format をそろえた minimal pair と shuffle control が必要です。
- 線形 subspace が無いことは、共有状態が無いことを意味しません。非線形符号化や
  position-dependent rotation の可能性があります。
- probe の成功だけでは因果性を示しません。最終判断は patch/ablation です。

## 関連する発想

- Sparse crosscoders は、複数 layer/model にまたがる共有 feature を共同で読む発想を
  提供します: <https://transformer-circuits.pub/2024/crosscoders/index.html>
- Multi-layer SAE は residual stream の layer 間共有性を直接調べています:
  <https://arxiv.org/abs/2409.04185>
- residual の position/context mean-effect 分解は、この実装の nuisance 除去に近い
  出発点です: <https://openreview.net/forum?id=1M0qIxVKf6>
- 低ランク communication channel の実証:
  <https://openreview.net/forum?id=LUsx0chTsL>
- SAE Lens の現行 API:
  <https://decoderesearch.github.io/SAELens/>
