# High/low fixed-endpoint JEPA-SAE protocol

## Research question

Can a sparse dictionary be divided into:

- a small high-level group that is reconstructive and forecastable from prior
  residual positions; and
- a larger low-level group that captures incremental reconstruction detail?

The model contains no unsplit SAE condition. For a window of `W` residuals,
`T = W - 1` is the fixed endpoint and every `k < T` is a context.

## Architecture

The dictionary and sparsity budget are partitioned:

```text
d_high + d_low = d_sae
k_high + k_low = k

z_k_high = TopK_high(ReLU(E_high(normalize(h_k))))
z_k_low  = TopK_low(ReLU(E_low(normalize(h_k))))
```

The default split is 20% high and 80% low, following the experiment
configuration in
[AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes).
Independent Top-K operators prevent the low group from being starved by global
feature competition.

The endpoint uses cumulative reconstruction:

```text
h_hat_high = bias + D_high(z_T_high)
h_hat_full = bias + D_high(z_T_high) + D_low(z_T_low)
```

Only high features receive endpoint-prediction supervision:

```text
z_hat_T_high(k) = softplus(P(z_k_high, embedding(k)))
z_T_target = stopgrad(E_EMA_high(h_T))
```

## Training

The online high/low SAE is initialized directly from the Pile activation mean
and scalar RMS. There is no standard or unsplit SAE pretraining.

Training has two phases:

1. SAE-only warm-up: high-only and cumulative full reconstruction.
2. Joint phase: retain reconstruction and ramp the high endpoint-prediction
   objective.

```text
L_rec = alpha * FVU(h_hat_high, h_T)
      + (1-alpha) * FVU(h_hat_full, h_T)

L_pred = mean_k<T [
    1 - cosine(z_hat_T_high(k), z_T_target)
    + 0.25 * normalized_MSE(z_hat_T_high(k), z_T_target)
    + lambda_residual * FVU(
        bias_EMA + D_EMA_high(TopK_high(z_hat_T_high(k))), h_T
      )
]

L = L_rec + lambda_prediction * L_pred
```

The online encoder, decoder, and bias are gradient-trained. Their full
high/low state is EMA-updated. EMA decoder rows are unit-normalized after every
update. The final research artifact and all evaluation use the EMA SAE.

## Data

Training and activation evaluation use deterministic document-disjoint shards
from the pinned `EleutherAI/the_pile_deduplicated` release. The default budgets
are 5,242,880 training and 163,840 validation positions. Holding the position
budget fixed keeps storage approximately invariant to window size.

The default window size is 128 to match the activation context used by the
T-SAE evaluation entrypoint. Window size remains an explicit experimental
variable.

LLM loss recovered streams `monology/pile-uncopyrighted`, matching T-SAE's
`eval_temporal.py` default. Its default maximum context is 2048 tokens and its
batch size is one text, while the temporal activation metrics use 128-position
sequences, matching the two distinct upstream settings.

## Evaluation source of truth

Metric definitions are ported from:

- [`dictionary_learning/evaluation.py`](https://github.com/AI4LIFE-GROUP/temporal-saes/blob/main/dictionary_learning/dictionary_learning/evaluation.py)
- [`dictionary_learning/eval_temporal.py`](https://github.com/AI4LIFE-GROUP/temporal-saes/blob/main/dictionary_learning/dictionary_learning/eval_temporal.py)

The inspected upstream `evaluation.py` blob is pinned as
`0f0deec54f828137d8f637ecc8f12ec9af3a84cc` in each report.

The port keeps the upstream metric names and formulas.

### Reconstruction and sparsity

For final EMA codes `f` and full reconstruction `x_hat`:

```text
l2_loss = mean ||x - x_hat||_2
l1_loss = mean ||f||_1
l0 = mean count(f != 0)
sequence_l0 = mean count(sum_time(f) != 0)
cossim = mean cosine(x, x_hat)
l2_ratio = mean ||x_hat||_2 / ||x||_2
relative_reconstruction_bias = mean ||x_hat||^2 / mean <x, x_hat>
```

`frac_alive` is the fraction of dictionary features with nonzero summed
activation over evaluation.

### High/low reconstruction FVE

As in upstream `recon_splits`, individual high and low reconstructions exclude
the shared decoder bias:

```text
x_hat_high = f_high @ W_high
x_hat_low  = f_low  @ W_low
FVE = 1 - Var(x - x_hat) / Var(x)
```

The report includes full, high-only, and low-only FVE.

### Temporal smoothness

For high and low groups separately:

- `smoothness_tv`: adjacent absolute activation change summed over time and
  features;
- `lipschitz_cont`: for each active feature, maximum adjacent
  `|delta f| / ||delta x||_2`, averaged over active features;
- `fft`: high-frequency energy divided by low-frequency energy;
- `wavelet`: accumulated Haar-like detail energy divided by final
  approximation energy;
- `multiscale`: scale-1 difference variance divided by the coarsest valid
  difference variance.

The total group is also reported for all smoothness metrics except TV, exactly
following upstream output naming.

### LLM loss recovered

Run the frozen LLM three times on each evaluation text:

1. original residual;
2. residual replaced by the final EMA SAE reconstruction;
3. residual replaced by zero.

```text
frac_recovered =
    (loss_reconstructed - loss_zero)
    / (loss_original - loss_zero)
```

This is the same loss-recovered definition used by T-SAE. It is a functional
faithfulness metric, not an endpoint forecast metric.

## Interpretation

The intended high-level separation is supported when the high group is
temporally smoother than the low group while retaining meaningful high-only
FVE and the full SAE recovers model loss. Raw TV depends on group width, so it
must be interpreted together with Lipschitz, spectral, wavelet, multiscale,
and reconstruction metrics.

The claim is weakened if:

- high smoothness is explained only by high-group size or inactivity;
- high-only FVE is negligible;
- low features are equally or more temporally stable across normalized
  metrics;
- full reconstruction cosine/FVE or loss recovered is poor;
- a large fraction of either dictionary is dead;
- results fail to replicate across model, layer, window, and seed.

## Reproducibility

Each report records model revision, layer, hook point, data fingerprint,
window size, group sizes, Top-K budgets, package environment, GPU metadata,
training history, and the exact evaluation metric source URLs.
