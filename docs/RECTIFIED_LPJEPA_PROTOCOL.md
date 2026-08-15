# Predictor-free Rectified LpJEPA-SAE protocol

## Research question

Does a local language sequence contain a sparse residual-stream component that is
shared across token positions, while the remaining sparse component preserves
position-specific information?

The primary object is not future prediction. The experiment treats two positions
from the same sampled span as exchangeable views and learns a high/low SAE:

- high: dense ReLU candidates learn view invariance and distribution matching,
  while a Top-K subset is the final sparse SAE code;
- low: position-specific reconstruction increment.

## Prespecified architecture

1. Draw a long residual sequence from the document-disjoint Pile training split.
2. Draw span length `L` uniformly from `[L_min, W_max]`.
3. Draw a valid span end independently of `L`.
4. Draw two distinct ordered positions uniformly without replacement from the span.
5. Encode both residuals with the same high/low SAE.
6. Apply shifted ReLU to obtain dense high candidates, then Top-K those candidates
   for the reconstruction high code. Apply ReLU + Top-K to low preactivations.
7. Reconstruct each residual using only the sparse high and low codes.
8. Match the two dense high candidate codes with squared L2 invariance.
9. Match each dense high marginal distribution to an i.i.d. Rectified Generalized
   Gaussian product target using random-projection sliced two-sample
   2-Wasserstein distance.
10. Also match randomly sampled high coordinates directly to the target's
    coordinate marginals with axis-aligned two-sample 2-Wasserstein distance.

## I/O-amortized training sampling

Training materializes multiple independent random-span pairs while each long
residual sequence shard is resident in CPU memory. The primary settings are:

```text
pairs_per_sequence = 8
pair_shuffle_buffer_pairs = 4096
max_pairs_per_sequence_per_batch = 2
validation_pairs_per_sequence = 1
```

Pairs enter a bounded CPU shuffle buffer. Batch construction takes one pair per
physical sequence first and uses a second pair only when the buffer tail cannot
fill the batch otherwise. Thus a normal batch contains close to `batch_size`
distinct sequences, while each expensive long-sequence shard read supplies up to
eight independently sampled training pairs per sequence. Training logs record the
number of unique sequences and maximum within-batch multiplicity. Validation
retains exactly one pair per sequence and remains document-disjoint.

There is no predictor, position embedding, horizon embedding, target encoder,
stop-gradient target, contrastive negative, variance loss, compatibility loss, or
predicted-residual loss.

## Target distribution

The high target is

```text
Y_j = ReLU(X_j)
X_j iid~ GN_p(mu, sigma)
```

Primary: `p=1` (Rectified Laplace). Control: `p=2` (Rectified Gaussian).
`sigma=0` in the CLI selects the paper's unit pre-rectification variance scale.
`mu` is solved analytically from the requested `target_active_fraction`.

Primary target active fraction: `0.025`. Prespecified sweep:

```text
{0.01, 0.025, 0.05}
```

## Objective

```text
L = (1-lambda_H) L_full-rec
  + lambda_H L_high-rec
  + lambda_inv L_invariance
  + lambda_rdm L_RDMReg
```

Primary weights:

```text
lambda_H   = 0.1
lambda_inv = 1
lambda_rdm = 5
lambda_axis = 1
axis_coordinates = min(512, d_high)
K_high = 128  # default for high_fraction=0.2; use 256 for high_fraction=0.5
K_low = 64
```

Both invariance and RDMReg are normalized by target-distribution scale. The RDM
term is random-projection RDM plus `lambda_axis` times axis-aligned RDM. There is
no SAE warm-up: both terms ramp from the first step with the same regularization
ramp. Learning-rate warm-up remains independent.

## Primary validation claims

The method passes the representation-validity test only if all of the following
hold on the document-disjoint Pile validation split:

1. Sparse Top-K same-span high cosine exceeds shuffled-sequence high cosine with a
   bootstrap 95% CI lower bound above zero.
2. The positive-minus-shuffled margin remains positive at the longest adequately
   sampled token distance.
3. Same-span high-code swap reconstruction has lower FVU than shuffled high-code
   swap reconstruction. Each distance-bin FVU is computed as total squared error
   divided by total centered residual energy, never as a mean of per-row ratios.
4. Sparse high L0 equals `K_high` on nearly every evaluated position. Dense high
   active fraction is monitored separately against the RGG target.
5. Full SAE reconstruction and loss recovered remain usable relative to the
   reconstruction-only ablation.

## Ablations

All ablations use the same activation fingerprint, split seed, initialization seed,
training budget, and evaluation data.

1. Reconstruction-only high/low SAE (`lambda_inv=0`, `lambda_rdm=0`).
2. Invariance only (`lambda_inv>0`, `lambda_rdm=0`) to expose collapse.
3. RDMReg only (`lambda_inv=0`, `lambda_rdm>0`).
4. Full Rectified LpJEPA-SAE.
5. Rectified Laplace (`p=1`) versus Rectified Gaussian (`p=2`).
6. Active-fraction sweep `{0.01, 0.025, 0.05}`.
7. Axis-aligned RDM on versus off (`lambda_axis=1` versus `0`).

## Secondary evaluations

- SAEBench Core on OpenWebText: explained variance, L0, KL/CE preservation,
  shrinkage, feature density, and loss recovered.
- Optional SAEBench Sparse Probing for feature usefulness.
- Dense versus sparse high margin, energy retention, cosine, and Top-K saturation.
- High-only and low-only reconstruction.

SAEBench results must be compared across matched sparsity sweeps because its
metrics are strongly L0-dependent. The primary comparison uses full sparse codes;
high-only and low-only adapters are optional decomposition diagnostics. AutoInterp
is excluded because it requires an external API, and Unlearning is excluded
because its official protocol targets an instruction-tuned Gemma model.

## Compute

Primary hardware is one RTX 4090 with 23.5 GiB VRAM, CUDA 12.1, and PyTorch
2.5.1. The primary run uses 1024 random projections in chunks of 128 and 512
axis-aligned coordinates sampled without replacement per step. A
projection-count convergence check uses `{256, 512, 1024, 2048}` on held-out
checkpoints. SAEBench uses context size 128 and LLM batch size 1 for Pythia-6.9B
on the same 24 GiB GPU.
