# Predictor-free Rectified LpJEPA-SAE protocol

## Research question

Does a local language sequence contain a sparse residual-stream component that is
shared across token positions, while the remaining sparse component preserves
position-specific information?

The primary object is not future prediction. The experiment treats two positions
from the same sampled span as exchangeable views and learns a high/low SAE:

- high: directly view-invariant, non-negative, sparse, and high-entropy;
- low: position-specific reconstruction increment.

## Prespecified architecture

1. Draw a long residual sequence from the document-disjoint Pile training split.
2. Draw span length `L` uniformly from `[L_min, W_max]`.
3. Draw a valid span end independently of `L`.
4. Draw two distinct ordered positions uniformly without replacement from the span.
5. Encode both residuals with the same online high/low SAE.
6. Apply shifted ReLU to high preactivations and ReLU + Top-K to low preactivations.
7. Reconstruct each residual with the additive high and low decoder partitions.
8. Match the two online high codes with squared L2 invariance.
9. Match each high-code marginal distribution to an i.i.d. Rectified Generalized
   Gaussian product target using sliced two-sample 2-Wasserstein distance.
10. Update a full-SAE EMA after each optimizer step. EMA is never a target in the
    training loss.

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
```

Both invariance and RDMReg are normalized by target-distribution scale. RDMReg
ramps from the first step. Invariance begins after the SAE warm-up and then ramps.

## Primary validation claims

The method passes the representation-validity test only if all of the following
hold on the document-disjoint Pile validation split:

1. EMA same-span high cosine exceeds shuffled-sequence high cosine with a
   bootstrap 95% CI lower bound above zero.
2. The positive-minus-shuffled margin remains positive at the longest adequately
   sampled token distance.
3. Same-span high-code swap reconstruction has lower FVU than shuffled high-code
   swap reconstruction.
4. Learned EMA high active fraction is close to the RGG target without excessive
   dead features.
5. Full EMA SAE reconstruction and loss recovered remain usable relative to the
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

## Secondary evaluations

- Conventional online/EMA SAE FVU, FVE, cosine, L0, and loss recovered.
- High-only and low-only reconstruction.
- Effective rank and collapse diagnostics on memory-bounded selected dimensions.
- MMLU semantics, context, and syntax locked probes.
- EMA high-feature patching, ablation, and norm-matched random ablation.

MMLU is secondary: the core claim concerns shared representation within held-out
Pile spans, not task accuracy.

## Compute

Primary hardware is one RTX 4090 with 23.5 GiB VRAM, CUDA 12.1, and PyTorch
2.5.1. The primary run uses 1024 random projections in chunks of 128. A
projection-count convergence check uses `{256, 512, 1024, 2048}` on held-out
checkpoints; it does not tune on the locked MMLU test.
