# Offset-conditioned transition JEPA-SAE protocol

## Confirmatory question

Can joint latent forecasting reshape an SAE dictionary so that a sparse code
at the present position exposes more of the future residual trajectory's
forecastable state than a standard SAE dictionary?

For a prespecified ten-token window:

```text
z0_x = TopK(ReLU(E_online(normalize(h0))))
zk_y = stopgrad(TopK(ReLU(E_EMA(normalize(hk)))))
z_hat_k = softplus(P(z0_x, embedding(k))), k = 1,...,9
```

The online SAE reconstructs all ten positions. The EMA target encoder is
initialized from the online encoder and updated only during joint training.
No target representations are averaged.

## Interpretation boundary

`P(z0, k)` does not receive intervening tokens. It estimates the component of
`zk` that is forecastable from the present state and offset under the training
distribution:

```text
P(z0, k) approximately estimates E[zk | z0, k].
```

It is not claimed to be a deterministic transition operator. The innovation
contains later token information, lexical realization, and other
unforecastable updates.

## Training stages

1. Stream the official 22-component Pile mixture and extract disjoint
   document-level train/validation residual shards.
2. Train one standard Top-K SAE on all ten positions of the Pile windows.
3. Copy its online encoder into the EMA target encoder.
4. Initialize all forecasting conditions from this exact checkpoint and data
   fingerprint.
5. Warm up the predictor with the SAE frozen.
6. For the joint condition only, unfreeze the online encoder and decoder,
   ramp the forecasting weight, and update the target encoder by EMA.

The fixed and offset-only controls receive the same number of predictor
optimizer steps as the joint condition.

The controlled finite-state, arithmetic, and logic prompts are never used for
SAE or predictor optimization. They are generated independently and opened
only for locked evaluation and causal intervention.

## Training corpus

The default corpus is the `all` configuration of
`EleutherAI/the_pile_deduplicated`, pinned to an immutable dataset revision and
streamed from Parquet. It inherits the upstream preweighted 22-component Pile
mixture and receives an additional finite shuffle buffer. The confirmatory
default extracts 524,288 ten-token train windows (5,242,880 residual positions)
and 16,384 validation windows.

Every source document is assigned wholly to train or validation by a
deterministic hash. Activations are stored as BF16 shards. The manifest records
the observed counts for all 22 Pile components, normalization statistics,
model revision, layer, and a data fingerprint. The public deduplicated Parquet
schema contains text but not per-document component labels, so the manifest
explicitly marks source metadata unavailable rather than reporting inferred
component counts. Documents shorter than the model extraction sequence are
right-padded and only their valid ten-token windows are retained, avoiding a
systematic loss of short-document components. The legacy labelled release
remains an opt-in audit path. A run is invalid if any training condition uses a
different fingerprint.

## Objective

```text
L = L_reconstruction
  + lambda_prediction * (
      mean_k[1 - cosine(z_hat_k, zk_y)
             + 0.25 * MSE(z_hat_k, zk_y) / energy(zk_y)]
      + lambda_residual * FVU(decode(TopK(z_hat_k)), hk)
    )
  + lambda_variance * L_variance
```

The predictor output stays dense and non-negative during training. Top-K is
used for support metrics, residual decoding, and causal intervention.

## Confirmatory comparison

The primary comparison is:

```text
joint JEPA-SAE minus fixed standard-SAE predictor
```

The statistic is the per-window mean target-code cosine across offsets 1...9,
with a problem-group bootstrap 95% confidence interval.

The joint claim requires all of the following:

- joint forecasting exceeds the fixed-SAE predictor;
- the true context exceeds a different-group shuffled context;
- the true context exceeds the offset-only predictor;
- reconstruction remains close to the standard SAE;
- the representation does not collapse;
- any causal effect exceeds a norm-matched random control.

## Controls

- standard SAE checkpoint shared by all conditions;
- fixed standard SAE plus the same predictor;
- offset-only predictor with z0 removed;
- shuffled z0 at evaluation;

Future extensions should add an intervening-token-conditioned transition model.

## Locked-test outcomes

For each offset 1...9:

- target-code cosine and normalized MSE;
- true-context minus shuffled-context cosine;
- Top-K support precision, recall, and Jaccard;
- residual prediction FVU;
- innovation-to-target energy ratio;
- predictor and target norms.

Secondary outcomes:

- future task-state probes from z0 and predicted z9;
- paraphrase invariance and state specificity;
- dead-feature and variance-participation diagnostics;
- top forecastable features and activating examples;
- forecastable-component patching and ablation.

## Causal intervention

The actual future code is not replaced. Only the forecastable component is
edited:

```text
delta_hk = D(
    TopK(P(z0_source, k))
    - TopK(P(z0_target, k))
)
```

Ablation removes `D(TopK(P(z0_target, k)))`. Every learned edit is compared
with an independently sampled norm-matched random direction.

## Falsification conditions

The main claim is rejected or weakened if:

- the group-bootstrap interval for joint minus fixed includes zero;
- shuffled z0 performs as well as the corresponding true z0;
- the offset-only model explains the apparent forecasting performance;
- the joint dictionary gains prediction only by materially degrading
  reconstruction;
- forecasted features collapse to position/template shortcuts;
- learned causal edits are indistinguishable from norm-matched random edits;
- results fail across model, layer, task family, or feature seed.

## Replication

The unit of replication is model x layer x task family x feature seed. The
recommended matrix uses Pythia 1.4B, 2.8B, and 6.9B; three prespecified layer
fractions; finite-state, arithmetic, and logic tasks; and at least three seeds.
