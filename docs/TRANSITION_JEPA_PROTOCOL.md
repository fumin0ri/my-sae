# Offset-conditioned transition JEPA-SAE protocol

## Confirmatory question

Can joint latent forecasting reshape an SAE dictionary so that a sparse code
at the present position exposes more of the future residual trajectory's
forecastable state than a standard SAE dictionary?

For a prespecified window of `W >= 2` token positions:

```text
z0_x = TopK(ReLU(E_online(normalize(h0))))
zk_y = stopgrad(TopK(ReLU(E_EMA(normalize(hk)))))
z_hat_k = softplus(P(z0_x, embedding(k))), k = 1,...,W-1
```

The online SAE reconstructs all W positions. The EMA target encoder is
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
2. Train one standard Top-K SAE on all W positions of the Pile windows.
3. Copy its online encoder into the EMA target encoder.
4. Initialize all forecasting conditions from this exact checkpoint and data
   fingerprint.
5. Warm up the predictor with the SAE frozen.
6. For the joint condition only, unfreeze the online encoder and decoder,
   ramp the forecasting weight, and update the target encoder by EMA.

The fixed and offset-only controls receive the same number of predictor
optimizer steps as the joint condition.

MMLU is never used for SAE or predictor optimization. Its test questions are
loaded from a pinned dataset revision and opened only for locked evaluation and
causal intervention. For a window W, both residual extraction and frozen
base-model scoring use exactly the questions whose rendered prompt contains at
least W real tokens. Short prompts are excluded rather than creating artificial
residual targets with padding. Stable MMLU question IDs are checked before the
locked test is opened.

## Training corpus

The default corpus is the `default` configuration of
`EleutherAI/the_pile_deduplicated`, pinned to an immutable dataset revision and
streamed from Parquet. It inherits the upstream preweighted 22-component Pile
mixture and receives an additional finite shuffle buffer. The confirmatory
default fixes the budget at 5,242,880 train and 163,840 validation residual
positions. At `W=10` these become 524,288 and 16,384 windows; at `W=128` they
become 40,960 and 1,280 windows. Keeping the position budget fixed prevents
activation storage and extraction compute from scaling linearly with W.

Every source document is assigned wholly to train or validation by a
deterministic hash. Activations are stored as BF16 shards with a fixed
40,960-position default shard budget, so one Pythia-6.9B shard remains about
320 MiB for every W. Writes use a same-filesystem partial file followed by an
atomic rename. A capacity preflight includes serialization overhead and a
5 GiB free-space reserve. The manifest records the observed counts for all 22
Pile components, normalization statistics, model revision, layer, resolved
budgets, storage estimate, and a data fingerprint. The public deduplicated
Parquet schema contains text but not per-document component labels, so the
manifest explicitly marks source metadata unavailable rather than reporting
inferred component counts. Documents shorter than the model extraction
sequence are right-padded and only their valid W-token windows are retained,
avoiding a systematic loss of short-document components. The legacy labelled
release remains an opt-in audit path. A run is invalid if any training
condition uses a different fingerprint.

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

The statistic is the per-window mean target-code cosine across offsets
1...W-1,
with a question-group bootstrap 95% confidence interval.

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

For each offset 1...W-1:

- target-code cosine and normalized MSE;
- true-context minus shuffled-context cosine;
- Top-K support precision, recall, and Jaccard;
- residual prediction FVU;
- innovation-to-target energy ratio;
- predictor and target norms.

Evaluation computes these quantities batch by batch and retains only scalar
per-question, per-offset statistics. Dense target, prediction, and shuffled
prediction tensors are not retained across the locked test. Dense codes are
kept only at the final offset for feature analysis.

Secondary outcomes:

- semantics accuracy: linear decoding of the balanced correct option A/B/C/D;
- context accuracy: linear decoding of the official four broad MMLU domains;
- syntax accuracy: linear decoding of four independently balanced prompt forms;
- zero-shot answer accuracy of the frozen base LLM;
- dead-feature and variance-participation diagnostics;
- top forecastable features and activating examples;
- forecastable-component patching and ablation.

The base-model answer score uses the same balanced zero-shot prompt forms as
the representation analysis, applies the same minimum-W-token eligibility
rule, and is not presented as the official five-shot MMLU leaderboard protocol.

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

Both source and target prompts must contain at least W real tokens. The
pipeline deterministically generates an oversized candidate pool, filters it
with the checkpoint tokenizer, and takes the first 128 eligible pairs. Patch,
ablation, and random-control conditions therefore use identical pair IDs.
The default pool contains 2,048 candidates; the run fails before intervention
if it cannot supply all 128 prespecified eligible pairs.

## Falsification conditions

The main claim is rejected or weakened if:

- the group-bootstrap interval for joint minus fixed includes zero;
- shuffled z0 performs as well as the corresponding true z0;
- the offset-only model explains the apparent forecasting performance;
- the joint dictionary gains prediction only by materially degrading
  reconstruction;
- forecasted features collapse to position/template shortcuts;
- learned causal edits are indistinguishable from norm-matched random edits;
- results fail across model, layer, MMLU split seed, or feature seed.

## Replication

The unit of replication is model x layer x MMLU split seed x feature seed. The
recommended matrix uses Pythia 1.4B, 2.8B, and 6.9B; three prespecified layer
fractions; and at least three seeds.
