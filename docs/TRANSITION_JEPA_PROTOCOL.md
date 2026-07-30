# All-context fixed-endpoint JEPA-SAE protocol

## Confirmatory question

Can joint latent forecasting reshape an SAE dictionary so that every earlier
position exposes a sparse component of one fixed future endpoint that is
forecastable from that position?

For a prespecified window of `W >= 2` residual positions, define `T = W - 1`.

```text
zk_online = TopK(ReLU(E_online(normalize(hk)))), k = 0,...,T
zT_target = stopgrad(TopK(ReLU(E_EMA(normalize(hT))))
z_hat_T_from_k = softplus(P(zk_online, embedding(k))), k = 0,...,T-1
```

Every earlier position is a separate context. All contexts predict the same
EMA endpoint code. Target representations are neither averaged nor pooled,
and the predictor is not autoregressive.

## Interpretation boundary

The predictor does not observe the tokens between `k` and `T`. It estimates
the component of the fixed endpoint that is forecastable under the data
distribution:

```text
P(zk, k) approximately estimates E[zT | zk, k].
```

This is not a deterministic transition operator. The innovation contains
intervening-token information, lexical realization, and other unpredictable
updates. Comparing different `k` values measures how endpoint information
becomes available as context approaches the endpoint.

## Training stages

1. Stream the Pile mixture and extract document-disjoint train/validation
   residual windows.
2. Train one standard Top-K SAE on all positions as a shared initialization.
3. Copy the online encoder, decoder, and normalization bias into a full EMA
   target SAE.
4. Initialize joint, fixed-SAE, and position-only conditions from the exact
   same checkpoint and activation fingerprint.
5. Warm up each predictor with the SAE frozen.
6. For the joint condition, unfreeze the online encoder and decoder, ramp the
   forecasting loss, update the full target SAE by EMA, and row-normalize the
   EMA decoder after every update.

MMLU is never used for SAE or predictor optimization. Its pinned test split is
opened only after training for grouped locked evaluation and causal tests.
Residual extraction and base-model scoring must cover identical stable
question IDs and exclude prompts shorter than `W` real tokens.

## Training corpus

The default is the pinned `default` configuration of
`EleutherAI/the_pile_deduplicated`, streamed from Parquet with an additional
finite shuffle buffer. The default budget is 5,242,880 training and 163,840
validation residual positions. Thus activation storage remains approximately
constant as `W` changes.

Each document is assigned wholly to train or validation by deterministic hash.
BF16 shards use atomic partial-file replacement and a fixed position budget.
The manifest records source configuration, model revision, layer, counts,
normalization statistics, capacity estimate, and a data fingerprint. All
conditions must share that fingerprint. If component labels are absent from
the public Parquet schema, the manifest reports that limitation rather than
inferring labels.

## Full EMA SAE and objective

The online endpoint code reconstructs `hT` through the online decoder. The
online encoder-decoder pair is the gradient-trained student. The EMA encoder,
decoder, and normalization bias form the teacher and final SAE. The EMA
decoder is row-normalized after each EMA update.

Predicted sparse codes are decoded through the frozen EMA decoder. Gradients
therefore propagate from residual prediction to the predictor output, but
never into the EMA SAE. No EMA-code/online-decoder compatibility objective and
no variance regularizer are used.

```text
L = FVU(D_online(zT_online), hT)
  + lambda_prediction * mean_{k<T}[
      1 - cosine(z_hat_T_from_k, zT_target)
      + 0.25 * normalized_MSE(z_hat_T_from_k, zT_target)
      + lambda_residual * FVU(D_EMA(TopK(z_hat_T_from_k)), hT)
    ]
```

The predictor output remains dense and non-negative for the latent regression
loss. Top-K is applied for sparse-support metrics, residual decoding, and
causal interventions. Input-dependent EMA targets, online reconstruction,
latent prediction, and predicted-residual reconstruction provide the
anti-collapse constraints. Collapse statistics remain monitored outcomes.

## Confirmatory comparison

The primary statistic is the per-question mean endpoint-code cosine across all
context positions:

```text
joint JEPA-SAE minus fixed standard-SAE predictor
```

A question-group bootstrap supplies the 95% confidence interval. Horizon-wise
curves remain available and must not be replaced by only the average.

The joint claim requires:

- joint forecasting to exceed the fixed-SAE predictor;
- matching contexts to exceed different-question, matching-position contexts;
- matching contexts to exceed the position-only predictor;
- direct `cosine(zk_EMA, zT_EMA)` to be reported separately from
  predictor performance;
- online-student and final-EMA endpoint reconstruction to remain acceptable;
- representations not to collapse;
- learned causal effects to exceed norm-matched random edits.

## Controls

- one standard SAE checkpoint shared by all conditions;
- frozen standard SAE plus the same position-conditioned predictor;
- position-only predictor with the context projection disabled;
- different-question context trajectories at evaluation;
- raw final-EMA context versus final-EMA endpoint cosine;
- online versus EMA encoding of the exact same endpoint, with separate probe,
  collapse, alignment, and reconstruction diagnostics;
- norm-matched random causal edits.

## Locked-test outcomes

Locked evaluation uses `E_EMA` for every context and endpoint code and uses
`D_EMA` for all predicted-residual metrics. The online SAE is retained only
for the same-endpoint student-versus-final diagnostic.

For each context `k = 0,...,T-1`, report:

- horizon `T-k`;
- raw context-target cosine;
- predicted target-code cosine and normalized MSE;
- matching-context minus shuffled-context cosine;
- Top-K support precision, recall, and Jaccard;
- endpoint residual-prediction FVU and innovation-energy ratio;
- predictor and endpoint-target norms.

Evaluation streams dense tensors batch by batch and retains per-question
scalar horizon statistics. Dense codes are retained only for the prespecified
longest-horizon feature analysis, preventing memory from scaling as
`questions x W x d_sae`.

Secondary outcomes are:

- MMLU semantics accuracy: balanced correct-option decoding;
- context accuracy: official broad MMLU domain decoding;
- syntax accuracy: independently balanced prompt-form decoding;
- frozen base-LLM zero-shot accuracy;
- collapse diagnostics;
- top longest-horizon forecast features and examples;
- endpoint patching, ablation, and norm-matched random controls.

The base score uses the same balanced zero-shot prompts and minimum-token
eligibility rule as representation analysis. It is not the official five-shot
leaderboard protocol.

## Causal intervention

One horizon is prespecified; the default is the longest horizon, `k=0`.
Contexts are encoded by `E_EMA`; the learned forecast component is decoded by
`D_EMA` and applied exactly once at the fixed endpoint residual:

```text
delta_hT = D_EMA(
    TopK(P(z_source_k, k))
    - TopK(P(z_target_k, k))
)
```

Ablation removes `D_EMA(TopK(P(z_target_k, k)))`. The random control uses an
independent direction with the exact learned-edit norm. Source and target
prompts must each have at least `W` real tokens. Patch, ablation, and random
conditions use identical eligible pair IDs.

## Falsification conditions

The main claim is rejected or weakened if:

- the joint-minus-fixed group-bootstrap interval includes zero;
- shuffled or position-only contexts match the true-context result;
- predictor performance is explained entirely by raw code similarity;
- improvement requires materially worse online or EMA-code reconstruction;
- codes collapse to horizon, position, or prompt-template shortcuts;
- learned endpoint edits do not beat norm-matched random edits;
- results fail to replicate across models, layers, split seeds, or feature
  seeds.

## Replication

The replication unit is model x layer x MMLU split seed x feature seed.
Recommended scaling uses Pythia 1.4B, 2.8B, and 6.9B, three prespecified layer
fractions, and at least three seeds. Changing the fixed-endpoint architecture
invalidates old transition checkpoints, but matching Pile activations and the
standard-SAE initialization can be reused by restarting at pipeline stage 6.
