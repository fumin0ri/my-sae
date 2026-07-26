# Confirmatory protocol: predictive sparse states in LLM residual streams

## 1. Research question

Does a frozen decoder-only language model maintain a sparse internal state that:

1. can be predicted from earlier residual-stream positions across a nonzero gap;
2. is stable under paraphrase but changes when task state changes;
3. is represented by decoder directions in the original residual space; and
4. causally affects the model's continuation?

This claim is stronger than adjacent residual vectors being similar. Position,
token identity, prompt template, attention sinks, and slowly changing style can
all be predictable without being a computational state.

## 2. Prespecified hypotheses

- **H1 — masked predictability (primary):** On an untouched group-held-out test,
  the joint predictive SAE predicts target sparse codes better than a standard
  SAE whose predictor is trained only after the SAE is frozen.
- **H2 — semantic persistence:** Joint predictable codes have higher similarity
  between paraphrases of the same problem than between different problems with
  the same answer state, while retaining a positive same-state versus
  different-state margin.
- **H3 — predictable / innovation separation:** Predictable codes decode the
  persistent task state; innovation residuals preferentially encode local
  update, boundary, negation, or surprise variables in task families where
  these are annotated.
- **H4 — causal relevance (primary):** Ablating predictable decoder writes
  changes answer log probability more than a norm-matched random residual edit.
  Patching a contradictory source's predictable code moves probability toward
  the source-consistent answer.
- **H5 — reproducibility:** Results replicate across feature-learning seeds,
  model sizes, and task families; matched features or decoder subspaces are
  stable enough to support the same semantic and causal conclusion.

H1 and H4 are primary. Probe accuracy alone is not confirmatory evidence.

## 3. Estimand and model

For frozen residual `h_t`:

```text
z_t = TopK(ReLU(E_online(h_t - b)))
h_reconstructed_t = b + D z_t
```

For a context `C`, an excluded gap `G`, and a future target span `B`:

```text
z_target_B = stopgrad(E_EMA(h_B - b))
z_pred_B   = TopK(ReLU(P(E_online(h_C - b), relative_position(B))))
```

The objective is:

```text
L = normalized_MSE(h, b + D z)
  + alpha * [1 - cosine(z_pred_B, z_target_B)
             + 0.25 * normalized_MSE(z_pred_B, z_target_B)]
  + alpha * beta * normalized_MSE(b + D z_pred_B, h_B)
```

The proposed decomposition is:

```text
predictable_B = b + D z_pred_B
innovation_B  = h_B - predictable_B
```

This is not assumed to be an orthogonal decomposition. Report all three
energies and prediction FVU; do not present their energies as percentages that
must sum to one.

## 4. Leakage-aware masking

The confirmatory mask for a decoder-only LLM is:

```text
past context C | unused gap G | future target B
```

No residual position after `B` may enter `C`: a later decoder residual has
already attended to target tokens and would reveal them. The implementation's
`causal` mode enforces this invariant and its unit test checks it.

`retrospective` mode deliberately uses left and right context. It is an
information-leaking upper-bound ablation and must be labelled as such in every
table and figure.

Prespecified multi-scale grid:

- context width: 16, 24, or 32 depending on model context and memory;
- target span: 2, 4, 8 tokens;
- gap: 2, 4, 8 tokens.

The primary run uses all nine target/gap combinations during training. Report
each combination separately on test. A result limited to gap 2 is consistent
with local continuation prediction and does not establish a persistent state.

## 5. Conditions

All learned conditions use the same activation data, group split, dictionary
width, Top-K, optimizer budget, and random seed where applicable.

1. **Joint predictive SAE (proposed):** reconstruction and prediction train the
   sparse dictionary jointly; target encoder is an EMA copy.
2. **Standard SAE + post-hoc predictor:** train only reconstruction, freeze SAE
   and decoder, then train the same predictor.
3. **Raw residual:** pooled target residual.
4. **Innovation residual:** target residual minus predictable decoder write.
5. **Low-rank random-effects baseline:** previous cross-position generalized
   eigenspace method.
6. **Retrospective JEPA:** optional leakage-positive control.
7. **Shuffled context-target pairing:** required negative control in paper-level
   runs.

Two-stage JEPA followed by an SAE in JEPA latent space may be reported as an
additional representation-learning baseline, but it does not by itself produce
directions in the original LLM residual space and is therefore not a substitute
for the causal condition.

## 6. Data and split

The included controlled benchmark is a four-state cyclic program:

- answer states are balanced;
- every latent program has multiple independently worded paraphrases;
- all paraphrases share `group_id`;
- causal source/target pairs are generated from independent programs;
- the prompt ends before the answer token.

The fixed split is made before optimization:

```text
60% independent groups: training
20% independent groups: validation
20% independent groups: locked test
```

Never split overlapping windows or paraphrases independently. Validation may be
used for early stopping and prespecified hyperparameter selection. Locked test
must not be inspected until the model and analysis choices are frozen.

For a paper-level study, add at least:

- modular arithmetic state tracking;
- symbolic entailment with explicit negation;
- a boundary/update benchmark with annotated stable, update, boundary, and
  surprise positions;
- a representation-negative task that requires no persistent state.

Keep surface form and answer-token distribution matched across state labels.

## 7. Metrics

### H1: prediction

- target-code cosine;
- target-code NRMSE;
- decoded residual prediction FVU;
- performance by gap and target span;
- joint-minus-post-hoc paired difference;
- group-bootstrap 95% confidence interval.

### H2/H3: content

- held-out linear probe on predictable code;
- the same probe on standard SAE code, raw target residual, and innovation;
- same-problem paraphrase cosine;
- different-problem/same-state cosine;
- different-state cosine;
- semantic and paraphrase margins;
- template, task-family, token-identity, position, and length nuisance probes.

Probe regularization must be fixed or selected only on development data.

### Collapse and dictionary health

- reconstruction FVU;
- average L0;
- active and dead feature fractions;
- mean per-feature standard deviation;
- effective rank of centered predictable codes;
- activation frequency distribution;
- top examples for each reported feature.

Reconstruction anchors the dictionary to the original residual stream, but does
not remove the obligation to report collapse diagnostics. EMA is not treated as
a sufficient collapse guarantee.

### H4: causality

- change in target-consistent answer log probability;
- change in source-consistent contrast answer log probability;
- source-minus-target directional patch effect;
- first-token KL;
- intervention L2 norm;
- learned-minus-random paired effect;
- pair-bootstrap 95% CI;
- sign-flip p-value;
- Cohen's `d_z`.

Random residual edits are matched per example and target position to the learned
edit's L2 norm. Report alpha dose-response, not only alpha 1.

## 8. Feature interpretation

For each candidate predictable feature:

1. rank held-out examples by activation;
2. inspect false positives and zero-activation counterexamples;
3. quantify label, template, token, and position selectivity;
4. check stability across paraphrase;
5. locate decoder direction in the original residual space;
6. ablate the feature alone and in matched feature sets;
7. repeat in an independently trained seed.

Feature labels are hypotheses. A natural-language label without quantitative
counterexamples and causal testing is not a result.

## 9. Selection and multiplicity

Prespecify model, layer, hook, target/gap grid, dictionary widths, `k`, loss
weights, seeds, and primary metrics. If layer is selected:

1. select using development groups only;
2. freeze the layer and all settings;
3. evaluate the locked test once.

When testing several model × task × layer families, control the confirmatory
false discovery rate or family-wise error rate. Show each replication unit;
do not report only a pooled number.

## 10. Falsification criteria

The computational-state interpretation is not supported when any of these hold:

- joint training does not improve over the post-hoc SAE control;
- prediction disappears at nontrivial gaps;
- shuffled context-target pairs perform similarly;
- similarity is explained by template, token identity, or absolute position;
- predictable code fails to distinguish same-state from different-state items;
- innovation carries all task-state information;
- dictionary effective rank collapses or most features are dead;
- a norm-matched random intervention has the same effect;
- contradictory patches fail to move the source-versus-target answer contrast;
- results do not replicate across seeds or task families.

## 11. Replication matrix

Minimum serious study:

- Pythia 1.4B, 2.8B, and 6.9B on a single RTX 4090;
- 1 independently trained second model family;
- residual-pre and residual-post;
- prespecified early, middle, and late layers;
- 3 feature-learning seeds;
- 3 task families plus 1 negative-control family;
- at least 300 independent problem groups per task;
- dictionary expansion factors 4×, 8×, and 16× residual width;
- Top-K values selected to match reconstruction quality;
- causal dose response at alpha 0.25, 0.5, 1.0, and 2.0.

The prespecified RTX 4090 primary run uses Pythia-6.9B layer 16, an 8x
dictionary (32,768 features), Top-K 64, a three-layer width-256 predictor,
BF16 autocast, TF32, fused AdamW, and an effective batch of 32. Pythia-12B is
excluded from the single-GPU confirmatory matrix because its BF16 LLM weights
alone consume approximately the full 24GB budget and cannot coexist with the
SAE during causal intervention.

Model × task × seed is the replication unit. Compute intervals by resampling
independent problem groups or causal pairs, never individual overlapping token
windows.

## 12. Required artifacts

```text
activations/layer-*.pt
joint/predictive_sae.pt
joint/training_report.json
posthoc/predictive_sae.pt
posthoc/training_report.json
low-rank-baseline/subspace.pt
analysis/predictive_codes.pt
analysis/predictive_report.json
analysis/intervention-patch.jsonl
analysis/intervention-ablate.jsonl
analysis/intervention-random.jsonl
report/index.html
report/figures/*.png
report/figures/*.pdf
report/visualization_summary.json
```

Archive the code commit, exact model/tokenizer revision, dataset seed, split,
command line, package lock, CUDA/PyTorch versions, GPU type, wall time, and peak
memory with every reported run.
