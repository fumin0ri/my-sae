# Research protocol: token-shared residual representations

## Research question

Does a decoder-only language model maintain a low-dimensional latent state that
is simultaneously readable at several nearby token positions, varies across
reasoning instances, predicts the relevant task state, and causally affects the
model's answer?

The claim is deliberately stronger than "nearby residual vectors are similar."
Token identity, position, formatting, topic, and prompt template can all produce
similarity without constituting a computational state.

## Confirmatory hypotheses

- **H1 — cross-position reliability:** a learned shared subspace has positive
  held-out ICC and exceeds the position-shuffled null on a locked outer test.
- **H2 — representational specificity:** a rank-matched probe on shared
  coordinates predicts the latent task state at least as well as window-mean PCA
  and last-token PCA.
- **H3 — reproducibility:** independently fitted halves recover overlapping
  subspaces, measured by mean squared canonical correlation.
- **H4 — causality:** ablating the learned subspace changes correct-answer log
  probability more than a norm- and rank-matched random subspace. Patching a
  contradictory source state into a target decreases the target-consistent
  answer probability.

H1 and H4 are the primary hypotheses. H2 and H3 characterize a positive result
but cannot establish causality alone.

## Controlled benchmark

`scripts/make_research_data.py` generates four-state cyclic programs.

- The answer state is balanced across problems.
- Each latent program receives multiple independently worded paraphrases.
- All paraphrases of a program share `group_id`.
- Splitting is performed by `group_id`, never by prompt row.
- Causal source/target pairs are generated from new programs that do not occur
  in the representation-learning dataset.

For a paper-level experiment, add at least two further task families, such as
modular arithmetic and symbolic entailment, without changing the analysis code.
Do not choose tasks after inspecting the locked test.

## Estimand

For window `w`, relative position `t`, and residual width `d`:

```text
x[w,t] = μ + position[t] + z[w] + ε[w,t]
```

The estimator removes `μ` and the relative-position mean effect using training
data only. It estimates:

```text
Σ_ε = pooled within-window covariance
Σ_z = Cov(mean_t x[w,t]) - Σ_ε / T
```

Directions maximize:

```text
vᵀ Σ_z v / vᵀ(Σ_ε + ridge I)v
```

The window state coordinate is the projection of the centered window mean onto
the learned basis. The mean is therefore a per-window coordinate estimator, not
the definition of the shared subspace.

For wide models, the implementation optionally performs a training-only PCA
preprojection before the generalized eigendecomposition (`--pre-rank`, default
512). The retained variance is recorded. Confirm important results at multiple
pre-ranks, including a no-preprojection run when computationally feasible.

## Selection and locked evaluation

1. Hold out 20% of independent problem groups as the outer test.
2. On the remaining development groups, repeat stratified group train/validation
   splits over prespecified seeds.
3. Sweep layer, window width, rank, and ridge.
4. Select by mean validation `(ICC - shuffled-null ICC)`.
5. Fit the selected configuration once on all development groups.
6. Evaluate once on the untouched outer groups.
7. After recording the locked result, fit an all-data basis for downstream
   causal intervention. Do not report its in-sample metrics as test evidence.

The pipeline writes every candidate result, so selection can be audited.

## Prespecified metrics

Primary representation metric:

- mean held-out ICC across retained shared components;
- position-shuffled permutation p-value;
- problem-group bootstrap 95% confidence interval.

Secondary metrics:

- state-probe accuracy;
- rank-matched window-mean PCA accuracy;
- rank-matched last-token PCA accuracy;
- split-half mean squared canonical correlation;
- generalized eigenvalue spectrum.

Primary causal metric:

- paired difference in answer-log-probability change between learned and random
  ablation;
- contradictory-patch directionality, defined as the source-consistent answer
  change minus the target-consistent answer change;
- problem-pair bootstrap 95% confidence interval;
- paired sign-flip p-value;
- Cohen's `d_z`.

## Falsification and controls

A result does not support the shared-computational-state interpretation when any
of the following holds:

- observed ICC is indistinguishable from position-shuffled ICC;
- the effect disappears under paraphrase-group splitting;
- only the raw mean or last token decodes the state;
- split-half subspaces are unstable;
- learned ablation is not stronger than norm-matched random ablation;
- patch direction fails to move answer probability toward the source state;
- the signal is confined to punctuation, padding, BOS/EOS, or a single template.

Recommended additional controls:

- randomize token order inside windows while preserving token marginals;
- match prompt length and final-token strings across labels;
- regress template, absolute position, and sequence length;
- repeat with an unrelated label;
- compare residual-pre and residual-post;
- repeat with non-overlapping windows;
- test a representation-negative task where no persistent state is needed.

## Replication matrix

The minimal research quickstart validates the pipeline on Pythia-70M. A serious
study should prespecify:

- at least three model sizes from one family;
- at least one independently trained model family;
- early, middle, and late layers, preferably all layers when feasible;
- window widths such as 4, 8, 10, 16, and 32;
- ranks such as 2, 4, 8, 16, and 32;
- at least three inner split seeds;
- at least three task families;
- a minimum of several hundred independent problem groups per task.

Treat model × task as the replication unit. Correct confirmatory p-values across
that prespecified family, and show individual effects rather than only a pooled
average.

## Artifact checklist

The analysis is complete only when the run contains:

```text
activations/layer-*.pt
analysis/candidates.jsonl
analysis/selection.json
analysis/locked_test.json
analysis/final_subspace.pt
analysis/final_codes.pt
analysis/intervention-*.jsonl
report/index.html
report/figures/*.png
report/figures/*.pdf
report/visualization_summary.json
```

Archive the exact code commit, model revision, tokenizer revision, dataset seed,
GPU/software environment, and command line with the result.
