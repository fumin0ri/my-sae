# High/low fixed-endpoint JEPA-SAE protocol

## Research question

Does the high partition of a sparse residual representation contain
context-dependent information that forecasts a fixed future endpoint under the
training-matched online-to-EMA path, while both the online and EMA high/low SAE
pairs remain competitive conventional autoencoders?

Temporal uniqueness or smoothness is not a primary claim. TV, FFT, wavelet,
Lipschitz, and multiscale smoothness are therefore not confirmatory endpoints.

## Model

For a residual window `h_0 ... h_T`, one online SAE is divided into independent
high and low Top-K groups. The high group receives both high-only reconstruction
and endpoint prediction supervision. The low group receives incremental full
reconstruction supervision. Encoder, decoder, and bias are copied into a full
EMA teacher after every optimizer step. The EMA pair is the final artifact; the
online pair is retained as the training-matched student and as a conventional-SAE
comparison.

The loss is:

```text
L_rec = alpha * FVU(D_high(z_T_high), h_T)
      + (1-alpha) * FVU(D_high(z_T_high)+D_low(z_T_low), h_T)

L = L_rec + lambda_pred * (
      latent endpoint prediction loss
      + lambda_res * predicted-high-residual FVU
    )
```

There is no unsplit architecture, standard-SAE pretraining condition, fixed-SAE
condition, or separately trained position-only model.

## Data separation

Training and conventional SAE validation use deterministic document-disjoint
shards from the pinned `EleutherAI/the_pile_deduplicated` release. MMLU is used
only after training. MMLU rows are split by `question_id` before probe fitting;
the locked test is never used for feature selection or classifier fitting.

## Confirmatory evaluation A: conventional SAE quality

Evaluate the online and final EMA encoder-decoder pairs on exactly the same
held-out Pile residuals:

- reconstruction FVU and fraction of variance explained;
- reconstruction cosine and mean L2 error;
- L1, per-position total/high/low L0;
- alive and dead feature fraction;
- high-only, low-only, and full reconstruction FVE;
- fraction of downstream LLM loss recovered relative to zero ablation, for
  online and EMA reconstruction separately;
- online-to-EMA code cosine and support Jaccard as drift diagnostics.

These metrics test whether predictive supervision caused representation loss or
dictionary collapse.

## Confirmatory evaluation B: forecast validity

For every context position, the primary evaluation exactly matches the trained
student/teacher path:

```text
P(E_online(h_k), k) -> E_EMA(h_T)
```

On the question-locked MMLU test compare:

1. the learned predictor with the correct context;
2. the same predictor with a context from another question;
3. the same predictor with its context projection zeroed, retaining only the
   position embedding;
4. the raw online context-high versus EMA endpoint-high cosine.

Primary paired effects are:

```text
gain_shuffled = cosine(pred(correct context), target)
              - cosine(pred(other question), target)

gain_position = cosine(pred(correct context), target)
              - cosine(pred(position only), target)
```

Report question-cluster bootstrap 95% confidence intervals at every horizon.
The primary horizon is the longest (`k=0`). The core forecasting claim passes
only if both primary confidence-interval lower bounds exceed zero. Code NRMSE,
support precision/recall/Jaccard, and predicted-residual FVU are secondary.

Also report `P(E_EMA(h_k), k) -> E_EMA(h_T)` with matching shuffled and
position-only controls as a secondary compatibility analysis. This path was not
used to train the predictor and cannot replace the Online-matched primary test.

This within-checkpoint design isolates use of context without reintroducing an
unsplit or separately trained control architecture.

## Confirmatory evaluation C: MMLU representation probes

Use deterministic balancing of answer position and one of four syntax templates.
Fit regularized linear probes on development questions and report locked-test
accuracy, balanced accuracy, chance accuracy, and group-bootstrap intervals for:

- semantics: correct answer A/B/C/D;
- context: STEM, humanities, social sciences, other;
- syntax: four rendering templates.

Probe Online-matched context/prediction representations, EMA-context
compatibility representations, the actual EMA endpoint high code, online
context low code, and EMA endpoint low/full codes. Probe dimensions are selected
by development-set variance only.

Probe accuracy establishes decodability, not causal use. It must be interpreted
with the null-control and intervention results.

## Confirmatory evaluation D: causal interventions

Construct MMLU pairs matched on broad context and syntax but differing in answer.
At the fixed endpoint, use the online context encoder and EMA target/decoder:

- patch the difference between source and target forecastable high residuals;
- ablate the target forecastable high residual;
- apply a random direction with matched L2 norm.

Report target-answer and contrast-answer log-probability changes, output KL, and
intervention norm. The method is supported when learned edits are directional
and exceed the norm-matched random control.

## Decision table

| Question | Required evidence |
|---|---|
| Is it a usable SAE? | online and EMA full FVE/cosine, loss recovered, and dead-feature rate on matched residuals |
| Does prediction use context? | Online-matched longest-horizon gain over shuffled and position-only has CI lower bound above zero |
| What is encoded? | locked MMLU semantics/context/syntax probes, interpreted comparatively across high/low |
| Is the forecastable component causally relevant? | directional patch/ablation effect beyond matched random direction |

Failure on conventional reconstruction invalidates representation conclusions.
Probe success without null-control or causal success is descriptive only.

## Reproducibility

Every run stores the code commit, Python environment, GPU/driver information,
pinned model and dataset revisions, activation-manifest fingerprint, split seed,
and machine-readable JSON/CSV outputs. `WINDOW_SIZE` is a prespecified variable;
changing it requires a distinct `RUN_DIR`.
