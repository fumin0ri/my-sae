# Predictor-free persistent SAE protocol

## Confirmatory question

At a prespecified LLM layer, does a sparse state already present in the first
position of a ten-token residual window persist through each of the following
nine positions?

The confirmatory representation is:

```text
z0 = TopK(ReLU(E_online(h0 - b)))
zj = stopgrad(TopK(ReLU(E_EMA(hj - b)))), j = 1,...,9
```

There is no target average and no learned predictor. The direct persistence
term is:

```text
L_persist = (1/9) sum_j [
    1 - cosine(z0, zj)
    + 0.25 * MSE(z0, zj) / mean(zj^2)
]
```

The total objective adds residual reconstruction and a group-aware
different-window contrast:

```text
L = L_reconstruction + alpha * L_persist + beta * L_contrastive
```

Paraphrases of the same generated problem are positives rather than false
negatives in the contrastive term. The EMA target branch is stop-gradient.

## Prespecified primary outcomes

All outcomes are reported separately for offsets 1 through 9:

1. same-window z0-to-zj cosine;
2. cosine margin over a different-group shuffled-window null;
3. group-bootstrap 95% confidence interval for that margin;
4. same-window versus shuffled ROC AUC;
5. conditional support survival,
   `P(feature active at j | feature active at 0)`;
6. sparse-support Jaccard;
7. same-problem-group retrieval at 1.

The primary comparison is the direct persistent SAE versus an
architecture-, dictionary-, Top-K-, split-, seed-, step-, and
reconstruction-matched standard SAE.

## Secondary outcomes

- locked-group linear probe of generated task state from z0;
- same-problem paraphrase, different-problem same-state, and different-state
  similarities;
- reconstruction FVU;
- active/dead dimensions and variance participation dimension;
- top persistent features and their activating examples;
- residual-space patch, learned ablation, and norm-matched random ablation.

## Leakage control

All paraphrases sharing a generated problem have the same `group_id`.
Train, validation, and locked test are split by `group_id`, never by window.
The locked test is read only by the evaluation command. Hyperparameters should
not be selected from the locked-test report.

## Falsification conditions

The persistence claim is weakened or rejected when any of the following holds:

- same-window cosine does not exceed the shuffled null;
- the offset curve collapses rapidly and is not better than the standard SAE;
- support survival is high only because one or a few global features activate
  in nearly every window;
- effective feature usage collapses or the dead-feature rate is extreme;
- state probes improve but same-state/different-state specificity does not;
- learned ablation is indistinguishable from norm-matched random directions;
- the result fails to replicate across feature seeds, layers, task families,
  or model sizes.

## Recommended replication matrix

After the RTX 4090 single-run quickstart, repeat:

- models: Pythia 1.4B, 2.8B, and 6.9B;
- layers: early-middle, middle, and late-middle prespecified fractions;
- task families: finite-state, arithmetic, and logic;
- feature seeds: at least 0, 1, and 2;
- ablations: persistence without contrastive loss, contrastive without direct
  matching, frozen (non-EMA) target encoder, and the retained JEPA model.

The unit of replication is model x layer x task family x seed, not a single
prompt or token window.
