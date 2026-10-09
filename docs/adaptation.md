# Adapting ej to a workflow

A zero-shot model cannot know how often each answer occurs in your workflow. `model.adapt(examples)` learns one logit offset
per (question id, option key) from that workflow's labelled records and returns an `ej.AdaptedModel` that predicts the
same way as the base model, with the offsets added (no fine-tuning; the weights do not change).

```python
labelled = [{**r, 'answers': {'route': 'returns', 'needs_human': 'true', 'urgency': '2'}} for r in my_records[:8]]
adapted = model.adapt(examples=labelled)
adapted.observe(next_labelled_batch)      # fold in labelled records as they arrive
probs = adapted.predict(new_records)
```

- Answers are given by option key (`'answers': {qid: key}`) or by option index in the benchmark format (`'gold': {qid:
  {'label': i}}`). Records without answers add nothing; `model.adapt()` with no examples returns the model itself.
- `observe` refits on all labelled records seen so far: observing in batches gives the same offsets as adapting on all of
  them at once (`tests/test_adapt.py`).
- Offsets are matched by question id and option key: use **one adapted model per workflow**, with a fixed option set.
  Zero-shot `predict` is record-independent; an adapted model's predictions also depend on that workflow's labelled records.
- Cost: one zero-shot pass over the labelled records plus an L-BFGS fit of a few milliseconds (9-19 ms per fit in the
  evaluation below, `adapt_ms`); predicting with an adapted model costs the same as with the base model.

## The method

For a question q with option keys O_q, `p_b(y | x) = softmax_y(log p0(y | x) + b[q, y])`, with `p0` the zero-shot
distribution. The offsets get a Gaussian prior `b ~ N(0, 0.5^2)` and are the maximum a posteriori estimate on the labelled
questions (`ej.adapt_math.fit_tilt`). The prior scale was chosen on development data the model was trained on.

## Measured (ej 0.0.1, development suites)

k = labelled **records** per workflow (every question of a support record is labelled). Group-macro accuracy (mean over
workflows within a source, sources weighted equally). 95% CIs: a t interval over workflows when a suite has 10 or more
workflows, else a record-cluster bootstrap (records resampled, 3 support-set salts averaged inside each draw). The count
prior is the no-model baseline: add-one counts of the labelled answers. Aggregates:
`benchmarks/results/adaptation/ej-0.0.1.few-shot.json`.

| setting | 154 workflows, zs_wide dev (111 public-source + 43 GLM-synthetic) | 1 workflow, zs_td dev (selected on) | 3 seen workflows, td dev |
|---|---|---|---|
| k = 8 vs zero-shot | .356 → .389, **+.032 [−.011, +.076]: covers 0** (153 workflows; about 16.6 answers) | .396 → .530, +.134 [+.108, +.160] (40 answers) | +.008 [−.003, +.020]: covers 0 (40 answers) |
| k = 16 vs zero-shot | not measured: these workflows have at most 12 records | .396 → .545, +.149 [+.118, +.183] (80 answers) | +.016 [+.003, +.029] (80 answers) |
| k-shot vs the count prior | +.008 [−.0001, +.016] at k = 8: covers 0 | +.004 [+.001, +.007] at k = 16 | +.028 [+.012, +.046] at k = 16 |

How to read it: on unseen workflows what the labelled records teach is mostly the workflow's **label prior**; the add-one
count prior alone performs almost as well (k = 8 on zs_wide: within +.008, CI covering 0). The zs_wide figures weight the 43
GLM-synthetic workflows as one source of four, and their gold labels have not been independently checked. Labelled examples
also lower the log loss: zs_wide −.091 [−.112, −.071] NLL at k = 8, zs_td −.175 [−.194, −.156] at k = 16. zs_td is one
workflow that drove model selection, so its gains are not unseen-workflow evidence.

## Not included: label-free adaptation

A label-free variant (batch-calibration offsets from unlabelled requests, with shrinkage and a prior on the bias share) was
evaluated with the same protocol. Its gain over zero-shot was not distinguishable from 0 on any suite: zs_wide +.009
[−.014, +.031], zs_td +.011 [−.002, +.025], td +.010 [−.007, +.027]. It is not part of the package.
