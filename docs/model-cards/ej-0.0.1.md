# ej model card (0.0.1)

> **Weights: not yet published** (no hosting has been decided yet). The model is one file, `model.ejpack` (11,384,312
> bytes); `ej.load` needs a local copy of it. `python -m ej.train fit` and `python -m ej.train export` build one from a
> training pool. Release notes: `docs/releases/0.0.1.md`.

| Field | Value |
|---|---|
| Model | ej 0.0.1 (2026-10-08); the default fit seed, fixed before fitting, not a selected seed |
| State key | `3b3e66d28fb423f98b734bd0c1d324cdc92bb8e10eda0e7ddf9a8e54de3f3fe2` (recorded in the pack header; reproduced by `scripts/state_key.py --git <model-code repo> f46cf7c <training pool> --ck-dir lowbit-b3b010513f948ceb`) |
| Code | runtime: `ej/_runtime` of this repository (39 sha256-pinned modules; `ej.integrity.RELEASE_PROVENANCE`); fitted with model-code commit `f46cf7c` of the maintainers' development repository (not public) with the encoder switch set to `lowbit-b3b010513f948ceb`; the same recipe is `python -m ej.train` |
| Size | counted 10.87 MiB (11.40 MB; bit-level bound: encoder codes + vocabulary + int8 heads); on disk 11.4 MB: one file `model.ejpack` of 11,384,312 bytes (ejpack v1: 2/3-bit encoder codes and their fp16 steps / offsets 8.17 MB, int8 heads and cross weights 1.62 MB, hashed cross keys 1.27 MB, other tensors and skeleton 0.26 MB, tokenizer and encoder config 0.04 MB), the whole download (no base model is fetched); resident: peak live model tensors while predicting 128 MB by default (encoder dequantised to fp32 at load), 35 MB with `low_memory=True` (encoder kept at 2/3 bits, each weight dequantised when used); process RSS during predict about 558 / 455 MB, of which about 204 MB is `import torch` (README "Memory and latency") |
| Latency (CPU, Python reference, 1 thread, one record per call) | packed model, first 50 td development records, warm: 214 ms per record (317 ms with `low_memory=True`); cold 1,061 / 1,385 ms (2026-10-09, Intel Xeon @ 2.80GHz, 4 logical cores, contended). Benchmark protocol on the td development suite (weights-directory format, 2026-10-08): warm 356.6 ms mean / 363.1 ms median, cold 1,922.0 / 1,413.7 ms. The two measurements are not comparable (README "Latency") |
| Language | English only (§6) |
| Licence | weights **CC BY-SA 4.0**; code Apache-2.0 |
| Base model | `intfloat/e5-small-v2` (MIT), revision `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| Weights / code | not yet published / https://github.com/5ak3t/ej |

## 1. What the model does

A **System-1 decision model for devices**: given a `state` (text, or a JSON object as text) and typed questions, it returns
one probability distribution per question in **one pass of a small encoder plus small heads**, without decoding tokens.
Usage: the repository README.

| Question type | Input | Output |
|---|---|---|
| `choice` | instructions + a runtime list of option texts | a probability per option |
| `noul` | a yes/no statement | P(false), P(true) |
| `score` | instructions + an ordered list of level texts | a distribution over the levels |

- **No LLM at inference**; teachers are used at fit time only and their weights do not ship.
- **Calibration is fitted, then checked per suite**: a log-linear pool with per-cell calibration and a lapse term; whether
  the result is calibrated is measured on each suite against a perfect-calibration ECE floor (§5).
- **Record independence.** Zero-shot `predict` is record-independent: each record's output depends only on that record (up
  to the float noise of batching). `adapt` (and `AdaptedModel.observe`) is an opt-in, per-workflow transductive mode: it
  pools option statistics across that workflow's labelled records, so use it only for one workflow with a fixed option set
  (the same question ids and option keys).
- Option texts are read as text, so new label spaces can be asked zero-shot; quality on unseen workflows is limited (§6).

## 2. Architecture

A later set of architecture changes (no NLI slot, one signed and bounded bias weight, a pre-registered new-group
predictive) was evaluated as a candidate and **failed** its pre-registered release rule, so it is not part of this release
and its code is not in this repository; the weaknesses it addressed still apply here and are stated below. Module-level
detail: `docs/architecture.md`.

**2.1 Low-bit shared encoder (the only transformer on the device).** Base `intfloat/e5-small-v2`, mean pooling,
`query:` / `passage:` prefixes. Trimmed vocabulary chosen a priori (a dropped piece is re-split, never `[UNK]`). **2-bit**
embeddings and feed-forward matrices, **3-bit attention**, groups of 128 input columns with fp16 step and offset; GPTQ
initialisation, then quantisation-aware distillation to a 4-bit e5. Encoder checkpoint: `lowbit-b3b010513f948ceb`
(distilled on the 20,932 distinct training-pool texts; fidelity: pooled cosine to the 4-bit e5 .992 on held-out pool texts; fixed and
recorded before fitting). The state is encoded **once** per record; JSON fields are read by key-aware attention.

**2.2 Expert pool, int8 heads.** 11 expert slots: a deep convex conditional logit over label-agnostic features (zero-shot
capable), wide sparse state-token × option-slot crosses, a rich bilinear/ordinal/MLP scorer, a centred prior-free expert,
field attention, relational evidence over JSON cross-field relations, a slot-free relational reader, a distilled
decision-encoder scorer, a distilled NLI pair head, and a product-of-experts pair (§2.3). The NLI pair head is silent
(exactly 0) on JSON states (measured with this code). Every head is fitted on the low-bit encoder's features and compacted to
**int8** (per-row int8 matrices; int8 cross weights with 40-bit hashed keys).

**2.3 Option-text bias expert.** A **bias-only expert** reads the question and the option texts but not the state; the main
deep expert is trained as a product of experts with the frozen bias logits as an offset (Clark et al. 2019; He et al.
2019), and the bias enters the pool as a pair `[z_b, −z_b]` with two non-negative weights, i.e. one signed coefficient. The
coefficient is not bounded, so in unseen cells the option-text prior can be divided out more than once.
Whether the debiasing helps on a new workflow is a per-suite question, reported against state-free baselines (always the
option with the lowest / highest bias logit): on the development suites (micro accuracy per cell of question type ×
structured state) ej 0.0.1 beats both baselines in all 3 td cells, in 2 of 3 zs_td cells (noul: lowest-bias baseline .622,
model .573) and in 2 of 4 zs_wide cells (JSON noul: highest-bias baseline .590, model .482; JSON score: .308 vs .249); no
transfer claim is made where the model does not beat both.

**2.4 Teachers (fit time only; nothing ships).** A decision encoder (e5 layers 10-12 fine-tuned on the training pool)
distilled from its out-of-fold distributions, and the NLI cross-encoder `cross-encoder/nli-deberta-v3-xsmall` (Apache-2.0)
distilled into the NLI pair head. No large teacher is used.

**2.5 Group-honest stacking.**
- Log-linear pool with lapse per calibration cell (question type × seen/unseen option slots × structured state):
  `p = (1 - eps) softmax(sum_e a_e z_e) + eps / K`; a selective correctness head is adopted per regime only where it beats
  the pool.
- **Group-honest nested cross-fitting**: rows for a held-out group come only from models and teachers that never saw it.
- **New-group predictive, chosen in fit.** For each regime of the unseen-slot pool (plain text, JSON) one of four
  predictives is chosen by a leave-one-group-out check over the training groups; ej 0.0.1 chose 'groups' (plain text, 4
  groups) and 'mean' (JSON, 3 groups). With so few groups that choice is weakly supported, and the hierarchy's scales sit at
  the edge of their grid, so the hierarchy acts as a plug-in estimate, not a fitted hierarchy.

## 3. Training data

One fit on one training pool (9,719 records, 7 source groups; sha256
`ce1c1a6fecfd62a90317f6efc4f90fd5f9261becb7081fd00235a6f4d5ee9dbe`). Details and counts: `docs/data-card.md`.

| Source (groups) | Records | Licence |
|---|---|---|
| Typed Decisions, workflows `agent_trace_observability`, `customer_service`, `invoice_processing` (3) | 736 | Apache-2.0 |
| **Support tickets written by Claude Haiku** for this project, train + calibration splits (1) | 780 | in-house; **not released** |
| Banking77 / CLINC150 `plus` / GoEmotions (3) | 3,000 / 3,000 / 2,203 | CC BY 4.0 / CC BY 3.0 / Apache-2.0 |
| **Total** | 9,719 records, 7 groups | |

Amazon counterfactual is not used: its upstream licence is CC BY-NC 4.0 (one mirror declares CC BY 4.0).

**Disclosure: Claude Haiku.** The in-domain support tickets were written by Claude Haiku (labels by construction from the
generation spec, no human review). They are part of training; the corpus itself is **not released**, so the training pool
cannot be rebuilt from public data alone. Open licence questions on this and other inputs: `NOTICE` (Q1-Q4).
Not in the training data: Typed Decisions `security_incidents` (the `zs_td` suite), MASSIVE and the zs_wide workflows
(evaluation only).

## 4. Evaluation

- **Suites** (`benchmarks/README.md`; never trained on):
  - `td`: Typed Decisions test split, the 3 training workflows.
  - `zs_td`: zs_td is ONE held-out workflow (security_incidents; dev 300 and final 100 records of the same workflow).
    The model code was kept on its dev accuracy after about 55 logged comparisons on it, so final zs_td estimates accuracy on
    further records of a workflow the model was selected on: it is neither unbiased nor unseen-workflow evidence. Its
    fit-to-fit SD (same code, other RNG) is ~.022 accuracy per fit (SD of a difference .031, 4 pairs), twice its
    record-cluster SE (.011); one workflow gives no between-workflow variance. Unseen-workflow evidence = zs_wide final
    macro_real.
  - `zs_massive`: MASSIVE (en) intents with leak-free option sets. MASSIVE was never trained on, but its label space is
    not new: about 22% of its intent names (13 of 59 development option texts: 1 identical, 12 near) overlap CLINC150 /
    Banking77 option texts in the pool, and 4 final records are exact duplicates of pool records (removed at the next
    benchmark run).
  - `zs_wide`: workflows never trained on: dev 154 workflows (111 from public sources: SNI tasks, SGD services, ABCD; 43
    GLM-synthetic), final 147. The split is by task, not by dataset family: 53 of the 147 final workflows share a family
    with a dev workflow; pool-source tasks were dropped from the final set.
  - `tickets` (in-domain, private) and `tickets_ood` (held-out writing styles of the same generator: a style shift).
- **Metrics**: per-suite mean NLL of the gold option (headline: the mean over td, zs_td, zs_massive, tickets), micro
  accuracy (zs_wide: group-macro over the real sources), ECE over 15 bins with its perfect-calibration floor, certified
  automation (risk .10, δ .10; **this suite only**). Each cell has a 95% record-cluster CI. Scoring: `ej.eval`
  (`benchmarks/scoring.py` for rivals).
- **Selection.** Development numbers are selected: more than 55 logged comparisons were made on the development suites, so
  their numbers are optimistic. The final suites are read once per release; zs_td final is not unseen-workflow evidence
  (above).
- **Leak-free option sets**: for sampled option sets every option is equally likely to be the gold, so option frequency
  reveals nothing (a frequency rule reached .631 micro accuracy against chance .240 on a non-leak-free MASSIVE suite).
- **Rivals** and contamination flags: `benchmarks/METHOD.md`.

## 5. Results

**ej 0.0.1** (final suites, scored once, 2026-10-08: `benchmarks/results/README.md`). 95% record-cluster CIs; calibrated
= observed ECE15 at or below the 95th percentile of a perfectly calibrated model's ECE15 on that suite.

| Suite | NLL [95% CI] | Micro accuracy [95% CI] | ECE15 [95% CI] | Calibrated on this suite | Certified automation, this suite only (coverage / error) |
|---|---|---|---|---|---|
| td (seen workflows) | .663 [.613, .716] | .721 [.695, .746] | .071 [.050, .094] | no (floor q95 .034) | .300 / .056 |
| zs_td (`security_incidents`, selected on) | 1.145 [1.113, 1.180] | .422 [.388, .452] | .094 [.068, .135] | no (.059) | 0 |
| zs_massive (leak-free) | .481 [.430, .530] | .832 [.808, .857] | .051 [.041, .073] | no (.040) | .798 / .080 |
| tickets | .629 [.583, .680] | .712 [.679, .746] | .033 [.028, .080] | yes (.061) | 0 |
| tickets_ood | .715 [.650, .787] | .683 [.639, .726] | .050 [.039, .099] | yes (.072) | 0 |
| zs_wide (147 unseen workflows; micro, all sources) | 1.196 [1.176, 1.220] | .422 [.405, .439] | .121 [.108, .137] | no (.027) | 0 |

zs_wide final macro_real (group-macro accuracy over the real sources SNI, SGD and ABCD, 105 workflows; t interval over
workflows within sources) **.419 [.379, .458]**: the unseen-workflow number. GLM-synthetic workflows separately: .358
[.323, .394] (42 workflows). Mean NLL over td, zs_td, zs_massive and tickets: .7297. Seed variability of this code and
data (6 fits, development suites): between-seed SD .0105 in that metric, .019 in zs_wide macro_real, .037 in zs_td micro
accuracy; differences of that size between fits are not evidence of a change.
Against the rivals' sealed runs (paired CIs in `benchmarks/results/README.md`), ej 0.0.1 has lower micro accuracy than the
144M-0.8B rivals and the hosted Jev on zs_td and zs_massive (except Julia-1 on zs_massive) and than laya on td; no
calibration, latency or size ranking is claimed.

## 6. Limitations

- **Unseen workflows: low accuracy.** Expect weak decisions on a new workflow until you measure it on labelled cases of
  your own. A few labelled records help mainly by teaching the label prior (`docs/adaptation.md`).
- **Fit-to-fit noise.** Two fits of the same code that differ only in their random seed differ by about .03 zs_td micro
  accuracy (SD of a difference); single-fit differences of that size are not evidence (§4).
- **Option-text tilt** on new workflows is reduced by the bias expert (§2.3) but not removed.
- **More data did not simply help**: adding broad text and dialogue data made td, tickets and zs_massive worse (zs_td
  unchanged within noise), and synthetic workflow data did not improve unseen-workflow accuracy; neither is in the
  training data.
- **English only** (base encoder and all data). **Synthetic in-domain data**: the tickets come from one model family
  (Claude Haiku) and have no human labels.
- **Certified automation is per suite**: a threshold certified on one workflow can fail on another (a td-certified
  threshold had error .771 on zs_wide). Certify on labelled rows of the workflow you automate.
- **Python runtime only** (torch + transformers, CPU). Latency is the Python reference on a 4-core CPU, not a phone.
- **Float noise**: batch composition, chunk size and thread count move probabilities by at most about 6e-7 (measured).
- **No download at load**: the pack holds the tokenizer and the encoder config. A weights directory (the secondary format)
  instead builds the encoder from the base model, fetched from the Hugging Face Hub at revision `ffb93f3b` on first use.
- **Reproducibility**: a release is tied to its runtime (`ej/_runtime`, sha256 manifest checked at every load); a known
  pack is checked against its content digest in `ej.integrity.KNOWN_PACKS`.

## 7. Intended use

- On-device triage and routing decisions (support tickets, intents, workflow checks) where a probability distribution is
  needed per question and uncertain cases are **escalated** to a slower System 2 (an on-device LLM or a cloud model).
- Ranking, gating and abstention using the returned probabilities, with thresholds certified on your own labelled data.

## 8. Out-of-scope use

- Sole decision-maker for consequential decisions about people (credit, employment, health, legal, safety) without human
  review.
- Non-English text; generative tasks; free-text answers.
- New structured workflows without first measuring quality on labelled examples of that workflow (§6).
- Security-incident triage as a zero-shot claim (that workflow drove model selection).
- Transductive use across workflows, or on option sets that change between records: `adapt` / `observe` are for one
  workflow with a fixed option set (§1).

## 9. Licence and attribution

The weights are licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. Adaptations
must be shared under CC BY-SA 4.0 or a compatible licence. Suggested attribution:

> ej 0.0.1 by Saket Bhushan, licensed under CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/). Derived from intfloat/e5-small-v2 (MIT; Wang et al., arXiv:2212.03533)
> and trained on Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions (Apache-2.0) and an
> in-house support-ticket corpus written with Claude Haiku (not released), with distillation from
> cross-encoder/nli-deberta-v3-xsmall (Apache-2.0). Full notices, creators and open questions: NOTICE.
