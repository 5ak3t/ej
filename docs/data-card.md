# ej data card (training pool of ej 0.0.1)

ej 0.0.1 is fitted on one training pool of 9,719 records from 7 source groups. Counts come from counting records per source
and workflow in the pool file; no evaluation record was read.

## 1. Summary

| Item | Value |
|---|---|
| sha256 of the pool file | `ce1c1a6fecfd62a90317f6efc4f90fd5f9261becb7081fd00235a6f4d5ee9dbe` |
| Records / questions | 9,719 / 14,223 |
| Source groups (held-out units) | 7 |
| Language | English |

The pool is **not distributed** (it contains the in-house ticket corpus, §6); the release pins its sha256 only. The record
format, and how to assemble a pool of your own from the public sources below, are in `docs/training.md`.

## 2. Composition

| Source | Group(s) | Records | Questions | choice | noul | score | Licence |
|---|---|---|---|---|---|---|---|
| Typed Decisions | `agent_trace_observability` | 244 | 1,220 | | | | Apache-2.0 (LocalLLaMA/typed-decisions) |
| Typed Decisions | `customer_service` | 246 | 1,230 | | | | Apache-2.0 |
| Typed Decisions | `invoice_processing` | 246 | 1,230 | | | | Apache-2.0 |
| Typed Decisions (all 3) | — | 736 | 3,680 | 1,226 | 982 | 1,472 | Apache-2.0 |
| support tickets (**written by Claude Haiku**) | tickets | 780 | 2,340 | 780 | 780 | 780 | in-house corpus, not released (§6) |
| Banking77 | banking77 | 3,000 | 3,000 | 3,000 | | | CC BY 4.0 (PolyAI/banking77; mirror mteb/banking77) |
| CLINC150 | clinc150 | 3,000 | 3,000 | 3,000 | | | CC BY 3.0 (clinc/clinc_oos, config plus) |
| GoEmotions | goemotions | 2,203 | 2,203 | 2,203 | | | Apache-2.0 (google-research-datasets/go_emotions, simplified) |
| **Total** | 7 groups | **9,719** | **14,223** | 10,209 | 1,762 | 2,252 | |

## 3. Construction

Common record format: the ej input format plus `gold: {qid: {label, probs}}`; every converter is deterministic, seeded by
the record id.

- **Typed Decisions.** Train split minus the workflow `security_incidents` (held out entirely: the `zs_td` suite) minus one
  hash bucket of the seen workflows (development suite). Gold = label plus the dataset's teacher distribution where present.
- **Support tickets.** The in-house Claude-Haiku-written corpus, train + calibration splits only (its development and test
  splits are evaluation suites). Three questions per ticket: department (choice), urgent (noul), mood (score, 3 levels);
  labels by construction.
- **Banking77, CLINC150, GoEmotions** → choice questions over the gold label plus distractors (2-8 options) drawn
  **leak-free** (§4); CLINC150 out-of-scope is the ordinary label "none of these"; GoEmotions single-label rows, balanced to a
  maximum gold share of .11. At most 3,000 records per public source.

## 4. Leak-free candidate sampling

**Problem.** If distractors are drawn uniformly from the other labels, how often an option text appears across items
reveals the gold label: on a non-leak-free MASSIVE suite an option-count rule reached micro accuracy .631 against chance
.240 without reading any text.
**Construction.** A suite is leak-free iff `P(gold = j | S) = 1/|S|` for every `j` in the option set `S`: distractors by
**conditional Poisson sampling** with IPF-fitted weights, seeded hash order of records, over-represented labels dropped.
**Applied to** the Banking77 / CLINC150 / GoEmotions training parts and the MASSIVE evaluation suite. Builder:
`benchmarks/suites/leakfree.py`.

## 5. What is not in the pool

| Data | Status |
|---|---|
| Typed Decisions workflow `security_incidents` | held out (suite `zs_td`); security topics also filtered from new parts |
| MASSIVE (en) intents | evaluation only (about 22% of its intent names overlap CLINC150 / Banking77 option texts) |
| zs_wide workflows (SNI tasks, SGD services, ABCD, GLM-synthetic) | evaluation only; SGD (CC BY-SA 4.0), Super-NaturalInstructions permissive subset and ABCD (MIT) pools were also tested for training and discarded (worse on the development suites) |
| Amazon counterfactual | excluded: the upstream `amazon-research/amazon-multilingual-counterfactual-dataset` LICENSE is CC BY-NC 4.0 (the `mteb/amazon_counterfactual` mirror declares CC BY 4.0) |
| Other non-commercial data and models trained on it | excluded |

Licences are checked against the upstream LICENSE file of each source, not a mirror's metadata.

## 6. Disclosures

**Claude Haiku.** The support-ticket corpus (1,200 tickets written by Claude Haiku,
`claude-haiku-4-5-20251001`, Anthropic; labels by construction, no human review) contributes 780 tickets to the training
pool. The corpus itself is **not released**. Consequence: the pool cannot be rebuilt from public data alone. Whether the
provider terms allow training openly released weights on these outputs is an open question (`NOTICE`, Q1).
**GLM-synthetic evaluation workflows.** 43 zs_wide development workflows were generated with GLM (`glm-4.7`) and labelled
by the same model (blind self-consistency filter only; no independent annotation). They are evaluation data, never
training data, and are reported separately from the real-source workflows.
**Training-time derivatives.** All derived signals come from pool texts only: out-of-fold decision-encoder distributions,
NLI teacher labels on an augmented transfer set of pool pairs, the trimmed encoder vocabulary, and a fidelity slice of pool
texts for the encoder's quantisation-aware distillation (20,932 distinct pool texts). Remote GPU jobs received only pool
data and pool-derived caches.

## 7. Known data limitations

- Only 3 Typed Decisions training workflows.
- The tickets are synthetic, from one model family, English; labels are the generation spec, not human annotation.
- Typed Decisions gold distributions come from the dataset's own teacher; which model produced them is not recorded
  (`NOTICE`, Q2).
- Intent sources have one question per text with 2-8 options; deployments may offer many more options than seen in training.
