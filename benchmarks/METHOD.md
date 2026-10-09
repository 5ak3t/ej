# Benchmark method: rivals, contamination audit, adapters

How the rival models were chosen, what each one may have seen in training, and how each is run. Model facts below were
read from each model's card and repository files at the pinned revision (read as data; no downloaded code was executed
unless it is the model's own published package, named in its adapter). Where this page says "measured", the number comes
from running `benchmarks/run_bench.py`.

## 1. Rival set

ej is small and CPU-only, so the main table compares it with small typed-decision models that run on a CPU, one model
process at a time, under 4 GB of RAM.

| rival (registry name) | why it is in | not zero-shot on (flag these rows) |
|---|---|---|
| **Julia-1** `julia-1` | closest size class (144M) dedicated typed-decision model; Apache-2.0; served by llama.cpp `/v1/systemone` | td, zs_td (likely, §2); MASSIVE scenario labels adjacent |
| **laya** `laya` | strongest published Typed Decisions result (0.766 on the test split) | **td, zs_td** (trained on the Typed Decisions train split incl. `security_incidents`) |
| **kev-0.8b** `kev-0.8b` | open 0.8B decision model with its own `/v1/systemone` server | none of the benchmark suites |
| **OpenThai-SystemOne** `openthai-systemone` | 0.75B Thai + English slot-head decision model, Apache-2.0 | **zs_massive** (MASSIVE intent en train split in its training mix) |
| **GLiClass edge v3.0** `gliclass-edge` | 32.7M: the only rival in ej's size class; a clean zero-shot baseline | none |
| **Jev** `jev` | hosted reference (TypeSafe AI, closed weights) | td, zs_td likely seen or tuned (§2) |

Also written (adapters in `rivals/`) but not in the published table: kev-4b, decider-2b / 12b, Von 1.3, DecidaBERT-large,
NLI zero-shot classifiers (deberta-v3 xsmall / large, bart-large-mnli) and constrained-likelihood small LLMs (Qwen3-0.6B,
Qwen3.5-0.8B-Base). Most need a GPU or a large download.

## 2. Contamination audit

In-domain = the rival trained on that dataset's train split or on that exact workflow; zero-shot = its training list
excludes it; adjacent = same domain written by others; unknown = the card does not say.

| rival (HF id @ revision) | licence | size | training data (card) | td | zs_td | zs_massive | tickets |
|---|---|---|---|---|---|---|---|
| laya `convaiinnovations/laya-typed-decisions` @e929ae5 | Apache-2.0 | 421M | RLCD mix (AG News, BoolQ, spam, phishing, support triage, ...) + **Typed Decisions train split, all 4 workflows** | **in-domain** | **in-domain** | unknown | adjacent |
| kev-0.8b `jaredpalmer/kev-0.8b` @bf75a6a | Apache-2.0 | 0.8B + LoRA | banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia, reviews, IMDB, generated policy sets, CFPB, ... | zero-shot | zero-shot | zero-shot | zero-shot |
| Julia-1 `SupersonicLabs/Julia-1` @a85b127 (GGUF `ggml-org/Julia-1-GGUF` @16fee17) | Apache-2.0 | 144.3M | **undisclosed**; its validation sets are named after all four Typed Decisions workflows, banking77 and 52 MASSIVE locales | **likely in-domain** | **likely in-domain** | adjacent | adjacent |
| OpenThai-SystemOne `iapp/OpenThai-SystemOne` @dc67b02 | Apache-2.0 | 752.7M | Thai CPT + SFT on ~28 public sets incl. **MASSIVE intent (en, train)**, CLINC, AG News, MNLI, synthetic ticket routing and alert triage | zero-shot | adjacent | **in-domain** | adjacent |
| GLiClass edge `knowledgator/gliclass-edge-v3.0` @df03993 | Apache-2.0 | 32.7M | synthetic pretraining + logic LoRA | zero-shot | zero-shot | zero-shot | zero-shot |
| Jev `jev-1.13.0` (hosted) | proprietary API | undisclosed | undisclosed; TypeSafe publishes Typed Decisions results on the same test split | **likely seen/tuned** | **likely seen/tuned** | unknown | unknown |
| cross-encoder/nli-deberta-v3-xsmall @a150876 | Apache-2.0 | 22M + embeddings | SNLI + MultiNLI | zero-shot | zero-shot | zero-shot | zero-shot |

ej itself: trained on the Typed Decisions train split of three workflows (td in-domain), never on `security_incidents`
(zs_td: not trained on, but its development records were used for model selection), never on MASSIVE (zs_massive:
not trained on; about 22% of its intent names overlap CLINC150 / Banking77 options in the pool), and on the training split
of the private ticket corpus (tickets in-domain, tickets_ood a writing-style shift). The NLI cross-encoder above is a fit-time teacher of ej (distilled
into one of its heads), so it is not an independent comparison.

## 3. Which rival is fair on which suite

| suite | like-for-like | report with a flag | why |
|---|---|---|---|
| td | laya (same-split specialist, as ej); Julia-1 likely | kev, OpenThai, GLiClass (zero-shot generalists: ej has the in-domain advantage); Jev likely seen | head-to-head on the Typed Decisions test split |
| zs_td | kev, GLiClass, NLI baselines (zero-shot; ej was not trained on it but was selected on it); OpenThai adjacent | **laya, Julia-1, Jev** (trained on or likely saw this workflow) | a zero-shot claim cannot be tested against rivals that saw the workflow |
| zs_massive | kev, GLiClass, large NLI models (zero-shot by card) | **OpenThai** in-domain; Julia-1 adjacent; laya, Jev unknown | only zero-shot rivals support a zero-shot claim |
| tickets, tickets_ood | all rivals are zero-shot; ej is in-domain on tickets, out of distribution on tickets_ood | rivals trained on other support-ticket data (adjacent) | tickets_ood is the fairer of the two |

## 4. Adapters (`rivals/`) and common rules

- Records are converted to the System One wire format (`rivals/common.py`): choice `criteria` {key: description}, score
  `criteria` = ordered level texts, noul `criteria` only when the options carry real descriptions.
- Answers are mapped back by option **key** into the record's option order, renormalised, then smoothed with EPS = 1e-6
  so a rival's rounded 0.0 is not scored as -log(1e-15).
- A rival's own 4-decimal output rounding is switched off where it is a plain `round(x, 4)` (laya, decider, Von), so
  full-precision probabilities are scored. Each rival is otherwise run with its documented inference and its own
  calibration (temperatures as shipped).
- Hugging Face models are loaded in float32. torch threads = `BENCH_THREADS` (default 1); `run_bench.py` records the
  threads observed inside the timed calls (`threads_measured`, `threads_ok`).
- **ej** (`rivals/ej_adapter.py`): `ej.load(EJ_WEIGHTS, revision=EJ_REVISION, threads=BENCH_THREADS).predict(...)`; ej
  sets the thread count inside each predict call and reports it (`Model.last_threads`). `EJ_WEIGHTS` (a local weights
  directory) is required. `EJ_COLD=1` measures ej without its prediction-time caches.
- **Julia-1**: llama.cpp `llama-server` at commit b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea built CPU-only (`LLAMA_SERVER`
  = path of the binary); the adapter starts the server on the BF16 GGUF (`JULIA_GGUF=q8_0` for the smaller build) and
  stops it at exit. The GGUF path was not compared with Julia's own PyTorch runtime.
- **OpenThai-SystemOne**: its repository ships the model as remote code; the adapter does not execute it but rebuilds the
  documented inference with stock transformers (`Qwen3_5TextModel`) and the checkpoint's own tensors (all tensors
  matched, asserted). The rebuild reproduces the published 166-token encoding of the repository's API example.
  Order-invariant averaging (8 cyclic option orders) only triggers at >= 11 options; the benchmark suites have <= 8.
- **Jev** (`rivals/jev_adapter.py`): `POST https://api.typesafe.ai/v1/systemone`, pinned model `jev-1.13.0`
  (`JEV_MODEL`). The API key is read from `JEV_API_KEY_FILE` and only sent in the auth header; records carrying gold are
  refused; each response is cached on disk by the sha256 of the exact request (`JEV_CACHE_DIR`), so re-scoring never
  re-sends. Measured: probabilities come rounded to 2 decimals and are not deterministic (the same request re-sent moved
  values by up to 0.04), so one run is one cached draw.
- **kev**: kev's own server code path in process, full-precision calibrated distributions (its API body rounds to 4 dp).
- **GLiClass**: one pipeline call per question (labels = option texts, prompt = instructions, threshold 0); sigmoid scores
  are turned back into logits and softmaxed over the options.

## 5. Running

```bash
pip install -e .                       # ej; rivals need their own environments (rivals/__init__.py, column 3)
export EJ_WEIGHTS=/path/to/model.ejpack  # a local ej model file
export BENCH_FINAL_DIR=/path/to/sealed-suites   # suites under this directory refuse to run without --final
python benchmarks/run_bench.py my_suite.jsonl ej gliclass-edge --out bench-out
python benchmarks/run_bench.py $BENCH_FINAL_DIR/td.jsonl ej --final --out bench-out    # once per release
```

One model process at a time. Each run writes `<suite>.<model>.preds.jsonl` (per-record predictions: keep them private
when the suite is private) and `<suite>.<model>.summary.json` (aggregates; what `results/` publishes).

## 6. Methodological choices (state them next to any result)

1. Calibration: rivals are scored as shipped (their own temperatures); no per-rival temperature refit.
2. Von yes/no is scored as its calibrated posterior (`VON_NOUL_DECISION=raw`), not its default banded decision output.
3. Latency: one protocol for every model (one record per call after a warm-up record, wall clock, 1 thread checked
   inside the timed calls, box and load average recorded). Latencies are compared only when measured on the same suite,
   box and day; otherwise they are reported as **not comparable**. The rival latencies (final td, 2026-10-07, load not
   recorded) are not comparable with ej's numbers. Jev latency is a network round trip.
4. Julia-1 on td / zs_td is flagged likely in-domain from its validation-set names (training data undisclosed).
5. Jev is scored as served: 2-decimal probabilities plus EPS smoothing, one non-deterministic draw frozen by the cache
   (no draw-to-draw spread was measured, so differences of a few points against Jev are not claims).
6. Every cell carries a record-cluster CI and every ej-vs-rival difference a paired CI; an ordering is stated only where
   that CI excludes 0.
