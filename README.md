# ej

A small on-device model for typed decisions: one calibrated probability distribution per question, in one pass, without
generating tokens.

<p>
  <a href="https://github.com/5ak3t/ej/actions/workflows/tests.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/5ak3t/ej/tests.yml?style=for-the-badge&labelColor=000000&label=tests" height="28"></a>
  <img alt="Weights: not yet published" src="https://img.shields.io/badge/WEIGHTS-not%20yet%20published-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28">
  <a href="LICENSE"><img alt="Code: Apache-2.0" src="https://img.shields.io/badge/code-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSES/CC-BY-SA-4.0.txt"><img alt="Weights: CC BY-SA 4.0" src="https://img.shields.io/badge/weights-CC%20BY--SA%204.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

You give ej a `state` (free text, or a JSON object as text) and a set of typed questions; it returns one probability
distribution per question. There is no LLM at inference: a 2/3-bit `intfloat/e5-small-v2` encoder reads the record once and
small int8 heads score the options. This repository holds the package, the training and evaluation code, and the benchmark.

> **Status (2026-10-09): ej 0.0.1. Weights: not yet published** (no hosting has been decided yet). The model is one file,
> `model.ejpack` (11,384,312 bytes); `ej.load` needs a local copy of it, or a pack you build yourself (`python -m ej.train
> fit`, then `python -m ej.train export`). Every number below was measured with the 0.0.1 weights.

## Highlights

- Three question types in one call: `choice` (any option texts, chosen at run time), `noul` (a yes/no statement) and
  `score` (ordered levels). Every question of a record is answered in the same pass.
- Calibration is fitted, then **measured per suite** against the ECE a perfectly calibrated model would show there.
- Small: one model file of 11,384,312 bytes (11.4 MB; 10.87 MiB counted at the bit level) and nothing else to download: the
  tokenizer and the encoder config are inside it, no base model is fetched. CPU only, deterministic for a fixed thread count.
- Pickle-free weights: `model.ejpack` (ejpack v1) is a JSON header plus binary sections (2/3-bit encoder codes, int8 heads,
  hashed vocabularies, tokenizer), each sha256-checked before anything is decoded; classes come from an allowlist.
- `ej.load(path, low_memory=True)` keeps the encoder at 2/3 bits in memory and dequantises each weight when it is used:
  identical predictions, lower peak memory, slower ([Memory and latency](#memory-and-latency-of-the-packed-model)).
- Few-shot adaptation to your workflow from a handful of labelled records (`model.adapt`, `AdaptedModel.observe`).
- Train your own from a pool of labelled records (`python -m ej.train`) and evaluate it with the same evaluator that
  produced every number below (`python -m ej.eval`).

| Question type | You give | You get |
|---|---|---|
| `choice` | instructions + a list of option texts (any labels, chosen at run time) | P(option) for each option |
| `noul` | a yes/no statement, options `false` / `true` | [P(false), P(true)] |
| `score` | instructions + ordered level texts | a distribution over the levels |

## Model

| Model | Base (licence) | Size: counted / on disk / resident | CPU latency, 1 thread, one record per call (td) | Unseen workflows: macro_real accuracy | Card |
|---|---|---|---|---|---|
| ej 0.0.1 | `intfloat/e5-small-v2` (MIT), 2-bit weights, 3-bit attention | 10.87 MiB / 11.4 MB / 128 MB (35 MB with `low_memory=True`) | warm 214 ms (317 ms with `low_memory=True`) | .419 [.379, .458] | [ej-0.0.1](docs/model-cards/ej-0.0.1.md) |

Sizes: *counted* = a bit-level bound (encoder codes + vocabulary + int8 heads); *on disk* = the one file `model.ejpack`
(11,384,312 bytes), which is the whole download (no base model is fetched); *resident* = peak live model tensors while
predicting (default: the encoder is dequantised to fp32 at load; `low_memory=True`: it stays at 2/3 bits). Process memory
and the box: [Memory and latency](#memory-and-latency-of-the-packed-model); benchmark latency: [Latency](#latency).

## Results

Final test suites of ej 0.0.1, **scored once** for the release (2026-10-08), with 95% record-cluster CIs
([benchmarks/results/](benchmarks/results/README.md)). Development suites were used to choose among many candidates, so
development numbers are selected and optimistic; they are not reported as results here.

| final suite | records / questions | NLL [95% CI] | micro accuracy [95% CI] | ECE15 [95% CI]; calibrated on this suite? |
|---|---|---|---|---|
| td (seen workflows) | 300 / 1,500 | .663 [.613, .716] | .721 [.695, .746] | .071 [.050, .094]; no |
| zs_td (`security_incidents`, selected on) | 100 / 500 | 1.145 [1.113, 1.180] | .422 [.388, .452] | .094 [.068, .135]; no |
| zs_massive (leak-free) | 1,000 / 1,000 | .481 [.430, .530] | .832 [.808, .857] | .051 [.041, .073]; no |
| tickets (in-house) | 169 / 507 | .629 [.583, .680] | .712 [.679, .746] | .033 [.028, .080]; yes |
| tickets_ood (in-house, style shift) | 147 / 441 | .715 [.650, .787] | .683 [.639, .726] | .050 [.039, .099]; yes |

**Unseen workflows** (zs_wide final, 147 workflows never trained on, read once): group-macro accuracy over the real
sources **macro_real .419 [.379, .458]** (105 workflows: SNI .456, SGD .571, ABCD .229; t interval over workflows within
sources). GLM-synthetic workflows, reported separately: .358 [.323, .394] (42 workflows). This is the number to expect on
a new workflow before adaptation.

Against six rivals on the same final suites (micro accuracy, paired CIs; [BENCHMARKS.md](BENCHMARKS.md)): td below laya,
level with Jev and Julia-1, above the other three; zs_td below all six; zs_massive below Jev, laya, kev-0.8b and
OpenThai-SystemOne, above Julia-1 and gliclass-edge. On the in-house ticket suites ej's NLL is lower than every rival's.
No latency, size or calibration ranking is claimed.

## Release

| Version | State key | Weights | Notes |
|---|---|---|---|
| ej 0.0.1 | `3b3e66d28fb423f9` | not yet published (no hosting decided); one file `model.ejpack`, 11,384,312 bytes | [docs/releases/0.0.1.md](docs/releases/0.0.1.md) |

Before decoding anything, `ej.load` checks the pack's header digest and every section's sha256, the contents of a known
release against the digest shipped in `ej.integrity.KNOWN_PACKS`, and the runtime modules against their manifest. The
file's SHA-256 is in the release notes.

## Quickstart

Python >= 3.10 (tested on 3.11.15, CPU). Dependencies are pinned to the tested versions in `pyproject.toml`.

```bash
git clone https://github.com/5ak3t/ej && cd ej
pip install -e .            # inference;  -e '.[eval]' adds the evaluator, -e '.[train]' the training extras
```

You need an ej model file: the 0.0.1 `model.ejpack` once it is published, or one you build (`python -m ej.train fit`,
then `python -m ej.train export`; below). The example runs on any such file.

```python
import ej

model = ej.load('/path/to/model.ejpack')   # one local file (or a directory holding model.ejpack)
record = {
    'state': '{"customer_tier": "gold", "message": "The blender arrived with a cracked jug. Replace it before Friday."}',
    'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team should handle this request?',
                  'options': [{'key': 'returns', 'text': 'Returns and replacements for damaged or wrong items'},
                              {'key': 'billing', 'text': 'Billing, invoices and payment problems'}]},
        'needs_human': {'type': 'noul', 'instructions': 'The customer is upset enough that a human agent should reply.',
                        'options': ej.NOUL_OPTIONS},
        'urgency': {'type': 'score', 'instructions': 'How urgent is this request?',
                    'options': [{'key': '0', 'text': 'Low: can wait a week'},
                                {'key': '1', 'text': 'Medium: answer within two days'},
                                {'key': '2', 'text': 'High: answer today'}]},
    },
}
(probs,) = model.predict([record])
print(probs['route'])          # [p_returns, p_billing], sums to 1
```

`python examples/quickstart.py --weights model.ejpack [--records file.jsonl]` does the same for `ej.EXAMPLE_RECORD` or
your file. `ej.load('model.ejpack', low_memory=True)` gives the same predictions with less memory, more slowly. A weights
directory (safetensors + JSON, what `fit` writes) also loads; that format builds the encoder from the base model, which it
fetches from the Hugging Face Hub on first use.

**Input.** `{"id": "optional", "state": "...", "questions": {"<qid>": {"type": "choice | noul | score", "instructions":
"...", "options": [{"key": "k1", "text": "short description"}, ...]}}}`. At least 2 options per question, keys unique;
`noul` options are exactly `false` and `true` (`ej.NOUL_OPTIONS`); `score` options are ordered, lowest first. Records are
validated (`ej.validate_records`, raises `ej.RecordError`). **Output.** One dict per record, `{qid: [p_1, ..., p_K]}` in
option order, plain floats summing to 1 (within 1e-9).

**Determinism and process hygiene.** Predictions are deterministic for a fixed thread count and chunking; chunk size,
batch composition and thread count move probabilities only at float-noise level (measured max |Δp| about 6e-7).
`ej.load` leaves torch's thread count, its random state and `transformers` unchanged; `predict(records, threads=None,
chunk_size=None)` sets threads only inside the call. One loaded model per process. Environment: `EJ_CACHE` (default
`~/.cache/ej`; receives the pack's tokenizer files), `EJ_COLD=1` (bypass the prediction-time caches); for a weights
directory also `HF_HOME` and `HF_HUB_OFFLINE=1` (offline with a filled cache).

## Adapting to your workflow

**Record independence.** Zero-shot `predict` is record-independent: each record's output depends only on that record (up
to float noise). `adapt` is an opt-in, per-workflow transductive mode: it pools option statistics across that workflow's
labelled records, so use one adapted model per workflow with a fixed option set (the same question ids and option keys).

```python
labelled = [{**r, 'answers': {'route': 'returns', 'needs_human': 'true', 'urgency': '2'}} for r in my_records[:8]]
adapted = model.adapt(examples=labelled)   # per-(question, option) logit offsets; the weights do not change
adapted.observe(more_labelled)             # fold in labelled records as they arrive (same result as adapting at once)
probs = adapted.predict(new_records)
```

Measured on the development suites (**k (few-shot)** = the number of labelled *records* per workflow): on 154 unseen
workflows (111 public-source + 43 GLM-synthetic) k = 8 moves group-macro accuracy .356 → .389, +.032 [−.011, +.076]:
covers 0; on the single held-out zs_td workflow .396 → .545 at k = 16; on the seen td workflows +.016 [+.003, +.029].
What the labelled records teach is mostly the workflow's **label prior**: an add-one count of the labelled answers alone
comes within +.008 [−.0001, +.016] of the adapted model at k = 8 (record-cluster bootstrap CIs). Details and the dropped
label-free variant: [docs/adaptation.md](docs/adaptation.md).

## What to expect

- **Unseen workflows: low accuracy** (macro_real .419). Measure on your own labelled cases before automating; a few
  labelled records help, mostly by teaching the label prior.
- **Calibration is per suite**: calibrated on the two ticket suites, not on td, zs_td, zs_massive or zs_wide.
- **Certified automation (CA)** holds for **this suite only**: a confidence threshold certified on one workflow can fail
  on another (a td-certified threshold had error .771 on zs_wide). Certify on labelled rows of the workflow you automate.
- **English only**; Python runtime (torch + transformers, CPU); a pack needs no network at load or predict time.
- **Synthetic in-domain data**: the support tickets were written by Claude Haiku (no human labels).
- **Fit-to-fit noise**: six fits of the same code differ by SD .0105 in mean development NLL and .037 in zs_td micro
  accuracy; single-fit differences of that size are not evidence.
- More: [model card](docs/model-cards/ej-0.0.1.md) §6, [data card](docs/data-card.md) §7.

## How the benchmark works

Every model gets the same blind records (`benchmarks/run_bench.py`: gold removed, one record per call after a warm-up,
threads checked inside the timed calls) and is scored with the same functions (NLL, micro accuracy, ECE15, certified
automation; `benchmarks/scoring.py` = `ej.eval.metrics`). Final suites are read once per release; every cell carries a
record-cluster CI and every ej-vs-rival difference a paired CI, and an ordering is stated only where that CI excludes 0.
Rivals carry contamination flags ([benchmarks/METHOD.md](benchmarks/METHOD.md)). No suite file ships here: the public
suites are rebuilt from the public datasets with `benchmarks/suites/` (sha256 in `SHA256SUMS`); the ticket suites are
private. Summary: [BENCHMARKS.md](BENCHMARKS.md).

## Latency

`benchmarks/run_bench.py`, one record per call after a warm-up record, wall clock, 1 torch thread set inside each call
and checked (`threads_measured` = [1]), development suites, with the weights-directory format of the same model (the
encoder computes the same fp32 weights in both formats). Box: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores,
Python 3.11.15, torch 2.5.1 CPU; 1-minute load average 1.02-1.10; 2026-10-08.

| development suite | records | warm: mean / median ms per record | cold (`EJ_COLD=1`): mean / median ms per record |
|---|---|---|---|
| td (JSON states, 5 questions) | 164 | 356.6 / 363.1 | 1,922.0 / 1,413.7 |

Warm = texts seen in earlier records (such as option texts) are not re-encoded; cold = every text encoded again.
Summaries: `benchmarks/results/latency/`. Batching is faster per record. Rival latencies were measured on another day
under unrecorded load, so they are **not comparable** with these numbers and are not ranked against them.

### Memory and latency of the packed model

Measured on 2026-10-09 on the same kind of box (Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, shared and
contended, torch 2.5.1 CPU), one configuration per fresh process, with the procedure of `benchmarks/pack_memory.py`, on
the development tree's copy of this packed runtime (same code before it moved into the package). Memory: 72 development
records (24 each of td, zs_td, zs_wide) predicted in one call at 2 threads, cold; RSS from `/proc/self/status`, the peak
reset just before predict; model tensors = live torch tensors (one count per storage), peak sampled inside the encoder's
forward. Latency: `predict([r])` on the first 50 development td records, 1 thread, after one warm-up record.

| runtime | RSS after `import torch` | peak RSS during predict | model tensors, peak | warm ms per record | cold ms per record |
|---|---|---|---|---|---|
| default | 204 MB | 558 MB | 128 MB | 214 | 1,061 |
| `low_memory=True` | 204 MB | 455 MB | 35 MB | 317 (about 1.5x) | 1,385 |

Most of the process memory is code (torch, transformers), not the model. Predictions are identical in both runtimes
(max |Δp| = 0.0, `tests/test_pack_weights.py`). These latencies come from another day, load and harness than the table
above, so the two tables are **not comparable** with each other.

## Training your own

```bash
pip install -e '.[train]'
python -m ej.train fit --pool pool.jsonl --out my-weights      # all steps; checkpoints in my-weights.work, resumable
python -m ej.train export --weights my-weights --out model.ejpack   # the packed model file (what ej.load takes)
python -m ej.eval score --suite my_suite.jsonl --weights model.ejpack
```

The pool is JSONL in the input format plus `source` (the held-out group unit) and `gold: {qid: {label, probs}}`. The steps
(`encoder` on a GPU, `teachers`, `distil`, `fit` on CPU) can run one by one; the fit is the recipe of ej 0.0.1. The
0.0.1 training pool itself cannot be rebuilt from public data alone (its ticket corpus is not released). Pool format, public
sources and their licences, compute: [docs/training.md](docs/training.md); architecture: [docs/architecture.md](docs/architecture.md).
To compare two recipes, fit each several times with different seeds and use `python -m ej.eval compare` (seed-aware).

## Repository layout

| path | what |
|---|---|
| `ej/` | the package: `load`, `Model.predict`, `Model.adapt`, `AdaptedModel.observe`, records, integrity checks, process scope, pickle-free codec |
| `ej/pack/` | the packed format `ejpack v1`: writer, sha256-checked pickle-free reader, encoder build from the pack, streaming dequantisation (`low_memory=True`) |
| `ej/_runtime/` | the prediction code (39 sha256-pinned modules; [README](ej/_runtime/README.md)) |
| `ej/train/`, `ej/eval/` | training (`python -m ej.train`) and evaluation (`python -m ej.eval score / compare`) |
| `benchmarks/` | runner, scoring, rival adapters, suite builders + checksums, method, aggregate results |
| `docs/` | model card, data card, release notes, architecture, training, adaptation |
| `examples/`, `tests/`, `scripts/` | quickstart; tests (`pytest`; set `EJ_WEIGHTS=model.ejpack` for the prediction tests); `hf_layout.py`, `state_key.py`, `check_file_length.py` |

## Citation

```bibtex
@software{ej_2026,
  title   = {ej: calibrated typed decisions on device},
  author  = "{The ej contributors}",
  year    = {2026},
  version = {0.0.1},
  url     = {https://github.com/5ak3t/ej},
  note    = {Weights not yet published}
}
```

## Licence

- Code: **Apache-2.0** (`LICENSE`). Weights: **CC BY-SA 4.0** (`LICENSES/CC-BY-SA-4.0.txt`); adaptations must be shared
  under CC BY-SA 4.0 or a compatible licence. Base model `intfloat/e5-small-v2`: MIT (`LICENSES/MIT-e5-small-v2.txt`).
- Training data of ej 0.0.1: Typed Decisions (Apache-2.0), Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), GoEmotions
  (Apache-2.0) and an in-house ticket corpus written with Claude Haiku (not released); NLI teacher
  cross-encoder/nli-deberta-v3-xsmall (Apache-2.0). Creators, links and open licence questions: [NOTICE](NOTICE).
- "Jev" and "System One" are names used by TypeSafe.ai; ej is not affiliated with or endorsed by TypeSafe.ai.
