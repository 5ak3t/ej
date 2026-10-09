# Training

`python -m ej.train` trains an ej model from a pool of labelled records and writes a weights directory, which
`python -m ej.train export` turns into the packed model file `model.ejpack`; `ej.load` reads either. It runs the released ej 0.0.1 recipe: the fit code is `ej/_runtime/student.py` with its flags unchanged; `ej.train`
only prepares the work directory, the teachers and the encoder, and exports the result pickle-free.

Architecture and method: technical report forthcoming.

```bash
pip install -e '.[train]'
python -m ej.train fit --pool pool.jsonl --out my-weights          # every step; work directory my-weights.work
python -m ej.train export --weights my-weights --out model.ejpack   # the packed model file ej.load takes by default
python -m ej.eval score --suite dev_suite.jsonl --weights model.ejpack
```

## The pool (input format)

One JSON object per line: an ej input record plus the fields training needs.

```json
{"id": "banking77-train-00042", "source": "banking77", "workflow": null,
 "state": "I still have not received my new card, what should I do?",
 "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                          "options": [{"key": "card_arrival", "text": "card arrival"},
                                      {"key": "lost_or_stolen_card", "text": "lost or stolen card"}]}},
 "gold": {"intent": {"label": 0, "probs": null}}}
```

- `id` (string, unique) and `source` are required; `workflow` is optional. The held-out unit of every cross-fit is the
  group `source/workflow`, so give records of one task the same group and different tasks different groups. With more than
  12 groups, groups are blocked into 8 folds automatically.
- `gold.label` is the index of the correct option; `gold.probs` (optional) is a teacher distribution over the options,
  used where present.
- Options are read as text: write short descriptions. `noul` questions have exactly the options `false` / `true`.
- The fit holds out 10% of the states (by sha256 of the state text) for calibration; nothing else is needed.
- `python -m ej.train` validates every record before training (`ej.train.fit.check_pool`).

## Public sources (and their licences)

The ej 0.0.1 training pool (9,719 records, 7 groups) combined the sources below; converters that produce this format are in
`benchmarks/suites/sources.py`, and `benchmarks/suites/leakfree.py` draws distractor options so that option frequency
reveals nothing about the gold label (conditional Poisson sampling; `docs/data-card.md` §4).

| Source | Hugging Face dataset | Licence | ej 0.0.1 records |
|---|---|---|---|
| Typed Decisions (3 workflows; `security_incidents` held out) | `LocalLLaMA/typed-decisions` | Apache-2.0 | 736 |
| Banking77 | `PolyAI/banking77` | CC BY 4.0 | 3,000 |
| CLINC150 (`plus`) | `clinc/clinc_oos` | CC BY 3.0 | 3,000 |
| GoEmotions (simplified, single-label, balanced) | `google-research-datasets/go_emotions` | Apache-2.0 | 2,203 |
| support tickets written by Claude Haiku | not released | in-house | 780 |

The ticket corpus is not released, so a pool rebuilt from public data reproduces the recipe but not the ej 0.0.1 state.
Check the upstream licence of every source you add, not a mirror's metadata (one mirror of Amazon counterfactual declares
CC BY 4.0; upstream it is CC BY-NC 4.0, so ej does not use it). Keep evaluation data out of the pool: for the published suites that means Typed
Decisions `security_incidents`, MASSIVE and the zs_wide workflows.

## Steps

Each step reads the pool only and memoises its output under `<work>/ckpt/` by a sha256 of the code and the pool, so a
finished step is never repeated and an interrupted run resumes. `fit` trains whatever is missing in-process.

| Command | Produces (under `<work>/ckpt/`) | Device |
|---|---|---|
| `python -m ej.train encoder --pool P --work W` | the encoder checkpoint `lowbit-<key>/w23.pt` (`ej.train.lowbit`) | GPU strongly recommended (`EJ_DEVICE`) |
| `python -m ej.train teachers --pool P --work W` | the teachers: full model + one per group fold (`train_merge`), and one per fold pair (`ej.train.hcf`) | CPU |
| `python -m ej.train distil --pool P --work W` | distilled checkpoints `dn-*.pt` and `dd-*.pt` (downloads `cross-encoder/nli-deberta-v3-xsmall`) | CPU |
| `python -m ej.train fit --pool P --work W --out OUT` | the fitted state, written to `OUT` (`state.*`, `encoder/w23.*`, `config.json`) | CPU |

`--parts` runs single teachers (`full`, `fold0`..`fold3`, `p01`..`p23`), so they can run in parallel processes on one work
directory. Environment: `EJ_DEVICE=cuda|cpu`, `EJ_THREADS` (torch threads, default 2), `EJ_CACHE` (encoder cache; default
`<work>/cache`), `HF_HOME` (base models: `intfloat/e5-small-v2`, and for the teacher `cross-encoder/nli-deberta-v3-xsmall`).
Compute, for scale: the teachers take about an hour of CPU in total, and a fit of this code with every
checkpoint present took about 46 minutes on a 4-core CPU for a pool of about 12,700 records (2,785.7 s, recorded in the
fitted state's metadata).

## Output and its key

`OUT/config.json` records the ej version, the runtime manifest sha256, the e5 revision, the encoder directory, the pool
sha256 and the sha256 of every file; `ej.load(OUT)` checks them all. The state key is a sha256 over the state format
version, every runtime `student*` / `train_*` module, every `ej.train` module and the pool bytes
(`ej.train.export.state_key`): a new pool or a code change gives a new key. The training checkpoints under `<work>` are
pickles written by this training; only the exported weights directory is pickle-free. Never commit the work directory.

`python -m ej.train export --weights OUT --out model.ejpack` writes the packed model file, the default format of `ej.load`:
one file, pickle-free on both sides, with the trimmed tokenizer and the encoder
config inside, so loading it downloads nothing. The export reads the base tokenizer and config once (no weights); exporting
the 0.0.1 weights directory reproduces the release pack's content digest (`tests/test_pack_weights.py`).

`python scripts/hf_layout.py UPLOAD_DIR --pack model.ejpack` builds a Hugging Face upload directory for a pack (the file,
the model card with Hub metadata, NOTICE, licences, `SHA256SUMS`); `--from-safe OUT` does the same for a weights directory.
It never uploads. The 0.0.1 release lives at https://huggingface.co/5ak3t/ej (revision `v0.0.1`).

## Comparing recipes

Fits of the same code differ through their seed (for ej 0.0.1's code: SD .0065 of a difference in mean development NLL between two
fits). To decide whether a change helps, fit each arm S times (`--seed 1` .. `--seed S`, each seed in its own `--work`; copy
the `lowbit-*` encoder checkpoint into each to share it), score every fit on the same development
suites with `python -m ej.eval score --dump`, and run `python -m ej.eval compare --ref ... --cand ... [--ledger L --tag T]`:
it reports BETTER / EQUAL / WORSE / INCONCLUSIVE with intervals that include fit-to-fit noise, and with a ledger it counts
every comparison you made for the Bonferroni-adjusted p and the best-of-N optimism. Read final suites once per release.
