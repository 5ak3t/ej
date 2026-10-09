# Changelog

## 0.0.1.post1 (2026-10-09)

Packaging only; the model, its weights (`v0.0.1` on the Hub) and the code paths are those of 0.0.1.

- Published on PyPI as `ejai` (`pip install ejai`; the import name stays `ej`; the name `ej` is taken on PyPI).
- Dependencies are ranges (tested version as the lower bound, below the next major; transformers below 5.19, which needs
  torch >= 2.6) instead of exact pins.
- `release.yml` uploads the release files to PyPI by trusted publishing, and fills in a release created in the web UI.

## 0.0.1 (2026-10-09)

First public version. Weights: https://huggingface.co/5ak3t/ej, tag `v0.0.1`; the model is one file, `model.ejpack`
(11,384,312 bytes), and `ej.load("5ak3t/ej", revision="v0.0.1")` downloads and checks it. Architecture and method: technical
report forthcoming. Model: state `3b3e66d28fb423f9`, described in
`docs/model-cards/ej-0.0.1.md` and `docs/releases/0.0.1.md`.

- `ej` package: `ej.load(path_or_repo_id, revision=None, low_memory=False)` (a Hub repo id downloads only its
  `model.ejpack`, sha256-checked against `ej.integrity.KNOWN_PACK_FILES`), `Model.predict(records, threads=None, chunk_size=None)`, record validation
  (`ej.validate_records`, `ej.RecordError`), `ej.EXAMPLE_RECORD`, `ej.NOUL_OPTIONS`.
- Few-shot workflow adaptation: `Model.adapt(examples)` returns an `ej.AdaptedModel` with per-(question id, option key) logit
  offsets fitted on labelled records (the option tilt); `AdaptedModel.observe(batch)` folds in further labelled records and
  gives the same offsets as adapting on all of them at once. A label-free variant was evaluated and not included (its gain
  was not distinguishable from 0; `docs/adaptation.md`).
- Packed weights, the default format (`ej.pack`, ejpack v1): one file with a JSON header and binary sections, including
  the tokenizer and encoder config; header digest and every section sha256-checked before decoding; pickle-free (class
  allowlist); no base-model download. `low_memory=True` lowers peak memory; both runtimes give bit-identical predictions,
  also identical to the weights-directory format.
- Weights-directory format (secondary): safetensors plus a JSON skeleton, every file sha256-checked against `config.json`;
  it builds the encoder from the base model `intfloat/e5-small-v2` pinned to revision
  `ffb93f3bd4047442299a41ebb6fa998a38507c52`. The runtime modules are checked against `ej/_runtime/RUNTIME_SHA256`.
- Process hygiene: `ej.load` restores torch's thread count and RNG state after the runtime import and never patches
  `transformers` outside an ej call; threads are scoped to each predict call; after loading, no runtime module can unpickle;
  the encoder memo is an LRU of `memo_max` texts. Public environment names: `EJ_CACHE`, `EJ_COLD`.
- `ej.train`: `python -m ej.train encoder | teachers | distil | fit` (the training steps of this model, encoder on GPU or
  CPU) writing a pickle-free weights directory; `--seed` for
  replicate fits; `python -m ej.train export` writes its `model.ejpack`.
- `ej.eval`: `python -m ej.eval score` (NLL, micro accuracy, ECE15 with the perfect-calibration floor, certified automation
  with its per-suite scope, record-cluster CIs, group-macro accuracy with t intervals over workflows) and
  `python -m ej.eval compare` (seed-aware keep rule, TOST equivalence, selection accounting through a sha256-chained ledger).
- `benchmarks/`: runner with one latency protocol (threads checked inside the timed calls, box and load recorded),
  `pack_memory.py` (process memory and per-record latency of a pack), scoring,
  rival adapters, public suite builders with checksums, method and contamination notes, aggregate results of the final run.
- `scripts/`: `hf_layout.py` (Hub upload directory from a pack or a weights directory; never uploads), `state_key.py`,
  `check_file_length.py`.
- CI/CD: `.github/workflows/tests.yml` (push to `main`, pull requests, manual dispatch: tests without weights, and the full
  suite with the published weights downloaded from the Hub) and `.github/workflows/release.yml` (on a `v*` tag: tests,
  sdist + wheel, `SHA256SUMS`, GitHub Release).
