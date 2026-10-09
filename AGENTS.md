# AGENTS.md

Rules for contributors and coding agents working in this repository.

## Commands

```bash
python scripts/check_file_length.py          # every tracked text file <= 300 lines (licence texts excepted)
python -m pytest -q -rs                       # tests without weights (what CI runs)
EJ_WEIGHTS=/path/to/model.ejpack python -m pytest -q -rs   # plus the prediction tests
python -m ej.train --help; python -m ej.eval --help
```

## Rules

1. **Tests pass before a change is proposed.** Add a test for new behaviour; prefer tests that need no weights, and make
   weight-dependent tests skip unless `EJ_WEIGHTS` is set (optional: `EJ_REFERENCE_WEIGHTS`, a weights
   directory of the same state, for the cross-format parity test).
2. **No file longer than 300 lines.** Split by responsibility before the limit.
3. **No data, weights or caches in git**: no training pools, suite files, per-record predictions, score dumps, `*.pt`,
   `*.safetensors`, `*.ejpack`, training work directories. Only aggregate results go under `benchmarks/results/`.
4. **Claims need a command.** Every number in a document comes from a command that produced it (name the command or the
   results file). Development numbers are selected; final suites are read once per release. State orderings only where
   a CI excludes 0; say "not comparable" for latencies from different boxes, loads or days.
5. **Do not edit `ej/_runtime/`.** Its modules are sha256-pinned (`RUNTIME_SHA256`) and part of every weights release; a
   model change needs a new manifest and new weights. Training-side code goes in `ej/train/`.
6. **Licences.** Code may be copied only from MIT, Apache-2.0, BSD-2/3 or ISC sources, with the original header kept. Check
   every training source's upstream licence (not a mirror's metadata) and record it in `NOTICE` and the data card.
7. **Secrets.** Never print, copy or commit an API token or key; read them from files outside the repository.
8. **Determinism.** Seeded RNGs only; no wall-clock or float-derived values in keys or file names.
