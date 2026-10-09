# ej/_runtime: the prediction code

These 39 modules are the model code of ej: exactly the modules that load when a prediction runs (found by running a
prediction and listing the imported modules), taken from the maintainers' development tree at commit 46d0731 (named in
`RUNTIME_SHA256`). Only module docstrings, some function docstrings and comments were rewritten, and the development-machine
path defaults of a few environment variables were replaced by neutral ones under `~/.cache`; a syntax-tree comparison with
the source (docstrings ignored) shows no other difference, and predictions on the reference records are identical
(max |Δp| = 0.0). The same modules contain the fit path that `python -m ej.train` drives (flags unchanged: the released
configuration of ej 0.0.1).

- **Do not edit these files**: `ej.integrity.verify_runtime()` checks every file against `RUNTIME_SHA256` on each
  `ej.load()` and refuses to load on any difference, and a weights release records the sha256 of the `RUNTIME_SHA256` it
  was packaged for. A model change means a new runtime manifest and new weights.
- The modules import each other by **bare name** (`import student_lb`, ...). `ej.load()` puts this directory at the front of
  `sys.path`; a module of your own named like one of these files (e.g. `student.py`) would clash, and `ej.load()` refuses
  to continue when it detects that. Two fit-time modules live in `ej/train` and are registered under the bare names the fit
  code imports (`train_hcf` = `ej.train.hcf`, `student_hdn` = `ej.train.hdn`).
- Internal environment names: the modules read `EDGE_CKPT`, `EDGE_CACHE`, `EDGE_COLD`, `EDGE_DATA`. ej sets them from its
  public settings (`ej.load` from the weights directory and `EJ_CACHE`, `predict` from `EJ_COLD`, `ej.train` from its work
  directory), so the defaults in the code are not used.
- Importing the runtime sets `torch.manual_seed(0)` and `torch.set_num_threads(2)`; `ej.load()` restores the process's
  thread count and RNG state right after that import, and `Model.predict` sets threads only inside the call (`ej.scope`).
- After `ej.load()` each module's `torch.load` / `pickle.load` refuses, so no prediction-time code path can unpickle a file
  (`ej.scope.forbid_unpickling`).
