# ej benchmark

Code and aggregate results of the benchmark that compares ej with other typed-decision models. Method, rival choice and
contamination audit: [METHOD.md](METHOD.md). Suite builders and checksums: [suites/](suites/). Results: [results/](results/).

## What it measures

Every model gets the same records (a `state` plus typed questions, the ej input format, with the gold answers removed)
and must return one probability distribution per question. `scoring.py` scores them per suite:

| metric | meaning |
|---|---|
| NLL | mean negative log-probability of the gold option (lower is better; the headline metric) |
| acc | micro accuracy: share of questions whose most probable option is the gold |
| ECE15 | expected calibration error over 15 confidence bins (compare it with the perfect-calibration floor of the same suite before calling a model calibrated) |
| CA | certified automation: coverage on one half of the questions of a confidence threshold certified on the other half to keep the error rate <= 10% (Clopper-Pearson, delta 0.10); valid for that suite only |
| soft_ce | cross-entropy against gold distributions, where a suite provides them |
| ms/record | wall-clock latency per record, one record per call, after a warm-up record; the summary records the torch threads observed inside the timed calls, the box and the load average |

## Suites: public sources vs private data

| suite | records / questions | source | public? |
|---|---|---|---|
| td | 300 / 1,500 | Typed Decisions **test** split, the three workflows ej was trained on | public dataset (Apache-2.0); builder in `suites/` |
| zs_td | 100 / 500 | Typed Decisions test split, workflow `security_incidents` (not trained on, but its development records were used for model selection) | public dataset; builder in `suites/` |
| zs_massive | 1,000 / 1,000 | MASSIVE (en) intent **test** split, options sampled leak-free; about 22% of its intent names overlap CLINC150 / Banking77 options in ej's pool | public dataset (CC BY 4.0); builder in `suites/` |
| tickets | 169 / 507 | private support-ticket corpus, test split (same distribution as part of ej's training data) | **private** |
| tickets_ood | 147 / 441 | private support-ticket corpus, tickets in held-out writing styles (a style shift, same generator and departments) | **private** |

The final suites are scored once per release. zs_td's workflow was used for model selection on its development records,
so its final numbers are not unseen-workflow evidence. **No suite file ships with this repository**: the public suites
can be rebuilt from the public datasets with the builders in `suites/`, and their sha256 is in `suites/SHA256SUMS`; the
ticket suites cannot (private corpus). `run_bench.py` runs unchanged on any suite file in the ej input format with gold
answers (`scoring.py` docstring), for example your own labelled records.

## Files

| path | what |
|---|---|
| `run_bench.py` | runner: blind records, one record per call, `scoring.check`, summary JSON per model and suite (threads observed, box, load average) |
| `scoring.py` | NLL, accuracy, ECE15, soft cross-entropy, certified automation |
| `rivals/` | one adapter per model (`rivals/__init__.py` lists them with the environment each needs) |
| `suites/` | builders of the public suites (td, zs_td, zs_massive), the leak-free sampler, sha256 of every suite file |
| `METHOD.md` | rival set, contamination audit, per-suite fairness, adapter rules, methodological choices |
| `results/` | final benchmark of ej 0.0.1 ([results/README.md](results/README.md)): aggregates, paired rival differences, the rivals' sealed summaries (`rivals/`), latency (`latency/`), few-shot adaptation (`adaptation/`) |
