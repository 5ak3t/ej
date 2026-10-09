# Benchmarks

Summary of the final benchmark of ej 0.0.1 (2026-10-08). Full tables, paired differences and the variability analysis:
[benchmarks/results/README.md](benchmarks/results/README.md); method, rival choice and contamination flags:
[benchmarks/METHOD.md](benchmarks/METHOD.md); code: [benchmarks/](benchmarks/README.md).

## Protocol

- Same blind records for every model (gold removed), one record per call after a warm-up record, torch threads set to 1 and
  checked inside the timed calls (`benchmarks/run_bench.py`); one scoring module for everyone (`benchmarks/scoring.py`, equal
  to `ej.eval.metrics`).
- Metrics per suite: NLL of the gold option (headline; lower is better), micro accuracy, ECE15 next to the ECE a perfectly
  calibrated model would show on the same suite, certified automation (CA; valid for that suite only).
- Final suites are read once per release. Every cell has a 95% record-cluster CI; every ej-vs-rival difference has a paired
  CI, and an ordering is stated only where it excludes 0. Development suites drove model selection, so development
  numbers are not reported as results.
- No suite file ships: td, zs_td and zs_massive are rebuilt from the public datasets (`benchmarks/suites/`, sha256 in
  `SHA256SUMS`); the ticket suites come from a private corpus.

## ej 0.0.1 on the final suites

| suite | what it tests | records / questions | NLL [95% CI] | micro accuracy [95% CI] | calibrated on this suite? |
|---|---|---|---|---|---|
| td | Typed Decisions test split, the 3 trained workflows | 300 / 1,500 | .663 [.613, .716] | .721 [.695, .746] | no |
| zs_td | one held-out workflow (`security_incidents`); used for model selection | 100 / 500 | 1.145 [1.113, 1.180] | .422 [.388, .452] | no |
| zs_massive | MASSIVE intents, leak-free option sets; never trained on, label names partly overlap the pool | 1,000 / 1,000 | .481 [.430, .530] | .832 [.808, .857] | no |
| tickets | in-house support tickets (in-domain) | 169 / 507 | .629 [.583, .680] | .712 [.679, .746] | yes |
| tickets_ood | tickets in held-out writing styles | 147 / 441 | .715 [.650, .787] | .683 [.639, .726] | yes |
| zs_wide | 147 workflows never trained on | 1,764 / 3,702 | 1.196 [1.176, 1.220] | .422 [.405, .439] | no |

Unseen-workflow headline: zs_wide **macro_real .419 [.379, .458]** (group-macro accuracy over the real sources SNI, SGD and
ABCD, 105 workflows; t interval over workflows within sources); GLM-synthetic workflows separately .358 [.323, .394].

## Against the rivals (micro accuracy; sealed rival runs of 2026-10-07 on the same suites)

| model | td | zs_td | zs_massive | tickets | tickets_ood |
|---|---|---|---|---|---|
| **ej 0.0.1** | .721 | .422 | .832 | .712 | .683 |
| Jev 1.13.0 (hosted API) | .737 | .742 | .957 | .730 | .626 |
| laya | .766 | .766 | .866 | .669 | .617 |
| kev-0.8b | .415 | .520 | .895 | .679 | .608 |
| OpenThai-SystemOne | .515 | .632 | .971 | .659 | .617 |
| Julia-1 (llama.cpp BF16) | .733 | .706 | .792 | .513 | .465 |
| gliclass-edge v3.0 | .367 | .484 | .545 | .304 | .367 |

Read with the flags of `benchmarks/METHOD.md`: laya, Julia-1 and Jev likely saw the zs_td workflow; OpenThai-SystemOne
trained on MASSIVE; the ticket suites are ej's home turf. With paired CIs: td below laya, level with Jev and Julia-1, above
kev-0.8b, OpenThai-SystemOne and gliclass-edge; zs_td below all six; zs_massive below Jev, laya, kev-0.8b and
OpenThai-SystemOne, above Julia-1 and gliclass-edge; on tickets and tickets_ood ej's NLL is lower than every rival's.

## Latency

ej 0.0.1 on the td development suite, one record per call, 1 thread: warm 356.6 ms mean (363.1 median), cold 1,922.0 ms
mean (1,413.7 median), measured with the weights-directory format; box and load in the README and
`benchmarks/results/latency/`. The packed model's memory and per-record latency (default and `low_memory=True`) were
measured on another day with `benchmarks/pack_memory.py`'s procedure (README "Memory and latency of the packed model"); they
are not comparable with the numbers above. Rival latencies were measured on another day under unrecorded load: **not
comparable**, not ranked.
