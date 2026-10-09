# Final benchmark: ej 0.0.1 (2026-10-08)

ej 0.0.1 (state `3b3e66d28fb423f9`) was scored once on the final test suites, including zs_wide final; nothing was selected
on them. Scoring: `ej.eval` (the same `check` / `rows` / `summary` as `benchmarks/scoring.py`) with record-cluster bootstrap
CIs (1,000 draws) and the ECE floor of a perfectly calibrated model (500 draws). Files here: `ej-0.0.1.final.json` (all
aggregates of the run), `ej-0.0.1.vs-rivals.paired.json` (paired differences), `rivals/` (the rivals' sealed summaries),
`latency/` (per-record latency) and `adaptation/` (few-shot adaptation on the development suites). Aggregates only: no
per-record predictions, no suite inputs or labels.

## ej 0.0.1 per suite

| suite | records / questions | NLL [95% CI] | micro acc [95% CI] | ECE15 [95% CI] | ECE floor q95 → calibrated? | CA coverage / error (this suite only) |
|---|---|---|---|---|---|---|
| td | 300 / 1,500 | .6634 [.6132, .7157] | .7213 [.6953, .7460] | .0707 [.0501, .0944] | .0344 → no | .300 / .056 |
| zs_td | 100 / 500 | 1.1450 [1.1132, 1.1804] | .4220 [.3880, .4520] | .0935 [.0680, .1346] | .0592 → no | 0 (no threshold certified) |
| zs_massive | 1,000 / 1,000 | .4811 [.4297, .5302] | .8320 [.8080, .8570] | .0507 [.0405, .0731] | .0395 → no | .798 / .080 |
| tickets | 169 / 507 | .6291 [.5830, .6804] | .7120 [.6785, .7456] | .0334 [.0284, .0800] | .0607 → **yes** | 0 (no threshold certified) |
| tickets_ood | 147 / 441 | .7152 [.6503, .7871] | .6825 [.6394, .7256] | .0500 [.0391, .0991] | .0716 → **yes** | 0 (no threshold certified) |
| zs_wide | 1,764 / 3,702 | 1.1963 [1.1763, 1.2195] | .4217 [.4046, .4385] | .1214 [.1083, .1373] | .0274 → no | 0 (no threshold certified) |

"Calibrated" = observed ECE15 at or below the 95th percentile (q95) of a perfectly calibrated model's ECE15 at the same
confidences. Certified automation (CA) holds for that suite only. Mean NLL over td, zs_td, zs_massive and tickets: .7297.

**Unseen workflows (zs_wide final, 147 workflows never trained on).** Group-macro accuracy over the real sources
(macro_real; SNI, SGD and ABCD weighted equally, 105 workflows; Welch-Satterthwaite t interval over workflows within
sources): **.4187 [.3791, .4584]**; SNI .4557 [.4018, .5096] (79 workflows), SGD .5713 [.5286, .6140] (22), ABCD .2292
[.1022, .3561] (4). GLM-synthetic workflows, reported separately (their labels are not independently checked): .3583
[.3230, .3936] (42 workflows). No rival was run on zs_wide.

## Rivals (sealed runs of 2026-10-07 on the same five final suites)

The rival rows are `rivals/<suite>.<rival>.summary.json`. Contamination and fairness flags: `../METHOD.md` (laya, Julia-1 and Jev likely saw the zs_td workflow; the tickets
suites are ej's home turf; Jev's probabilities are rounded to 2 decimals).

| model | td | zs_td | zs_massive | tickets | tickets_ood |
|---|---|---|---|---|---|
| **ej 0.0.1** acc / NLL / ECE15 | .7213 / .6634 / .0707 | .4220 / 1.1450 / .0935 | .8320 / .4811 / .0507 | .7120 / .6291 / .0334 | .6825 / .7152 / .0500 |
| Jev 1.13.0 (hosted API) acc / NLL / ECE15 | .7367 / .6239 / .0377 | .7420 / .7497 / .0933 | .9570 / .2342 / .0204 | .7298 / 1.5360 / .1668 | .6259 / 2.4570 / .2421 |
| laya acc / NLL / ECE15 | .7660 / .6898 / .2014 | .7660 / .7582 / .2489 | .8660 / .5011 / .1570 | .6686 / .8410 / .0694 | .6168 / .9646 / .0717 |
| kev−0.8b acc / NLL / ECE15 | .4147 / 1.0848 / .0784 | .5200 / 1.0350 / .1050 | .8950 / .3805 / .1157 | .6785 / .7224 / .0315 | .6077 / .8564 / .0836 |
| OpenThai-SystemOne acc / NLL / ECE15 | .5153 / 1.2929 / .2941 | .6320 / .9007 / .1014 | .9710 / .1204 / .0128 | .6588 / 1.0699 / .2181 | .6168 / 1.1450 / .2247 |
| Julia−1 (llama.cpp BF16) acc / NLL / ECE15 | .7333 / 1.8139 / .2263 | .7060 / 2.0002 / .2460 | .7920 / .7302 / .1204 | .5128 / 1.9070 / .3314 | .4649 / 2.2996 / .4057 |
| gliclass-edge v3.0 acc / NLL / ECE15 | .3667 / 2.5769 / .4008 | .4840 / 1.9114 / .3532 | .5450 / 1.5201 / .2294 | .3037 / 2.6182 / .5037 | .3673 / 2.2584 / .4474 |

Paired differences, ej 0.0.1 minus rival (record-cluster bootstrap over the suite's records, 2,000 draws; each rival's
accuracy and NLL recomputed from the pairing equal its sealed summary). An ordering is stated only where the CI excludes 0.

| ej 0.0.1 − rival: Δ micro acc [95% CI]; Δ NLL [95% CI] | td | zs_td | zs_massive | tickets | tickets_ood |
|---|---|---|---|---|---|
| Jev 1.13.0 (hosted API) | −.015 [−.043, +.011]; +.040 [−.015, +.096] | −.320 [−.370, −.272]; +.395 [+.259, +.521] | −.125 [−.148, −.101]; +.247 [+.156, +.329] | −.018 [−.055, +.020]; −.907 [−1.194, −.642] | +.057 [+.002, +.113]; −1.742 [−2.087, −1.403] |
| laya | −.045 [−.065, −.023]; −.026 [−.067, +.012] | −.344 [−.394, −.294]; +.387 [+.355, +.420] | −.034 [−.059, −.009]; −.020 [−.062, +.020] | +.043 [+.006, +.087]; −.212 [−.270, −.154] | +.066 [+.014, +.118]; −.249 [−.332, −.164] |
| kev−0.8b | +.307 [+.263, +.347]; −.421 [−.476, −.367] | −.098 [−.162, −.034]; +.110 [+.081, +.140] | −.063 [−.089, −.037]; +.101 [+.060, +.140] | +.034 [−.004, +.071]; −.093 [−.144, −.042] | +.075 [+.025, +.127]; −.141 [−.219, −.065] |
| OpenThai-SystemOne | +.206 [+.166, +.247]; −.629 [−.742, −.520] | −.210 [−.264, −.160]; +.244 [+.164, +.327] | −.139 [−.163, −.117]; +.361 [+.313, +.409] | +.053 [+.010, +.097]; −.441 [−.562, −.323] | +.066 [+.011, +.118]; −.430 [−.553, −.315] |
| Julia−1 (llama.cpp BF16) | −.012 [−.033, +.009]; −1.151 [−1.323, −.978] | −.284 [−.338, −.228]; −.855 [−1.216, −.517] | +.040 [+.012, +.069]; −.249 [−.348, −.157] | +.199 [+.150, +.253]; −1.278 [−1.486, −1.087] | +.218 [+.154, +.277]; −1.585 [−1.832, −1.336] |
| gliclass-edge v3.0 | +.355 [+.324, +.386]; −1.913 [−2.068, −1.755] | −.062 [−.124, −.004]; −.766 [−.926, −.607] | +.287 [+.254, +.322]; −1.039 [−1.155, −.926] | +.408 [+.355, +.462]; −1.989 [−2.190, −1.803] | +.315 [+.254, +.376]; −1.543 [−1.732, −1.356] |

**Reading.** td: micro accuracy below laya, level with Jev and Julia-1, above kev-0.8b, OpenThai-SystemOne and
gliclass-edge v3.0. zs_td (one workflow, selected on, see below): micro accuracy below all six rivals. zs_massive: below
Jev, laya, kev-0.8b and OpenThai-SystemOne, above Julia-1 and gliclass-edge v3.0. tickets / tickets_ood (in-house): NLL
lower than every rival. No latency, size or calibration ranking is made: rival latencies were measured on another day
under unrecorded load (not comparable).

## Variability and selection

Six fits of the same code and data with different seeds (development suites) give a between-seed SD of .0105 in the mean
development NLL, .019 in zs_wide development macro_real and .037 in zs_td development micro accuracy. This release is the
default seed, fixed before fitting. Differences of that size between fits are not evidence of a better or worse model.

zs_td is ONE held-out workflow (security_incidents; dev 300 and final 100 records of the same workflow). The model code was
kept on its dev accuracy after about 55 logged comparisons on it, so final zs_td estimates accuracy on further records of a
workflow the model was selected on: it is neither unbiased nor unseen-workflow evidence. Its fit-to-fit SD (same code,
other RNG) is ~.022 accuracy per fit (SD of a difference .031, 4 pairs), twice its record-cluster SE (.011); one workflow
gives no between-workflow variance. Unseen-workflow evidence = zs_wide final macro_real.

Latency (one record per call, one torch thread): README "Latency" and `latency/`.
