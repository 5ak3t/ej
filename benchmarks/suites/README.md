# Benchmark suites: builders and checksums

No suite file ships with this repository. The three public suites can be rebuilt from the public datasets; the two ticket
suites come from a private corpus and are given by checksum only (`SHA256SUMS`, all nine suite files, development and final).

| file | what |
|---|---|
| `build_public.py` | rebuilds `td`, `zs_td`, `zs_massive` (development and final) and compares each file's sha256 with `SHA256SUMS` |
| `sources.py` | converters from the raw datasets to the ej record format (the converters that built the published suites) |
| `leakfree.py` | leak-free option sampling (conditional Poisson sampling, exchangeable option sets); only the raw-path default and an internal write guard were removed |
| `SHA256SUMS` | sha256 of every suite file used by the benchmark (`dev/` = development suites, `final/` = final test suites) |

```bash
pip install pyarrow numpy
# RAW: typed-decisions/all-train.parquet + all-test.parquet (LocalLLaMA/typed-decisions),
#      massive/validation.parquet + test.parquet (mteb/amazon_massive_intent, en)
python benchmarks/suites/build_public.py RAW OUT --split dev     # development suites
python benchmarks/suites/build_public.py RAW OUT --split final   # final suites: verify the checksums, do not tune on them
```

Construction: `td` = the 3 training workflows of Typed Decisions (development: train split, sha256 hash bucket 0 of 5 of the
record id; final: test split); `zs_td` = workflow `security_incidents` (development: train split; final: test split);
`zs_massive` = 1,000 MASSIVE records in seeded hash order with leak-free option sets (development: validation; final:
test). The record ids of the final suites are the ids the builder writes (Typed Decisions `id`, `massive-<split>-<id>`).

Verified 2026-10-08: `python benchmarks/suites/build_public.py RAW OUT --split dev` reproduced the sha256 of all three public
development suites (`dev/td`, `dev/zs_td`, `dev/zs_massive`: `matches_SHA256SUMS: true`). The final suites were not rebuilt
in that session (they are read once per release); their checksums come from the data manifest.
