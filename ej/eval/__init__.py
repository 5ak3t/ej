"""ej.eval: the evaluator behind every published ej number.

    python -m ej.eval score --suite td.jsonl --suite zs_wide.jsonl --weights DIR --groups zs_wide --dump run.json
    python -m ej.eval compare --ref ref_s1.json ref_s2.json --cand cand_s1.json cand_s2.json

``score`` (ej.eval.score): blind prediction, distribution checks, NLL / accuracy / ECE15 / certified automation per suite
(ej.eval.metrics), record-cluster CIs and the perfect-calibration ECE floor (ej.eval.stats), and group-macro accuracy with t
intervals over workflows for a multi-workflow suite. ``compare`` (ej.eval.compare): the seed-aware keep rule for two arms of
S fits each (BETTER / EQUAL / WORSE / INCONCLUSIVE), with selection accounting through an optional run ledger. No suite path
is built in: every suite is a file you pass. stats and compare need numpy + scipy (``pip install 'ej[eval]'``)."""
