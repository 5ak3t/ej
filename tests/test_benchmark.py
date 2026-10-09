"""Benchmark scoring and runner helpers (no models needed)."""
import math

import pytest

import rivals
import run_bench
import scoring


def _suite():
    recs = []
    for i in range(60):
        recs.append({'id': f'r{i}', 'source': 'toy', 'state': f'case {i}',
                     'questions': {'q': {'type': 'choice', 'instructions': 'pick',
                                         'options': [{'key': 'a', 'text': 'A'}, {'key': 'b', 'text': 'B'}]}},
                     'gold': {'q': {'label': i % 2, 'probs': None}}})
    return recs


def test_summary_perfect_and_uniform():
    recs = _suite()
    perfect = [{'q': [1.0, 0.0] if r['gold']['q']['label'] == 0 else [0.0, 1.0]} for r in recs]
    s = scoring.summary(scoring.rows(recs, perfect))
    assert s['acc'] == 1.0 and s['nll'] == 0.0 and s['ece15'] == 0.0 and s['ca'] > 0.9 and s['questions'] == 60
    uniform = [{'q': [0.5, 0.5]} for _ in recs]
    s = scoring.summary(scoring.rows(recs, uniform))
    assert s['nll'] == round(math.log(2), 4) and s['ca'] == 0.0


def test_check_and_blind():
    r = _suite()[0]
    assert 'gold' not in scoring.blind(r)
    scoring.check({'q': [0.25, 0.75]}, r)
    with pytest.raises(AssertionError):
        scoring.check({'q': [0.2, 0.7]}, r)


def test_clopper_pearson_minimum_sample():
    assert scoring.betainc_upper(0, 22, 0.10) <= 0.10 < scoring.betainc_upper(0, 21, 0.10)


def test_sample_is_deterministic_and_registry_lists_ej():
    recs = _suite()
    assert [r['id'] for r in run_bench.sample(recs, 5)] == [r['id'] for r in run_bench.sample(list(reversed(recs)), 5)]
    assert 'ej' in rivals.RIVALS and all(len(v) == 4 for v in rivals.RIVALS.values())


def test_benchmark_scoring_equals_the_package_evaluator():
    import random

    from ej.eval import metrics
    recs, rng = _suite(), random.Random(3)
    preds = []
    for _ in recs:
        a = rng.random()
        preds.append({'q': [a, 1 - a]})
    assert scoring.summary(scoring.rows(recs, preds)) == metrics.summary(metrics.rows(recs, preds))
