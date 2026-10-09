"""Score suites with an ej model (model.ejpack or a weights directory) or with saved predictions: the report behind ``python -m ej.eval score``.

Each suite is predicted blind (gold removed), every distribution is checked, and the suite gets the metrics of ej.eval.metrics
plus record-cluster 95% CIs, the perfect-calibration ECE floor and CA's scope (ej.eval.stats.suite_stats). A suite named with
``groups`` is reported as group-macro accuracy over its 'source/workflow' groups and is left out of `metric` (the mean over the
other suites of the per-suite mean NLL)."""
import json
import os
import time

from . import metrics as M
from . import stats as ES


def suite_name(path):
    """'td' for '.../td.jsonl'."""
    return os.path.basename(path).split('.')[0]


def read_preds(path, recs):
    """Predictions in suite order from a JSONL file of {'id', 'probs': {qid: [..]}} (benchmarks/run_bench.py output)."""
    by = {}
    for row in M.load(path):
        by[str(row['id'])] = row['probs']
    missing = [r['id'] for r in recs if str(r['id']) not in by]
    if missing:
        raise ValueError(f'{path}: no prediction for {len(missing)} records, e.g. {missing[:3]}')
    return [by[str(r['id'])] for r in recs]


def predictor(weights, threads=None):
    """predict(records) of an ej model (loaded once)."""
    import ej
    model = ej.load(weights)
    return lambda recs: model.predict(recs, threads=threads)


def score(suites, predict=None, preds=None, groups=None, real=None, draws=1000):
    """(report, dump). suites: suite file paths; predict: callable on blind records, or preds: one predictions file per suite.
    dump = {suite: [(record id, probs, gold label)]} (the input of ej.eval compare; it holds predictions, keep it private when
    the suite is)."""
    out, dump, lat = {}, {}, {}
    for k, path in enumerate(suites):
        s, recs = suite_name(path), M.load(path)
        t0 = time.time()
        p = predict([M.blind(r) for r in recs]) if predict else read_preds(preds[k], recs)
        lat[s] = round((time.time() - t0) * 1000 / max(len(recs), 1), 2)
        for x, r in zip(p, recs):
            M.check(x, r)
        rs = M.rows(recs, p)
        out[s] = {**M.summary(rs), **ES.suite_stats(rs, draws), 'records': len(recs)}
        if s == groups:
            out[s].update(M.groups_report(rs, {r['id']: M.group_of(r) for r in recs}, real))
        dump[s] = [(rid, pr, y) for rid, pr, y, _ in rs]
    main = [s for s in out if s != groups]
    report = {'metric': round(sum(out[s]['nll'] for s in main) / len(main), 4) if main else None, 'metric_suites': main, **out}
    if predict:
        report['ms_per_record_batched'] = lat
    return report, dump


def write_dump(dump, path):
    """Write a score dump (JSON)."""
    with open(path, 'w') as f:
        json.dump(dump, f)
