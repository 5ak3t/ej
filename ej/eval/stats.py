"""Evaluation statistics of ej.eval (numpy + scipy).

  macro_t       group-macro accuracy (per source: mean over its groups; sources weighted equally) with a Welch-Satterthwaite t
                interval: Var = sum_s var_s / (G_s S^2), df combined from the per-source G_s - 1 (coverage >= .94 in simulation
                on a 154-workflow design, tests/test_eval_stats.py, where a percentile group bootstrap reached .908). A source
                with < 2 groups gives NO interval (None), never a zero-width one.
  cluster_ci    record-cluster bootstrap of a suite's nll / acc / ece15 (all questions of a record resampled together).
  ece_floor     the ECE15 a PERFECTLY calibrated model would show at this n and these confidences (labels redrawn from the
                model's own top-label probability): mean and 95th percentile. 'calibrated' = observed <= q95.
  certify / ca_transfer   CA's guarantee is PER SUITE (threshold certified on labelled rows of that suite); the risk of a threshold
                certified on suite A, measured on suite B, is reported, and a cross-suite claim is allowed only if it stays within
                the same certificate.
  ledger_*      append-only, sha256-chained ledger of evaluation runs (selection accounting for ej.eval compare)."""
import datetime
import hashlib
import json
import math
import os

import numpy as np
from scipy import stats as sps

REAL = ('sni', 'sgd', 'abcd')  # zs_wide sources with real labels (the GLM-synthetic source is reported separately)
RISK, DELTA = 0.10, 0.10


def tq(level, df):
    """Two-sided Student-t quantile (normal for df = inf)."""
    q = 1 - (1 - level) / 2
    return float(sps.norm.ppf(q)) if not df or not math.isfinite(df) else float(sps.t.ppf(q, df))


def satterthwaite(parts):
    """(V, df) for V = sum of independent variance parts [(v_i, df_i)] (df_i = inf allowed)."""
    V = sum(v for v, _ in parts)
    den = sum(v * v / d for v, d in parts if math.isfinite(d) and d > 0)
    return V, (V * V / den if den > 0 else math.inf)


def macro_t(acc, keep=None, level=0.95):
    """{'macro', 'lo', 'hi', 'df', 'groups', 'sources'} over {'source/group': value}; keep = sources to include (None = all)."""
    by = {}
    for g in sorted(acc):
        s = g.split('/', 1)[0]
        if keep is None or s in keep:
            by.setdefault(s, []).append(float(acc[g]))
    if not by:
        return None
    S = len(by)
    m = sum(sum(v) / len(v) for v in by.values()) / S
    out = {'macro': round(m, 4), 'lo': None, 'hi': None, 'df': None, 'groups': sum(map(len, by.values())), 'sources': sorted(by)}
    if any(len(v) < 2 for v in by.values()):
        return out
    V, df = satterthwaite([(float(np.var(v, ddof=1)) / len(v) / S ** 2, len(v) - 1) for v in by.values()])
    h = tq(level, df) * math.sqrt(V)
    out.update(lo=round(m - h, 4), hi=round(m + h, 4), df=round(df, 1) if math.isfinite(df) else None)
    return out


def by_source_t(acc, level=0.95):
    """{source: macro_t of that source alone} (per-source CIs)."""
    return {s: macro_t(acc, (s,), level) for s in sorted({g.split('/', 1)[0] for g in acc})}


def _arrays(rs):
    """Question arrays of metric rows [(rid, probs, y, ...)]: record index, top-label conf, correct, nll."""
    rid = {}
    ri = np.array([rid.setdefault(r[0], len(rid)) for r in rs])
    conf = np.array([max(r[1]) for r in rs])
    corr = np.array([max(range(len(r[1])), key=r[1].__getitem__) == r[2] for r in rs], dtype=float)
    nll = np.array([-math.log(max(r[1][r[2]], 1e-15)) for r in rs])
    return ri, conf, corr, nll, len(rid)


def ece15(conf, corr, w=None):
    """Top-label ECE over 15 equal-width bins (ej.eval.metrics.summary's definition), optional question weights w."""
    w = np.ones_like(conf) if w is None else w
    b = np.minimum((conf * 15).astype(int), 14)
    return float(np.abs(np.bincount(b, w * corr, 15) - np.bincount(b, w * conf, 15)).sum() / w.sum())


def cluster_ci(rs, draws=1000, seed=0, level=0.95):
    """Record-cluster bootstrap percentile CIs of nll, acc, ece15 for one suite's rows."""
    ri, conf, corr, nll, R = _arrays(rs)
    rng = np.random.default_rng(seed)
    out = {'nll': [], 'acc': [], 'ece15': []}
    for _ in range(draws):
        w = np.bincount(rng.integers(0, R, R), minlength=R)[ri].astype(float)
        out['nll'].append(float((w * nll).sum() / w.sum()))
        out['acc'].append(float((w * corr).sum() / w.sum()))
        out['ece15'].append(ece15(conf, corr, w))
    a = (1 - level) / 2
    return {k: [round(float(np.quantile(v, a)), 4), round(float(np.quantile(v, 1 - a)), 4)] for k, v in out.items()}


def ece_floor(rs, reps=500, seed=0):
    """ECE15 of a perfectly calibrated model with these confidences: labels redrawn so P(correct) = top-label probability."""
    _, conf, corr, _, _ = _arrays(rs)
    rng = np.random.default_rng(seed)
    sims = np.array([ece15(conf, (rng.random(len(conf)) < conf).astype(float)) for _ in range(reps)])
    obs = ece15(conf, corr)
    return {'ece15': round(obs, 4), 'null_mean': round(float(sims.mean()), 4), 'null_q95': round(float(np.quantile(sims, .95)), 4),
            'calibrated': bool(obs <= np.quantile(sims, .95))}


def suite_stats(rs, draws=1000):
    """Per-suite additions to the metrics: record-cluster CIs, the ECE noise floor, CA's scope."""
    return {'ci': cluster_ci(rs, draws), 'ece_floor': ece_floor(rs), 'ca_scope': 'this suite only (threshold certified on half A '
            'of this suite, measured on half B; not a guarantee for other suites or workflows)'}


def cp_upper(k, n, delta=DELTA):
    """One-sided Clopper-Pearson upper bound at level 1 - delta."""
    return 1.0 if n == 0 else 1.0 if k >= n else float(sps.beta.ppf(1 - delta, k + 1, n - k))


def certify(conf, wrong, risk=RISK, delta=DELTA):
    """metrics.certified_coverage's fixed-sequence threshold on calibration rows (conf, wrong) -> threshold or None."""
    conf, wrong = np.asarray(conf, float), np.asarray(wrong, bool)
    n_min = next(n for n in range(1, 10 ** 4) if cp_upper(0, n, delta) <= risk)
    best = None
    for t in [1 - i / 100 for i in range(1, 100)]:
        acc = conf >= t
        if acc.sum() >= n_min:
            if cp_upper(int(wrong[acc].sum()), int(acc.sum()), delta) > risk:
                break
            best = t
    return best


def ca_transfer(rows_a, rows_b, risk=RISK, delta=DELTA):
    """Certify on suite A (all rows), apply to suite B: {'t', 'coverage_b', 'risk_b', 'claim_ok'}; claim_ok needs B's realised
    risk's Clopper-Pearson upper bound <= risk (the same certificate the within-suite claim has). No threshold -> no claim."""
    _, ca, ka, _, _ = _arrays(rows_a)
    _, cb, kb, _, _ = _arrays(rows_b)
    t = certify(ca, 1 - ka, risk, delta)
    if t is None:
        return {'t': None, 'coverage_b': 0.0, 'risk_b': None, 'claim_ok': False}
    acc = cb >= t
    n, k = int(acc.sum()), int((1 - kb[acc]).sum())
    return {'t': t, 'coverage_b': round(n / len(cb), 4), 'risk_b': round(k / max(n, 1), 4),
            'claim_ok': bool(n > 0 and cp_upper(k, n, delta) <= risk)}


def claim_allowed(certified_on, applied_to):
    """CA's guarantee is a per-suite (per-workflow) statement: a threshold certified on one suite is never claimed for another."""
    return certified_on == applied_to


def _last_hash(path):
    if not os.path.exists(path):
        return '0' * 64
    last = ''
    with open(path, 'rb') as f:
        for line in f:
            if line.strip():
                last = line.rstrip(b'\n')
    return hashlib.sha256(last).hexdigest() if last else '0' * 64


def ledger_append(path, row):
    """Append one JSON row with 'prev' = sha256 of the previous line (chain) and a UTC time; returns the row written."""
    row = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), **row, 'prev': _last_hash(path)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
    return row


def ledger_rows(path):
    """All rows of a ledger (empty when the file does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ledger_verify(path):
    """True iff every row's 'prev' is the sha256 of the line before it (no row edited, removed or reordered)."""
    prev = '0' * 64
    with open(path, 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            if json.loads(line).get('prev') != prev:
                return False
            prev = hashlib.sha256(line.rstrip(b'\n')).hexdigest()
    return True
