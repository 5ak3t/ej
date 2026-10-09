"""Per-suite metrics of the ej evaluator (standard library only, except groups_report: numpy + scipy). The definitions are the ones behind every published ej number.

Suite file: JSONL records in the ej input format plus 'gold': {qid: {'label': option index, 'probs': [..] or None}}; optional
'source' and 'workflow' fields name the workflow a record belongs to (used by the group-macro report).
Per question: NLL of the gold label (the headline metric; lower is better), soft cross-entropy against gold distributions where
given, accuracy of the top option, ECE over 15 equal-width confidence bins, and CA = certified automation: the coverage, on half
B of the questions, of a confidence threshold certified on half A (fixed sequence over thresholds .99 -> .01, one-sided
Clopper-Pearson upper bound on the error rate <= RISK at level DELTA); the halves are fixed by sha256 of the record id. CA holds
for the suite it was certified on only."""
import hashlib
import json
import math

RISK, DELTA = 0.10, 0.10


def load(path):
    """Records of a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def blind(r):
    """The record without its gold answers (what a model is shown)."""
    return {k: v for k, v in r.items() if k not in ('gold', 'answers')}


def check(p, r):
    """Assert that p holds one distribution per question of r (right length, finite, >= 0, sums to 1 within 1e-6)."""
    for qid, q in r['questions'].items():
        v = p[qid]
        assert len(v) == len(q['options']) and all(x >= 0 and math.isfinite(x) for x in v) and abs(sum(v) - 1) < 1e-6, \
            (r.get('id'), qid)


def rows(recs, preds):
    """Per question: (record id, probs, gold label, gold probs or None)."""
    return [(r['id'], p[q], r['gold'][q]['label'], r['gold'][q].get('probs')) for r, p in zip(recs, preds) for q in r['questions']]


def betainc_upper(k, n, delta):
    """Clopper-Pearson one-sided upper bound on a binomial rate (bisection on the binomial tail)."""
    if n == 0:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > delta else (lo, mid)
    return hi


def certified_coverage(rs):
    """(coverage on half B, error rate on half B) of the threshold certified on half A; (0.0, None) when none certifies."""
    half = lambda x: int(hashlib.sha256(x[0].encode()).hexdigest()[:8], 16) % 2  # noqa: E731
    a, b = [x for x in rs if half(x) == 0], [x for x in rs if half(x) == 1]
    conf = lambda x: max(x[1])  # noqa: E731
    wrong = lambda x: max(range(len(x[1])), key=x[1].__getitem__) != x[2]  # noqa: E731
    n_min = next(n for n in range(1, 10 ** 4) if betainc_upper(0, n, DELTA) <= RISK)  # 22 at risk .10, delta .10
    best = None
    for t in [1 - i / 100 for i in range(1, 100)]:  # fixed sequence: strict -> lenient
        acc = [x for x in a if conf(x) >= t]
        if len(acc) >= n_min:
            if betainc_upper(sum(map(wrong, acc)), len(acc), DELTA) > RISK:
                break
            best = t
    if best is None or not b:
        return 0.0, None
    acc = [x for x in b if conf(x) >= best]
    return round(len(acc) / len(b), 4), round(sum(map(wrong, acc)) / max(len(acc), 1), 4)


def summary(rs):
    """{'nll', 'soft_ce', 'acc', 'ece15', 'ca', 'ca_risk', 'questions'} over the rows of one suite."""
    nll = sum(-math.log(max(p[y], 1e-15)) for _, p, y, _ in rs) / len(rs)
    soft = [-sum(g * math.log(max(x, 1e-15)) for g, x in zip(gp, p)) for _, p, _, gp in rs if gp]
    acc = sum(max(range(len(p)), key=p.__getitem__) == y for _, p, y, _ in rs) / len(rs)
    bins = [[0, 0, 0.0] for _ in range(15)]
    for _, p, y, _ in rs:
        j = max(range(len(p)), key=p.__getitem__)
        i = min(int(p[j] * 15), 14)
        bins[i][0] += 1
        bins[i][1] += j == y
        bins[i][2] += p[j]
    ca, ca_risk = certified_coverage(rs)
    return {'nll': round(nll, 4), 'soft_ce': round(sum(soft) / len(soft), 4) if soft else None, 'acc': round(acc, 4),
            'ece15': round(sum(abs(c - s) for n, c, s in bins if n) / len(rs), 4), 'ca': ca, 'ca_risk': ca_risk,
            'questions': len(rs)}


def group_of(r):
    """'source/workflow' of a record (the unit of the group-macro report)."""
    return f"{r.get('source')}/{r.get('workflow')}"


def group_acc(rs, grp):
    """{group: question accuracy of its rows}; grp maps record id -> 'source/workflow'."""
    c = {}
    for rid, p, y, *_ in rs:
        c.setdefault(grp[rid], []).append(max(range(len(p)), key=p.__getitem__) == y)
    return {g: sum(v) / len(v) for g, v in c.items()}


def groups_report(rs, grp, real=None):
    """Group-macro accuracy (per group: question accuracy; per source: mean over its groups; macro: mean over sources, sources
    weighted equally) with Welch-Satterthwaite t intervals over groups within sources (ej.eval.stats.macro_t). `real` = the
    sources of the primary statistic (macro_real; e.g. sources with independently checked labels), None = all sources."""
    from . import stats as ES  # numpy + scipy
    acc = group_acc(rs, grp)
    allt, bys = ES.macro_t(acc), ES.by_source_t(acc)
    prim = ES.macro_t(acc, real) if real else allt
    out = {'primary': 'macro_real' if real else 'macro_acc', 'macro_acc': allt['macro'], 'macro_ci': [allt['lo'], allt['hi']],
           'groups': len(acc), 'by_source': {}, 'ci_method': 'Welch-Satterthwaite t over groups within sources'}
    if real:
        out.update(macro_real=prim['macro'] if prim else None, macro_real_ci=[prim['lo'], prim['hi']] if prim else None,
                   macro_real_groups=prim['groups'] if prim else 0, macro_real_sources=prim['sources'] if prim else [])
    for s in sorted(bys):
        q = [x for x in rs if grp[x[0]].startswith(s + '/')]
        out['by_source'][s] = {'macro_acc': bys[s]['macro'], 'ci': [bys[s]['lo'], bys[s]['hi']], 'groups': bys[s]['groups'],
                               'acc': round(sum(max(range(len(p)), key=p.__getitem__) == y for _, p, y, *_ in q) / len(q), 4),
                               'questions': len(q)}
    return out
