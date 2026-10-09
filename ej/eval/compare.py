"""Seed-aware comparison of two models (or two training recipes) on the same suites: ``python -m ej.eval compare``.

Two fits of the same code differ through their random seed, and that fit-to-fit noise is about twice the record-level standard
error of a single fit on ej's development suites, so a single-fit record bootstrap keeps null changes far too often. Design:
S fits per arm (seed i of both arms shares its seed), each scored once with ``python -m ej.eval score --dump`` on the same
records. ESTIMAND: Delta = E_fit[metric(cand)] - E_fit[metric(ref)] (metric = mean over the chosen suites of the per-suite
question-mean NLL). ESTIMATOR: d = mean_s D_s, D_s = metric(cand, s) - metric(ref, s). Seeds x records are CROSSED (every fit
scores every record); per suite the record contribution is z_sr = (delta_sr - d_ks n_r) / nbar_k (ratio-mean linearisation).
Method of moments:
  E[s_D^2] = sa^2 + se^2/R,   E[v_each] = (sm^2 + se^2)/R,   E[v_mean] = sm^2/R + se^2/(S R),   Var(d) = sa^2/S + sm^2/R + se^2/(S R)
  => V = s_D^2/S - (v_each - v_mean)/(S - 1) + v_mean   (seed part clamped at 0, so V >= v_mean)
(s_D^2: between-seed variance of D_s; v_each: mean single-fit record variance; v_mean: record variance of the seed-mean difference).
df = Welch-Satterthwaite (seed part S - 1, record parts R_k - 1). S = 1: the seed part is a PRIOR fit-noise SD (``--prior-sd``,
default SD_FIT, measured for ej 0.0.1's code on its four development suites with 3 df), never silently zero.
RULE (fixed before looking at the result):
  BETTER  d <= -KEEP_D and the 95% upper bound < 0;   EQUAL  TOST: the 90% CI inside (-EQ_MARGIN, +EQ_MARGIN);
  WORSE   the 95% lower bound > 0;                     INCONCLUSIVE otherwise (no keep, no 'equal').
SELECTION: with ``--ledger`` every compare run is appended to a sha256-chained ledger first; the report adds the one-sided p
of BETTER, its Bonferroni adjustment over the N comparisons logged under the same ``--tag`` and the best-of-N optimism m_N * se."""
import math

import numpy as np
from scipy import stats as sps

from . import stats as ES

KEEP_D, EQ_MARGIN = 0.010, 0.010  # quality bar and TOST margin on the metric (mean NLL)
SD_FIT, DF_FIT = 0.0065, 3  # prior fit-noise SD of a metric DIFFERENCE (4 same-code fit pairs), used only when S = 1
ACC_D = 0.02  # accuracy bar for the accuracy verdicts (per-suite accuracy, group-macro accuracy)
SIM_R, SIM_Q = (104, 164, 300, 1000), (312, 820, 1500, 1000)  # simulation design: records / questions per suite


def estimate(D, Z, ws, prior=None):
    """Crossed seeds x units. D (S,) per-seed statistic; Z [(S, U_k)] per-unit contributions per stratum; ws stratum weights."""
    D = np.asarray(D, float)
    S, d = len(D), float(np.mean(D))
    rec = [(w * w * float(np.var(z.mean(0), ddof=1)) / z.shape[1], z.shape[1] - 1) for w, z in zip(ws, Z) if z.shape[1] > 1]
    v_mean = sum(v for v, _ in rec)
    if S >= 2:
        sD2 = float(np.var(D, ddof=1))
        v_each = sum(w * w * float(np.mean(np.var(z, axis=1, ddof=1))) / z.shape[1] for w, z in zip(ws, Z) if z.shape[1] > 1)
        seed = (max(sD2 / S - max(v_each - v_mean, 0.0) / (S - 1), 0.0), S - 1)
        noisy = (sD2 / S, S - 1)  # V's sampling noise is s_D^2's (it carries the record noise too), not the corrected part's
    else:
        sD2, seed = None, (prior ** 2, DF_FIT) if prior is not None else (0.0, math.inf)
        noisy = seed
    V = seed[0] + v_mean
    den = sum(v * v / k for v, k in [noisy] + rec if math.isfinite(k) and k > 0)
    df = min(V * V / den, 1e6) if den > 0 else math.inf
    return {'d': d, 'se': math.sqrt(V), 'df': df, 'S': S, 'seed_var': seed[0], 'rec_var': v_mean, 'sd_D': None if sD2 is None
            else math.sqrt(sD2), 'fit_noise': 'measured' if S >= 2 else 'prior' if prior is not None else 'NOT included'}


def verdict(e, keep_d=KEEP_D, eq=EQ_MARGIN, lower_better=True):
    """BETTER / EQUAL / WORSE / INCONCLUSIVE with the 95% and 90% intervals and the one-sided p of 'cand better'. lower_better: NLL
    (d < 0 = better); False for accuracy (d > 0 = better, keep_d = ACC_D)."""
    d, se, df = e['d'], e['se'], e['df']
    h95, h90 = ES.tq(.95, df) * se, ES.tq(.90, df) * se
    g = d if lower_better else -d  # oriented: g < 0 = candidate better
    better, equal, worse = g <= -keep_d and g + h95 < 0, abs(d) + h90 < eq, g - h95 > 0
    p = float((sps.t if math.isfinite(df) else sps.norm).cdf(g / se, *([df] if math.isfinite(df) else []))) if se > 0 else float(g >= 0)
    return {**{k: round(v, 5) if isinstance(v, float) and math.isfinite(v) else v for k, v in e.items()},
            'ci95': [round(d - h95, 5), round(d + h95, 5)], 'ci90': [round(d - h90, 5), round(d + h90, 5)], 'p_better': round(p, 5),
            'verdict': 'BETTER' if better else 'EQUAL' if equal else 'WORSE' if worse else 'INCONCLUSIVE'}


def metric_parts(A_ref, A_cand, ns):
    """(D (S,), Z, ws) of the dev metric from per-suite record NLL sums A_* [(S, R_k)] and question counts ns [(R_k,)]."""
    D, Z = 0.0, []
    for ar, ac, n in zip(A_ref, A_cand, ns):
        delta = ac - ar
        dk = delta.sum(1) / n.sum()
        D = D + dk / len(ns)
        Z.append((delta - dk[:, None] * n[None]) / n.mean())
    return D, Z, [1 / len(ns)] * len(ns)


def _records(dumps, suite):
    """(question counts (R,), record NLL sums (F, R), correct counts (F, R), rids); every dump must hold the same questions."""
    rows0 = dumps[0][suite]
    rids = list(dict.fromkeys(r[0] for r in rows0))
    pos = {r: i for i, r in enumerate(rids)}
    ix = np.array([pos[r[0]] for r in rows0])
    nll, cor = [], []
    for dmp in dumps:
        rows = dmp[suite]
        assert len(rows) == len(rows0) and all(a[0] == b[0] and a[2] == b[2] for a, b in zip(rows, rows0)), f'{suite}: dumps differ'
        nll.append(np.bincount(ix, [-math.log(max(r[1][r[2]], 1e-15)) for r in rows], len(rids)))
        cor.append(np.bincount(ix, [float(max(range(len(r[1])), key=r[1].__getitem__) == r[2]) for r in rows], len(rids)))
    return np.bincount(ix, minlength=len(rids)).astype(float), np.array(nll), np.array(cor), rids


def _ratio_parts(n, xr, xc):
    """One suite's question-mean difference of per-record sums xc - xr (S, R) as an estimate() input (ratio linearisation)."""
    delta = xc - xr
    dk = delta.sum(1) / n.sum()
    return dk, [(delta - dk[:, None] * n[None]) / n.mean()], [1.0]


def wide_parts(ref, cand, grp, keep, suite):
    """Group-macro accuracy difference on `suite` over sources in `keep` (sources equal, groups equal within a source)."""
    _, _, cr, rids = _records(ref, suite)
    n, _, cc, _ = _records(cand, suite)
    g = [grp[r] for r in rids]
    D, Z, src = 0.0, [], sorted({x.split('/', 1)[0] for x in g if x.split('/', 1)[0] in keep})
    for s in src:
        gs = sorted({x for x in g if x.startswith(s + '/')})
        M = np.array([[1.0 if x == gg else 0.0 for x in g] for gg in gs])  # (G, R) record -> group
        acc = lambda c: (c @ M.T) / (M @ n)[None]  # noqa: E731
        delta = acc(cc) - acc(cr)
        D = D + delta.mean(1) / len(src)
        Z.append(delta)
    return D, Z, [1 / len(src)] * len(src), src


def per_arm(dumps, suites):
    """Seed mean and between-seed SD per arm of the metric and of each suite's NLL / accuracy."""
    out = {}
    for s in suites:
        n, nll, cor, _ = _records(dumps, s)
        for k, v in (('nll', nll.sum(1) / n.sum()), ('acc', cor.sum(1) / n.sum())):
            out[f'{s}_{k}'] = {'mean': round(float(v.mean()), 4), 'sd': round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                               'by_seed': [round(float(x), 4) for x in v]}
    m = np.mean([np.array(out[f'{s}_nll']['by_seed']) for s in suites], 0)
    out['metric'] = {'mean': round(float(m.mean()), 4), 'sd': round(float(m.std(ddof=1)), 4) if len(m) > 1 else None}
    return out


def score(ref, cand, suites=None, grp=None, groups_suite=None, real=None, n_cmp=1, prior_sd=SD_FIT):
    """The full report of one reference/candidate comparison (S_ref == S_cand; seed i paired with seed i). suites: the suites
    of the metric (default: every suite in the dumps except groups_suite); grp {record id: 'source/workflow'} enables the
    group-macro verdicts on groups_suite (real = the sources of macro_real)."""
    assert len(ref) == len(cand) >= 1, 'pair seeds: the same number of fits per arm'
    suites = list(suites or [s for s in ref[0] if s != groups_suite])
    parts = [_records(ref, s) for s in suites], [_records(cand, s) for s in suites]
    D, Z, ws = metric_parts([p[1] for p in parts[0]], [p[1] for p in parts[1]], [p[0] for p in parts[0]])
    shown = suites + ([groups_suite] if groups_suite in ref[0] else [])
    rep = {'suites_in_metric': suites, 'arms': {'ref': per_arm(ref, shown), 'cand': per_arm(cand, shown)},
           'metric': verdict(estimate(D, Z, ws, prior_sd)), 'suites': {}}
    for s, pr, pc in zip(suites, *parts):
        rep['suites'][s] = {'nll': verdict(estimate(*_ratio_parts(pr[0], pr[1], pc[1]))),
                            'acc': verdict(estimate(*_ratio_parts(pr[0], pr[2], pc[2])), ACC_D, ACC_D, lower_better=False)}
    if grp is not None and groups_suite in ref[0]:
        every = tuple(sorted({g.split('/', 1)[0] for g in grp.values()}))
        for name, keep in (('macro_real', tuple(real or every)), ('macro_all', every)):
            D, Z, ws, src = wide_parts(ref, cand, grp, keep, groups_suite)
            rep[f'{groups_suite}_{name}'] = {**verdict(estimate(D, Z, ws), ACC_D, ACC_D, lower_better=False), 'sources': src}
    e = rep['metric']
    mN = -float(sps.norm.ppf((n_cmp - .375) / (n_cmp + .25))) if n_cmp > 1 else 0.0  # Blom: E[min of N std normals]
    hb = ES.tq(1 - .05 / n_cmp, e['df']) * e['se']
    rep['selection'] = {'n_comparisons': n_cmp, 'p_better': e['p_better'], 'p_bonferroni': round(min(1.0, n_cmp * e['p_better']), 5),
                        'ci95_bonferroni': [round(e['d'] - hb, 5), round(e['d'] + hb, 5)], 'best_of_n_optimism': round(-mN * e['se'], 5)}
    return rep


def _sim_arm(rng, S, shift, sd_fit, sd_q, tdf=None):
    """Per-suite record NLL sums (S, R_k) of one arm: fit effect per (seed, suite) + per-question noise (t-distributed if tdf)."""
    out = []
    for R, Q in zip(SIM_R, SIM_Q):
        n = np.full(R, Q // R, float)
        n[:Q - int(n.sum())] += 1
        a = rng.normal(0, sd_fit * 2, (S, 1))  # per-suite fit effect; the 4-suite mean has SD sd_fit
        e = rng.standard_t(tdf, (S, R)) * math.sqrt((tdf - 2) / tdf) if tdf else rng.normal(size=(S, R))
        out.append((n, (0.7 + shift + a) * n + e * sd_q * np.sqrt(n)))
    return out


def simulate(S, delta, reps=2000, seed=0, sd_d=SD_FIT, se_rec=0.0030, tdf=None):
    """Rates of BETTER / EQUAL / WORSE and the 95% CI coverage of delta, NULL-calibrated to ej's measured fit noise: two fits of the same code
    differ by SD sd_d on the metric, of which se_rec is record-level (fit-specific) noise."""
    rng = np.random.default_rng(seed)
    sd_q = se_rec * 4 / math.sqrt(2 * sum(1 / q for q in SIM_Q))
    sd_fit = math.sqrt(max(sd_d ** 2 - se_rec ** 2, 0) / 2)
    cnt = {'BETTER': 0, 'EQUAL': 0, 'WORSE': 0, 'INCONCLUSIVE': 0, 'cover': 0}
    for _ in range(reps):
        r, c = _sim_arm(rng, S, 0.0, sd_fit, sd_q, tdf), _sim_arm(rng, S, delta, sd_fit, sd_q, tdf)
        v = verdict(estimate(*metric_parts([x[1] for x in r], [x[1] for x in c], [x[0] for x in r]), SD_FIT))
        cnt[v['verdict']] += 1
        cnt['cover'] += v['ci95'][0] <= delta <= v['ci95'][1]
    return {k: round(v / reps, 4) for k, v in cnt.items()}
