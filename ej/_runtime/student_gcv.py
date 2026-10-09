"""Group cross-fitted calibration pool for unseen option slots: pool coefficients fitted with a second-level cross-fit over
groups, so an expert is trusted on a new task only as far as it transferred to held-out groups. Off in the released
configuration."""
import torch

import student_pool as P

LAMS = (1e-3, 1e-2, 1e-1, 1.0)  # a priori ridge path on the pool coefficients (1e-3 = the round-1..3 fit)


def tens(rows):
    K = max(r[0].shape[1] for r in rows)
    Z = torch.full((rows[0][0].shape[0], len(rows), K), -1e9)
    for n, r in enumerate(rows):
        Z[:, n, :r[0].shape[1]] = r[0]
    return Z, torch.tensor([r[1] for r in rows]), [r[2] for r in rows]


def uniform(E):
    return (0.0,) * E + (0.0,)


def fit_sub(Z, y, S, lam):
    """Pool restricted to experts S (others a = 0); returns the full parameter tuple."""
    E = Z.shape[0]
    if not S:
        return uniform(E)
    prm = P.fit_one(Z[list(S)], y, lam)
    a = [0.0] * E
    for e, v in zip(S, prm[:-1]):
        a[e] = v
    return tuple(a) + (prm[-1],)


def nll(Z, y, prm):
    E = Z.shape[0]
    return P._nll(Z, y, torch.tensor(prm[:E]), prm[E]).item()


def _live(Z):
    M = Z[0] > -1e8
    return [e for e in range(Z.shape[0]) if Z[e].masked_fill(~M, 0).abs().sum() > 0]


def fit_key(rows):
    """Returns (params, cfg, report); cfg = (experts, lam) or None for uniform (used by the selective head's group cross-fit)."""
    Z, y, g = tens(rows)
    E, gs = Z.shape[0], sorted(set(g))
    if len(gs) < 2:
        return uniform(E), None, {'groups': len(gs), 'choice': 'uniform (no out-of-group evidence)'}
    masks = [torch.tensor([x == h for x in g]) for h in gs]
    cache = {}

    def cv(S, lam):
        k = (tuple(S), lam)
        if k not in cache:
            cache[k] = sum(nll(Z[:, m], y[m], fit_sub(Z[:, ~m], y[~m], S, lam)) for m in masks) / len(masks)
        return cache[k]
    u = cv((), None)
    sup = {e: cv((e,), LAMS[0]) for e in _live(Z)}
    S = tuple(e for e, v in sup.items() if v < u)
    rep = {'groups': len(gs), 'cv_uniform': round(u, 4), 'support': {e: round(v, 4) for e, v in sup.items()}}
    if not S:
        return uniform(E), None, {**rep, 'choice': 'uniform'}
    path = {lam: cv(S, lam) for lam in LAMS}
    lam = min(LAMS, key=path.get)
    rep.update(path={str(k): round(v, 4) for k, v in path.items()}, experts=list(S), lam=lam)
    if path[lam] >= u:
        return uniform(E), None, {**rep, 'choice': 'uniform'}
    return fit_sub(Z, y, S, lam), (S, lam), {**rep, 'choice': 'pool'}


def fit(groups):
    """groups: key -> [((E, K) logits, label, group)]. Seen keys: iid held-out fit (student_pool); unseen keys: group cross-fit.
    Coarse fallback keys (type, seen) as in student_pool. Returns (pool params, unseen cfgs, report)."""
    coarse = {}
    for k, rows in groups.items():
        coarse.setdefault(k[:2], []).extend(rows)
    out, cfg, rep = {}, {}, {}
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])) + sorted(coarse.items(), key=lambda kv: str(kv[0])):
        if len(rows) < P.MIN_CAL:
            continue
        if k[1]:
            Z, y, _ = tens(rows)
            out[k] = P.fit_one(Z, y)
        else:
            out[k], cfg[k], rep[str(k)] = fit_key(rows)
    return out, cfg, rep
