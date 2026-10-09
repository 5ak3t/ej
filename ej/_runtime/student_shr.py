"""Conservative selective head under novelty: the correctness head with a transfer coefficient per regime cell, estimated on
held-out groups and shrunk hierarchically toward the identity. Off in the released configuration."""
import torch

import student_pool as P
from student_sel import _fit_lr, _mix_rows, features

M_PSEUDO = 1.0  # a priori: the parent curve counts as one pseudo-unit
GRID = tuple(i / 20 for i in range(21))
TYPE_COLS = slice(-5, -2)  # one-hot type columns of student_sel.features (followed by seen, json)


def _rows(groups, n_exp):
    """Cross-fitted pool rows exactly as student_sel.fit (2 row folds per key), keeping each row's unit and cell."""
    out = {True: [], False: []}
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(rows) < 2 * P.MIN_CAL:
            continue
        t, s, j = k
        for f in (0, 1):
            ev = [r for n, r in enumerate(rows) if n % 2 == f]
            prm = P.fit({k: [r for n, r in enumerate(rows) if n % 2 != f]})[k]
            Z, p = _mix_rows(ev, prm, n_exp)
            n = len(ev)
            phi = features(Z, p, torch.full((n,), t), torch.full((n,), bool(s)), torch.full((n,), bool(j)))
            unit = [f'fold{f}' if s else str(r[2]) for r in ev]
            out[bool(s)].append((phi, p, torch.tensor([r[1] for r in ev]), unit, t, bool(j)))
    return out


def _cat(parts):
    K = max(x[1].shape[1] for x in parts)
    pad = lambda x: torch.nn.functional.pad(x, (0, K - x.shape[1]))
    unit = [u for x in parts for u in x[3]]
    cell = [(x[4], x[5]) for x in parts for _ in x[3]]
    return (torch.cat([x[0] for x in parts]), torch.cat([pad(x[1]) for x in parts]).double(), torch.cat([x[2] for x in parts]),
            unit, cell)


def _head(X, corr):
    """Standardised L2 logistic correctness model (as student_sel), returned on raw features."""
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    w = _fit_lr((X - mu) / sd, corr, X[:, 0])
    return torch.cat([w[:-1] / sd, (w[-1] - (w[:-1] * mu / sd).sum())[None]])


def _delta(X, w):
    return (X @ w[:-1] + w[-1]).double()


def _curve(l, d, corr, m):
    """Mean binary log-loss of correctness over rows m for every lam in GRID (NLL of p' up to a lam-free constant)."""
    z = l[m][None] + torch.tensor(GRID, dtype=torch.float64)[:, None] * d[m][None]
    return torch.nn.functional.binary_cross_entropy_with_logits(z, corr[m].double()[None].expand_as(z), reduction='none').mean(1)


def _lams(X, p, y, unit, cell):
    """Held-out-unit corrections, then hierarchical group-equal lam: regime -> type -> (type, json). Returns (lams, w_full, rep)."""
    corr = (p.argmax(-1) == y).float()
    l = X[:, 0].double()
    units = sorted(set(unit))
    um = {u: torch.tensor([x == u for x in unit]) for u in units}
    d = torch.zeros(len(y), dtype=torch.float64)
    for u in units:
        d[um[u]] = _delta(X[um[u]], _head(X[~um[u]], corr[~um[u]]))
    curves = lambda sel: [_curve(l, d, corr, um[u] & sel) for u in units if (um[u] & sel).any()]
    every = torch.ones(len(y), dtype=torch.bool)
    root = torch.stack(curves(every)).mean(0)
    lams, rep = {}, {'units': len(units), 'root': GRID[int(root.argmin())]}
    for t in sorted({c[0] for c in cell}):
        mt = torch.tensor([c[0] == t for c in cell])
        ct = curves(mt)
        ft = (torch.stack(ct).sum(0) + M_PSEUDO * root) / (len(ct) + M_PSEUDO)
        for j in sorted({c[1] for c in cell if c[0] == t}):
            mj = mt & torch.tensor([c[1] == j for c in cell])
            cj = curves(mj)
            fj = (torch.stack(cj).sum(0) + M_PSEUDO * ft) / (len(cj) + M_PSEUDO)
            lams[(t, j)] = GRID[int(fj.argmin())]
            rep[f'{t}{"j" if j else "p"}'] = (lams[(t, j)], len(cj), round(fj[GRID.index(1.0)].item() - fj[0].item(), 4))
    return lams, _head(X, corr), rep


def fit(groups, n_exp):
    """groups: key (type, seen, json) -> [(Z (E, K), y, group)]. Returns ({'w': {seen: w}, 'lam': {(seen, t, j): lam}}, report)."""
    out, rep = {'w': {}, 'lam': {}}, {}
    for s, parts in _rows(groups, n_exp).items():
        if not parts:
            continue
        X, p, y, unit, cell = _cat(parts)
        lams, w, rep['seen' if s else 'unseen'] = _lams(X, p, y, unit, cell)
        out['w'][s] = w
        out['lam'].update({(s,) + c: v for c, v in lams.items()})
    return out, rep


def apply_rows(p, phi, head, seen, js):
    """c = sigmoid(logit p_max + lam_cell * delta) on the top label; the rest keeps its order and shares 1 - c; lam = 0 -> identity."""
    T = phi[:, TYPE_COLS].argmax(1)
    lam = torch.tensor([head['lam'].get((bool(s), int(t), bool(j)), 0.0) for s, t, j in zip(seen.tolist(), T.tolist(), js.tolist())],
                       dtype=torch.float64)
    out = p.double().clone()
    for s, w in head['w'].items():
        m = (seen == s) & (lam > 0)
        if m.any():
            c = torch.sigmoid(phi[m, 0].double() + lam[m] * _delta(phi[m], w))
            q = p[m].double()
            top = q.argmax(-1)
            pm = q.gather(1, top[:, None])
            out[m] = ((1 - c[:, None]) * q / (1 - pm).clamp(min=1e-12)).scatter(1, top[:, None], c[:, None])
    return out
