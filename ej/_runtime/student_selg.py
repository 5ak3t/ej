"""Group-honest selective head: one correctness head per regime (seen option slots x structured state), with unseen-slot heads
trained leave-one-group-out. Off in the released configuration."""
import torch

import student_gcv as GCV
import student_pool as P
from student_sel import _fit_lr, _mix_rows, _nll, _raw, apply, features

def _head(p, X, y, fold=None):
    """Standardised L2 logistic correctness model; outer CV NLL vs the pool (2 row folds, or leave-one-GROUP-out when `fold` gives
    group ids); returns (w on raw features or None, report)."""
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Xs = (X - mu) / sd
    corr = (p.argmax(-1) == y).float()
    fold = torch.arange(len(y)) % 2 if fold is None else fold
    cv = []
    for f in sorted(set(fold.tolist())):
        w = _fit_lr(Xs[fold != f], corr[fold != f], X[fold != f, 0])
        cv.append((apply(p[fold == f], X[fold == f], _raw(w, mu, sd)), y[fold == f]))
    nll_head = sum(_nll(q, t) * len(t) for q, t in cv) / len(y)
    nll_pool = _nll(p.double(), y)
    w_raw = _raw(_fit_lr(Xs, corr, X[:, 0]), mu, sd)
    return (w_raw if nll_head < nll_pool else None), {'rows': len(y), 'folds': len(cv), 'nll_pool_cf': round(nll_pool, 4),
                                                      'nll_head_cv': round(nll_head, 4)}


def _splits(k, rows, cfg, n_exp):
    """Cross-fitted pool rows of one key: seen keys 2 row folds (iid); unseen keys leave-one-GROUP-out with the key's group-cross-
    fitted pool configuration (student_gcv), so the head learns how the pool behaves on a task it was not calibrated on. Unseen keys
    whose pool is uniform (no expert with out-of-group support) give the head nothing to rank and are left out."""
    if k[1]:
        return [([r for n, r in enumerate(rows) if n % 2 != f], [r for n, r in enumerate(rows) if n % 2 == f], f, P.fit({k: [
            r for n, r in enumerate(rows) if n % 2 != f]})[k]) for f in (0, 1)]
    gs = sorted({r[2] for r in rows})
    if cfg is None or len(gs) < 2:
        return []
    out = []
    for h in gs:
        tr = [r for r in rows if r[2] != h]
        Z, y, _ = GCV.tens(tr)
        out.append((tr, [r for r in rows if r[2] == h], h, GCV.fit_sub(Z, y, *cfg)))
    return out


def fit(groups, key_meta, n_exp, cfgs):
    """groups: key -> [(Z (E, K), y, group)]; key_meta(key) -> (type, seen, json); cfgs: unseen key -> group-cross-fitted pool
    config (student_gcv). One head PER REGIME (seen option slots x structured state; the plain-text / structured split gives
    tickets and Typed Decisions their own ranking), each adopted only if it beats the pool in its own cross-validation (seen:
    2 row folds; unseen: leave-one-group-out). Returns ({(seen, json): weights or None}, report)."""
    rows_by = {}
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(rows) < 2 * P.MIN_CAL:
            continue
        t, s, j = key_meta(k)
        phis, ps, ys, fs = rows_by.setdefault((bool(s), bool(j)), ([], [], [], []))
        for _, ev, f, prm in _splits(k, rows, cfgs.get(k), n_exp):
            Z, p = _mix_rows(ev, prm, n_exp)
            n = len(ev)
            phis.append(features(Z, p, torch.full((n,), t), torch.full((n,), bool(s)), torch.full((n,), bool(j))))
            ps.append(p); ys.append(torch.tensor([r[1] for r in ev])); fs.extend([str(f)] * n)
    out, rep = {}, {}
    for rg, (phis, ps, ys, fs) in sorted(rows_by.items()):
        if not ps:
            out[rg], rep[str(rg)] = None, {}
            continue
        K = max(p.shape[1] for p in ps)
        p = torch.cat([torch.nn.functional.pad(x, (0, K - x.shape[1])) for x in ps])
        ids = {f: n for n, f in enumerate(sorted(set(fs)))}
        fold = None if rg[0] else torch.tensor([ids[f] for f in fs])
        out[rg], rep[str(rg)] = _head(p, torch.cat(phis), torch.cat(ys), fold)
    return out, rep


def apply_rows(p, phi, heads, seen, js):
    """Per-row head by regime (seen option slots, structured state); rows whose pooled distribution is uniform are left alone."""
    out = p.double().clone()
    M = p > 0
    flat = (p.masked_fill(~M, 0).amax(-1) - p.masked_fill(~M, 2).amin(-1)) < 1e-9
    for (s, j), w in heads.items():
        m = (seen == s) & (js == j) & ~flat
        if w is not None and m.any():
            out[m] = apply(p[m], phi[m], w)
    return out
