"""Selective correctness head: a small logistic model that re-scores the top label's probability from pool and expert
features, improving the ordering of confidences used for certified automation; adopted per regime only when it beats the
pool."""
import torch

import student_pool as P

L2 = 1e-3  # a priori: weak ridge, ~20 coefficients on thousands of rows


def _mix_rows(rows, prm, n_exp):
    K = max(r[0].shape[1] for r in rows)
    Z = torch.full((n_exp, len(rows), K), -1e9)
    for n, r in enumerate(rows):
        Z[:, n, :r[0].shape[1]] = r[0]
    par = torch.tensor([prm] * len(rows))
    return Z, P.mix(Z, par[:, :n_exp], par[:, n_exp:])


def features(Z, p, T, seen, js):
    """Z: (E, n, K) expert logits (-1e9 pad); p: (n, K) pooled probs; T: (n,) type; seen, js: (n,) bool."""
    M = Z[0] > -1e8
    K = M.sum(-1).float()
    top2 = p.topk(2, -1)
    a = top2.indices[:, 0]
    pe = torch.softmax(Z.masked_fill(~M, -1e9), -1)  # (E, n, K)
    sup = pe[:, torch.arange(p.shape[0]), a].T  # (n, E)
    live = (Z.masked_fill(~M, 0).abs().sum(-1) > 0).T.float()  # experts that are not silent (field attention on unseen slots)
    agree = ((pe.argmax(-1).T == a[:, None]).float() * live).sum(1) / live.sum(1).clamp(min=1)
    tv = 0.5 * (pe[:, None] - pe[None]).abs().sum(-1)  # (E, E, n)
    lv = live.T
    w = lv[:, None] * lv[None]
    dis = (tv * w).sum((0, 1)) / w.sum((0, 1)).clamp(min=1)
    ent = -(p.clamp(min=1e-12).log() * p).sum(-1) / K.log()
    oh = torch.nn.functional.one_hot(T, 3).float()
    pm = top2.values[:, :1].double().clamp(1e-9, 1 - 1e-9)
    return torch.cat([(pm.log() - (1 - pm).log()).float(), (top2.values[:, 0] - top2.values[:, 1])[:, None], ent[:, None],
                      K.log()[:, None], sup * live, agree[:, None], dis[:, None], oh, seen.float()[:, None], js.float()[:, None]], 1)


def apply(p, phi, w):
    """Reshape the pooled distribution so its top label gets the correctness probability c."""
    if w is None:
        return p
    c = torch.sigmoid(phi[:, 0].double() + (phi @ w[:-1] + w[-1]).double())  # residual on the pool's own top-label logit
    p = p.double()
    top = p.argmax(-1)
    pm = p.gather(1, top[:, None])
    rest = (1 - c[:, None]) * p / (1 - pm).clamp(min=1e-12)
    return rest.scatter(1, top[:, None], c[:, None])


def _fit_lr(X, y, off, l2=L2):
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = off + X @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def _nll(p, y):
    return -p[torch.arange(len(y)), y].clamp(min=1e-15).log().mean().item()


def _raw(w, mu, sd):
    """Fold the standardisation into the weights (the head is applied to raw features)."""
    return torch.cat([w[:-1] / sd, (w[-1] - (w[:-1] * mu / sd).sum())[None]])


def _head(p, X, y):
    """Standardised L2 logistic correctness model; outer 2-fold CV NLL vs the pool; returns (w on raw features or None, report)."""
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Xs = (X - mu) / sd
    corr = (p.argmax(-1) == y).float()
    fold = torch.arange(len(y)) % 2
    cv = []
    for f in (0, 1):
        w = _fit_lr(Xs[fold != f], corr[fold != f], X[fold != f, 0])
        cv.append((apply(p[fold == f], X[fold == f], _raw(w, mu, sd)), y[fold == f]))
    nll_head = sum(_nll(q, t) * len(t) for q, t in cv) / len(y)
    nll_pool = _nll(p.double(), y)
    w_raw = _raw(_fit_lr(Xs, corr, X[:, 0]), mu, sd)
    return (w_raw if nll_head < nll_pool else None), {'rows': len(y), 'nll_pool_cf': round(nll_pool, 4), 'nll_head_cv': round(nll_head, 4)}


def fit(groups, key_meta, n_exp):
    """groups: key -> [(Z (E, K), y)]; key_meta(key) -> (type, seen, json). One head PER REGIME (seen option slots: iid held-out
    rows; unseen: zero-shot cross-fitted rows), each adopted only if it beats the pool in its own cross-validation.
    Returns ({seen: weights or None}, report)."""
    rows_by = {True: ([], [], []), False: ([], [], [])}
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(rows) < 2 * P.MIN_CAL:
            continue
        t, s, j = key_meta(k)
        phis, ps, ys = rows_by[bool(s)]
        for f in (0, 1):  # 2-fold cross-fit of the pool on this key's rows
            fit_rows = [r for n, r in enumerate(rows) if n % 2 != f]
            ev = [r for n, r in enumerate(rows) if n % 2 == f]
            prm = P.fit({k: fit_rows})[k]
            Z, p = _mix_rows(ev, prm, n_exp)
            n = len(ev)
            phis.append(features(Z, p, torch.full((n,), t), torch.full((n,), bool(s)), torch.full((n,), bool(j))))
            ps.append(p); ys.append(torch.tensor([r[1] for r in ev]))
    K = max(p.shape[1] for v in rows_by.values() for p in v[1])
    cat = lambda v: (torch.cat([torch.nn.functional.pad(x, (0, K - x.shape[1])) for x in v[1]]), torch.cat(v[0]), torch.cat(v[2]))
    out, rep = {}, {}
    for s, v in rows_by.items():
        out[s], rep['seen' if s else 'unseen'] = _head(*cat(v)) if v[0] else (None, {})
    p, X, y = [torch.cat(t) for t in zip(cat(rows_by[True]), cat(rows_by[False]))]
    rep['global(diag)'] = _head(p, X, y)[1]
    return out, rep


def apply_rows(p, phi, heads, seen):
    """Per-row head by regime (seen option slots or not)."""
    out = p.double().clone()
    for s, w in heads.items():
        m = seen == s
        if w is not None and m.any():
            out[m] = apply(p[m], phi[m], w)
    return out
