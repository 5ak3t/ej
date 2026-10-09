"""Calibrated log-linear pool of expert logits with a lapse to uniform, per calibration key: p = (1 - eps) softmax(sum_e a_e
z_e) + eps / K with non-negative coefficients; fitting and mixing functions."""
import torch

EPS_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
MIN_CAL = 30
RIDGE = 1e-3  # keeps the coefficients finite on separable held-out keys
INIT = 0.5413  # softplus^-1(1): the first expert starts at a = 1, the others at ~0


def mix(Z, A, eps):
    """Z: (E, n, K) expert logits (-1e9 padded); A: (n, E) coefficients; eps: (n, 1)."""
    M = Z[0] > -1e8
    K = M.sum(-1, keepdim=True).float()
    z = (A.T[..., None] * Z.masked_fill(~M, 0)).sum(0).masked_fill(~M, -1e9)
    return ((1 - eps) * torch.softmax(z, -1) + eps / K) * M


def _nll(Z, y, a, eps):
    A = a.expand(Z.shape[1], -1)
    return -torch.log(mix(Z, A, torch.full((Z.shape[1], 1), float(eps)))[torch.arange(len(y)), y].clamp(min=1e-15)).mean()


def fit_one(Z, y, ridge=RIDGE):
    best = None
    for eps in EPS_GRID:
        th = torch.full((Z.shape[0],), -5.0); th[0] = INIT; th.requires_grad_()
        opt = torch.optim.LBFGS([th], max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            a = torch.nn.functional.softplus(th)
            loss = _nll(Z, y, a, eps) + ridge * (a * a).sum()
            loss.backward()
            return loss
        opt.step(closure)
        a = torch.nn.functional.softplus(th).detach()
        v = _nll(Z, y, a, eps).item()
        if best is None or v < best[0]:
            best = (v, tuple(round(x, 4) for x in a.tolist()) + (eps,))
    return best[1]


def fit(groups):
    """groups: key -> list of ((E, K) expert logit rows, label); coarse fallback keys (type, seen)."""
    out, coarse = {}, {}
    for k, rows in groups.items():
        coarse.setdefault(k[:2], []).extend(rows)
    for k, rows in list(groups.items()) + list(coarse.items()):
        if len(rows) >= MIN_CAL:
            K = max(r[0].shape[1] for r in rows)
            Z = torch.full((rows[0][0].shape[0], len(rows), K), -1e9)
            for n, r in enumerate(rows):
                Z[:, n, :r[0].shape[1]] = r[0]
            out[k] = fit_one(Z, torch.tensor([r[1] for r in rows]))
    return out


def params_for(pool, key, n_exp):
    return pool.get(key, pool.get(key[:2], (1.0,) + (0.0,) * (n_exp - 1) + (0.0,)))
