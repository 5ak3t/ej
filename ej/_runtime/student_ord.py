"""Ordinal-aware score expert: reads score questions through their level texts on a unimodal ordinal scale. Built but off in
the released configuration."""
import sys

import numpy as np
import torch

COLS = (0, 2, 4, 6, 9, 11)  # raw similarity scalars of student_deep.scalars (cos, hyp, lex word, lex char, late opt, late hyp)
L2 = 1e-2
SCORE = 2
NF = 2 * len(COLS)


def features(S, M, T):
    """(N, NF) ordinal location features (zeros on non-score rows) and the (N,) mask of usable score rows."""
    m = M.float()
    k = m.sum(1, keepdim=True)
    j = torch.arange(M.shape[1]).float()[None, :]
    pos = (j / (k - 1).clamp(min=1) - 0.5) * m
    ok = (T == SCORE) & (k[:, 0] >= 3)
    out = []
    for c in COLS:
        x = S[..., c] * m
        xc = (x - x.sum(1, keepdim=True) / k) * m
        cov = (xc * pos).sum(1)
        out += [cov / (pos * pos).sum(1).clamp(min=1e-9), cov / ((xc * xc).sum(1).sqrt() * (pos * pos).sum(1).sqrt() + 1e-6)]
    return torch.stack(out, 1) * ok[:, None].float(), ok, pos


def _logits(th, f, pos, M):
    eta = th['b'] + ((f - th['mu']) / th['sd']) @ th['w']
    z = -torch.nn.functional.softplus(th['rho']) * pos * pos + pos * eta[:, None]
    return z


def logits(th, data):
    """(N, K) logits; 0 on non-score rows (silent expert), -1e9 on padded slots."""
    M, S, T = data[2], data[3], data[5]
    f, ok, pos = features(S, M, T)
    z = torch.zeros(M.shape)
    if th is not None and ok.any():
        z[ok] = _logits(th, f[ok], pos[ok], M[ok])
    return z.masked_fill(~M, -1e9)


def train(data, y):
    """Fit on the usable score rows of `data` (None when there are none)."""
    M, S, T = data[2], data[3], data[5]
    f, ok, pos = features(S, M, T)
    if ok.sum() < 10:
        return None
    f, pos, Mo, yo = f[ok], pos[ok], M[ok], y[ok]
    mu, sd = f.mean(0), f.std(0).clamp(min=1e-6)
    th = {'w': torch.zeros(NF), 'b': torch.zeros(()), 'rho': torch.tensor(0.5413), 'mu': mu, 'sd': sd}
    par = [th[k].requires_grad_() for k in ('w', 'b', 'rho')]
    opt = torch.optim.LBFGS(par, max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = _logits(th, f, pos, Mo).masked_fill(~Mo, -1e9)
        loss = torch.nn.functional.cross_entropy(z, yo) + L2 * ((th['w'] ** 2).sum() + th['b'] ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    return {k: v.detach().float() for k, v in th.items()}


def _sub(data, sel):
    return tuple(t[sel] for t in data)


def _rep(z, y, M, T):
    ok = (T == SCORE) & (M.sum(1) >= 3)
    if not ok.any():
        return None
    zz, yy = z[ok], y[ok]
    return {'nll': round(torch.nn.functional.cross_entropy(zz, yy).item(), 4), 'acc': round((zz.argmax(1) == yy).float().mean().item(), 3),
            'unif': round(M[ok].sum(1).float().log().mean().item(), 4), 'n': int(ok.sum())}


def fit(data, y, hold, gid):
    """{'ho': held-out logits (fit on the rest), 'logo': {g: logits on g, fitted without g}, 'full': theta, 'rep': diagnostics}."""
    tr, ho = torch.tensor(~np.asarray(hold)), torch.tensor(np.asarray(hold))
    out = {'ho': logits(train(_sub(data, tr), y[tr]), _sub(data, ho)), 'logo': {}, 'rep': {}}
    gid = np.asarray(gid)
    for g in sorted(set(gid)):
        o = torch.tensor(gid == g)
        z = logits(train(_sub(data, ~o), y[~o]), _sub(data, o))
        out['logo'][g] = z
        r = _rep(z, y[o], data[2][o], data[5][o])
        if r:
            out['rep'][g] = r
            print('ord logo', g, r, file=sys.stderr, flush=True)
    out['full'] = train(data, y)
    if out['full'] is not None:
        out['rep']['full'] = {'w': [round(v, 3) for v in out['full']['w'].tolist()], 'b': round(out['full']['b'].item(), 3),
                              'kappa': round(torch.nn.functional.softplus(out['full']['rho']).item(), 3)}
    print('ord', out['rep'].get('full'), file=sys.stderr, flush=True)
    return out


def size_mb(state):
    th = state.get('ord')
    return sum(v.numel() for v in th.values()) * 2 / 2 ** 20 if th else 0.0


def _test():
    """Silent on non-score rows, unimodal on score rows, recovers a text-driven location, LOGO shapes."""
    torch.manual_seed(0)
    N, K = 400, 4
    M = torch.ones(N, K, dtype=torch.bool); M[:50, 3] = False
    T = torch.full((N,), SCORE); T[:40] = 0
    y = torch.randint(0, 3, (N,))
    S = torch.randn(N, K, 13) * 0.1
    for n in range(N):  # similarity profile peaks at the gold level
        S[n, y[n], 0] += 1.0
    data = (torch.zeros(N, 1), torch.zeros(N, 1), M, S, torch.zeros(N, 1), T)
    th = train(data, y)
    z = logits(th, data)
    assert (z[:40][M[:40]] == 0).all() and (z[:50, 3] == -1e9).all()
    p = torch.softmax(z[40:], -1)
    d = torch.sign(p[:, 1:] - p[:, :-1])
    assert all((torch.diff(r[r != 0]) <= 0).all() for r in d)  # up then down: unimodal
    acc = (z[40:].argmax(1) == y[40:]).float().mean().item()
    assert acc > 0.5, acc
    gid = ['a'] * 200 + ['b'] * 200
    hold = np.arange(N) % 10 == 0
    out = fit(data, y, hold, gid)
    assert set(out['logo']) == {'a', 'b'} and out['ho'].shape == (int(hold.sum()), K) and size_mb({'ord': out['full']}) > 0
    assert logits(None, data).abs().max() == 1e9 and (logits(None, data)[M] == 0).all()
    print('student_ord ok', acc)


if __name__ == '__main__':
    _test()
