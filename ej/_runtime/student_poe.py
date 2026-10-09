"""Option-text debiasing: a bias-only expert that sees the question and options but not the state, and a main expert trained
as a product of experts with the frozen bias logits as an offset; the negated bias expert enters the pool with its own
weight. device() returns both experts' logits at prediction time."""
import sys

import numpy as np
import torch

import student_deep as D

MODES = ('poe', 'conf')  # + '+neg': the bias expert also enters NEGATED (-z_b) as one more pool expert, so the pool's
# non-negative coefficients give the bias a SIGNED weight per cell: p ∝ prod_e p_e^a_e / p_b^c divides the option-text prior out of
# the WHOLE pool (every expert carries it, not only the deep one), with c chosen on the group-honest LOGO rows like every weight.


def n_extra(mode):
    """Extra pool experts after the round-1 slots: the bias expert, plus its negation in '+neg' modes."""
    return 0 if not mode else 2 if mode.endswith('+neg') else 1


def extras(zb, neg):
    """[z_b] or [z_b, -z_b] (padding kept at -1e9) for one row or a batch."""
    return [zb] + ([(-zb).masked_fill(zb < -1e8, -1e9)] if neg else [])


def bias_data(data, Ho):
    """State-free view (U_b, V, M, S_b, pos, T) of the question tensors (see the module docstring); Ho: (N, K, d) instr+option."""
    U, V, M, S, pos, T = data
    d = V.shape[-1]
    q = U[:, -d:]
    Ub = torch.cat([torch.zeros_like(q), torch.full_like(q, d ** -0.5), q], -1)
    z2 = torch.zeros(*M.shape, 2)
    Sb, _ = D.scalars(torch.nn.functional.normalize(q, dim=-1), V, q, Ho, M, z2, z2, T)
    return Ub, V, M, Sb, pos, T


def _sub(data, sel):
    return tuple(t[sel] for t in data)


def train(data, y, off=None, soft=None):
    """student_deep's conditional logit (L2, L-BFGS, convex) with an optional fixed logit offset (product of experts) or soft
    targets (confidence regularisation); stored fp16 like student_deep."""
    M = data[2]
    off = torch.zeros(M.shape) if off is None else off.masked_fill(~M, 0)
    theta = {k: v.requires_grad_() for k, v in D.init(data[0].shape[1]).items()}
    opt = torch.optim.LBFGS(list(theta.values()), max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = D.logits(theta, data) + off
        ce = (torch.nn.functional.cross_entropy(z, y) if soft is None
              else -(soft * torch.log_softmax(z, -1).masked_fill(~M, 0)).sum(-1).mean())
        loss = ce + D.L2 * sum((p * p).sum() for p in theta.values())
        loss.backward()
        return loss
    opt.step(closure)
    return {k: v.detach().half().float() for k, v in theta.items()}


def scaled_targets(zt, zb, y, M):
    """Utama et al. 2020: teacher distribution raised to (1 - bias probability of the gold label), renormalised."""
    pt = torch.softmax(zt.masked_fill(~M, -1e9), -1)
    beta = torch.softmax(zb.masked_fill(~M, -1e9), -1)[torch.arange(len(y)), y][:, None]
    s = pt.clamp(min=1e-12) ** (1 - beta) * M
    return s / s.sum(-1, keepdim=True)


def fit_pair(data, data_b, y, mode):
    """(main theta, bias theta) fitted on the given rows."""
    tb = D.train(data_b, y)
    zb = D.logits(tb, data_b)
    if mode == 'poe':
        return train(data, y, off=zb), tb
    zt = D.logits(D.train(data, y), data)
    return train(data, y, soft=scaled_targets(zt, zb, y, data[2])), tb


def _nll(z, y):
    return round(torch.nn.functional.cross_entropy(z, y).item(), 4)


def fit(data, Ho, y, hold, gid, mode='poe'):
    """{'ho': (main, bias) logits on the held-out slice (fitted on the rest), 'logo': {g: (main, bias) logits on g, fitted without g},
    'full': {'main', 'bias'} thetas on the whole pool, 'rep': LOGO diagnostics}."""
    neg, mode = mode.endswith('+neg'), mode.split('+')[0]
    assert mode in MODES
    db = bias_data(data, Ho)
    tr, ho = torch.tensor(~np.asarray(hold)), torch.tensor(np.asarray(hold))
    tm, tb = fit_pair(_sub(data, tr), _sub(db, tr), y[tr], mode)
    out = {'ho': (D.logits(tm, _sub(data, ho)), D.logits(tb, _sub(db, ho))), 'logo': {}, 'rep': {}}
    gid = np.asarray(gid)
    for g in sorted(set(gid)):
        o = torch.tensor(gid == g)
        tm, tb = fit_pair(_sub(data, ~o), _sub(db, ~o), y[~o], mode)
        zm, zb = D.logits(tm, _sub(data, o)), D.logits(tb, _sub(db, o))
        out['logo'][g] = (zm, zb)
        M, T, yo = data[2][o], data[5][o], y[o]
        rep = {'bias': _nll(zb, yo), 'main': _nll(zm, yo), 'prod': _nll(zm + zb, yo),
               'unif': round(M.sum(1).float().log().mean().item(), 4), 'n': int(o.sum())}
        for t in range(3):  # per question type (score = 2 is where the tilt sits)
            s = T == t
            if s.any():
                rep[f't{t}'] = (_nll(zb[s], yo[s]), _nll(zm[s], yo[s]), round(M[s].sum(1).float().log().mean().item(), 4),
                                round((zm[s].argmax(1) == yo[s]).float().mean().item(), 3), int(s.sum()))
        out['rep'][g] = rep
        print('poe logo', g, rep, file=sys.stderr, flush=True)
    tm, tb = fit_pair(data, db, y, mode)
    out['full'] = {'main': tm, 'bias': tb, 'mode': mode, 'neg': neg}
    return out


def device(st, data, Ho, z_slot):
    """predict side: (main logits for slot 7, [bias logits] as the extra expert); without a fitted 'poe' state: (z_slot, [])."""
    if not st:
        return z_slot, []
    return D.logits(st['main'], data), extras(D.logits(st['bias'], bias_data(data, Ho)), st.get('neg', False))


def size_mb(state):
    """Two convex heads at fp16 (3 d-blocks + scalar weights each)."""
    st = state.get('poe')
    return (D.n_params(st['main']) + D.n_params(st['bias'])) * 2 / 2 ** 20 if st else 0.0


def _test():
    """Shapes, offset training lowers the joint NLL, scaled targets normalise, device passthrough without a state."""
    torch.manual_seed(0)
    N, K, d = 60, 4, 8
    U = torch.nn.functional.normalize(torch.randn(N, 3 * d), dim=-1)
    V = torch.nn.functional.normalize(torch.randn(N, K, d), dim=-1)
    M = torch.ones(N, K, dtype=torch.bool); M[:10, 3] = False
    T = torch.randint(0, 3, (N,))
    S, pos = D.scalars(U[:, :d], V, U[:, :d], V, M, torch.zeros(N, K, 2), torch.zeros(N, K, 2), T)
    data = (U, V, M, S, pos, T)
    y = torch.randint(0, 3, (N,))
    db = bias_data(data, V)
    assert db[0].shape == U.shape and db[3].shape == S.shape and (db[0][:, :d] == 0).all()
    tm, tb = fit_pair(data, db, y, 'poe')
    zb = D.logits(tb, db)
    assert _nll(D.logits(tm, data) + zb, y) <= _nll(zb, y) + 1e-3
    s = scaled_targets(torch.randn(N, K), torch.randn(N, K), y, M)
    assert torch.allclose(s.sum(-1), torch.ones(N)) and (s[:10, 3] == 0).all()
    tm2, _ = fit_pair(data, db, y, 'conf')
    assert D.logits(tm2, data).shape == (N, K)
    z = torch.zeros(N, K)
    assert device(None, data, V, z)[0] is z and device(None, data, V, z)[1] == []
    assert size_mb({'poe': {'main': tm, 'bias': tb}}) > 0 and size_mb({}) == 0
    e = extras(torch.tensor([[1.0, -2.0, -1e9]]), True)
    assert n_extra('poe+neg') == 2 and n_extra('conf') == 1 and n_extra('') == 0 and e[1].tolist() == [[-1.0, 2.0, -1e9]]
    print('student_poe ok')


if __name__ == '__main__':
    _test()
