"""Bayesian hierarchical stacking of the unseen-slot expert pool: per-cell log-linear pool weights are partially pooled (root,
question type, cell) with group random effects under a Laplace approximation, and the predictive for a new group is
selected or stacked per regime. mix_rows applies the fitted pool at prediction time."""
import math
import os
import sys

import torch

import student_hbs2 as HB2
import student_pool as P
import student_sel as SEL

TAU_GRID = (0.1, 0.3, 1.0, 3.0)  # a priori; both scales on the pre-softplus / logit scale
ROOT_SD = 3.0
MODE = 'select'  # new-group predictive: one of MODES, or 'select' = chosen in fit by a group-level LOGO check
MODES = ('flat', 'mean', 'groups', 'gauss')
BY_REGIME = True  # select the mode per regime (json / plain-text cells), not one global mode
S_HALF = 16  # 32 antithetic draws per row at predict time
TYPES = (0, 1, 2)


def _stack(rows):
    K = max(r[0].shape[1] for r in rows)
    Z = torch.full((rows[0][0].shape[0], len(rows), K), -1e9)
    for n, r in enumerate(rows):
        Z[:, n, :r[0].shape[1]] = r[0]
    return Z, torch.tensor([r[1] for r in rows])


def _root0(E):
    th = torch.full((E + 1,), -5.0); th[0] = P.INIT; th[E] = math.log(0.05 / 0.95)
    return th


def _logp(Z, y, phi):
    """log pooled probability of y; phi: (n, E + 1) per row."""
    E = Z.shape[0]
    p = P.mix(Z, torch.nn.functional.softplus(phi[:, :E]), torch.sigmoid(phi[:, E:]))
    return p[torch.arange(len(y)), y].clamp(min=1e-15).log()


class Tree:
    """Node structure of the unseen regime: root, type, cell (type, json), (cell, group)."""

    def __init__(self, groups):
        self.blocks, self.nodes = [], ['m']
        for k in sorted(groups, key=str):
            t, s, j = k
            if s:
                continue
            by = {}
            for r in groups[k]:
                by.setdefault(r[2], []).append(r)
            for g in sorted(by):
                self.blocks.append(((t, j), g, by[g]))
        for (t, j), g, _ in self.blocks:
            for nd in (('t', t), ('c', t, j), ('u', t, j, g)):
                if nd not in self.nodes:
                    self.nodes.append(nd)
        self.idx = {nd: i for i, nd in enumerate(self.nodes)}
        self.Pm = torch.zeros(len(self.blocks), len(self.nodes))
        for b, ((t, j), g, _) in enumerate(self.blocks):
            for nd in ('m', ('t', t), ('c', t, j), ('u', t, j, g)):
                self.Pm[b, self.idx[nd]] = 1
        rows = [r for _, _, rs in self.blocks for r in rs]
        self.Z, self.y = _stack(rows)
        self.bid = torch.cat([torch.full((len(rs),), b) for b, (_, _, rs) in enumerate(self.blocks)])
        self.E = self.Z.shape[0]

    def prec(self, tc, tg):
        """Prior precision per node (root: 1 / ROOT_SD^2) and the root prior mean."""
        return torch.tensor([ROOT_SD ** -2 if nd == 'm' else tg ** -2 if nd[0] == 'u' else tc ** -2 for nd in self.nodes])

    def neg_post(self, th, tc, tg):
        m0 = torch.zeros_like(th); m0[0] = _root0(self.E)
        phi = (self.Pm @ th)[self.bid]
        return -_logp(self.Z, self.y, phi).sum() + 0.5 * (self.prec(tc, tg)[:, None] * (th - m0) ** 2).sum()

    def fit(self, tc, tg, th0=None):
        th = (th0.clone() if th0 is not None else torch.zeros(len(self.nodes), self.E + 1))
        if th0 is None:
            th[0] = _root0(self.E)
        th.requires_grad_()
        opt = torch.optim.LBFGS([th], max_iter=300, tolerance_grad=1e-6, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = self.neg_post(th, tc, tg)
            loss.backward()
            return loss
        opt.step(closure)
        return th.detach()

    def hessian(self, th, tc, tg):
        """Posterior Hessian over all node parameters = sum_b (p_b p_b^T) kron h_b + prior precision (h_b: 10 x 10 per block)."""
        phi = self.Pm @ th
        D, N = self.E + 1, len(self.nodes)
        H = torch.zeros(N, D, N, D)
        for b in range(len(self.blocks)):
            sel = self.bid == b
            Zb, yb = self.Z[:, sel], self.y[sel]
            hb = torch.autograd.functional.hessian(lambda v: -_logp(Zb, yb, v.expand(len(yb), -1)).sum(), phi[b])
            pb = self.Pm[b]
            H += torch.einsum('i,j,ab->iajb', pb, pb, hb)
        H = H.reshape(N * D, N * D)
        H = 0.5 * (H + H.T) + torch.diag(self.prec(tc, tg).repeat_interleave(D))
        return H

    def evidence(self, th, tc, tg):
        """Laplace log marginal likelihood up to constants shared by every (tc, tg)."""
        H = self.hessian(th, tc, tg)
        ev = torch.linalg.eigvalsh(H.double()).clamp(min=1e-10)
        n_c = sum(nd != 'm' and nd[0] != 'u' for nd in self.nodes)
        n_u = sum(nd != 'm' and nd[0] == 'u' for nd in self.nodes)
        D = self.E + 1
        return (-self.neg_post(th, tc, tg).item() - D * (n_c * math.log(tc) + n_u * math.log(tg)) - 0.5 * ev.log().sum().item()), H


def fit_unseen(groups, grid=TAU_GRID, fixed=None):
    """Type-II ML over (tau_c, tau_g) on the grid (or `fixed`), then the per-cell new-group predictive N(mu_c, Sigma_c + tg^2)."""
    tree = Tree(groups)
    big = HB2.big(tree)  # arrow-structured evidence with many groups (student_hbs2); dense path otherwise
    best, rep, th = None, {}, None
    for tc in (grid if fixed is None else (fixed[0],)):
        for tg in (grid if fixed is None else (fixed[1],)):
            th = tree.fit(tc, tg, th)
            ev, H = HB2.evidence(tree, th, tc, tg, _logp) if big else tree.evidence(th, tc, tg)
            rep[(tc, tg)] = round(ev, 2)
            if best is None or ev > best[0]:
                best = (ev, tc, tg, th, H)
    _, tc, tg, th, H = best
    D = tree.E + 1
    if big:
        covT, tpos = H
        cov_of = lambda ii: covT[[tpos[i] for i in ii]][:, :, [tpos[i] for i in ii]]  # noqa: E731
    else:
        w, V = torch.linalg.eigh(H.double())
        cov = (V / w.clamp(min=1e-6)) @ V.T  # posterior covariance (Laplace), non-PD directions clamped
        cov = cov.reshape(len(tree.nodes), D, len(tree.nodes), D)
        cov_of = lambda ii: cov[ii][:, :, ii]  # noqa: E731
    cells = {}
    for t in TYPES:
        for j in (False, True):
            path = [nd for nd in ('m', ('t', t), ('c', t, j)) if nd in tree.idx]
            ii = [tree.idx[nd] for nd in path]
            mu = th[ii].sum(0)
            S = cov_of(ii).sum((0, 2)).float() + tg ** 2 * torch.eye(D)
            S = S + (3 - len(path)) * tc ** 2 * torch.eye(D)  # missing type / cell node: its prior variance
            ws, Vs = torch.linalg.eigh(0.5 * (S + S.T))
            U = HB2.cap(tree, th, t, j)  # training groups' effects in c (at most HB2.CAP)
            cells[(t, False, j)] = (mu, Vs * ws.clamp(min=0).sqrt(), torch.stack(U) if U else mu[None] * 0)
    info = {'tau_c': tc, 'tau_g': tg, 'evidence': {f'{a}/{b}': v for (a, b), v in rep.items()},
            'a_cell': {str(k): [round(x, 3) for x in torch.nn.functional.softplus(v[0][:-1]).tolist()] + [round(torch.sigmoid(v[0][-1]).item(), 3)]
                       for k, v in cells.items()}, 'blocks': len(tree.blocks), 'rows': len(tree.y)}
    return {'cells': cells, 'tau': (tc, tg)}, info


def _draws(D):
    g = torch.Generator().manual_seed(0)
    e = torch.randn(S_HALF, D, generator=g, device='cpu')  # explicit CPU draw: identical on CPU, valid under the GPU shim's native mode
    return torch.cat([e, -e])


def mix_cell(Z, cell, mode=None):
    """New-group predictive of one unseen cell: mean of the log-linear pool over parameter draws. Z: (E, n, K).
    mode 'gauss': phi ~ N(mu_c, Sigma_c + tau_g^2 I) (S antithetic draws); 'groups': phi = mu_c + u_{c,g} for each training
    group g of the cell (the empirical distribution of the group effects; nonparametric random-effects predictive); 'mean': mu_c."""
    mu, L, U = cell
    E = Z.shape[0]
    phis = mu[None] if mode == 'mean' else mu[None] + U if mode == 'groups' else mu[None] + _draws(E + 1) @ L.T
    out = 0
    for phi in phis:
        A = torch.nn.functional.softplus(phi[:E]).expand(Z.shape[1], -1)
        out = out + P.mix(Z, A, torch.sigmoid(phi[E:]).expand(Z.shape[1], 1))
    return out / len(phis)


def _pred(un, Z, k, mode=None):
    """Unseen-cell predictive under `mode` ('flat' = the per-cell student_pool fit, else mix_cell)."""
    mode, E = mode or (un['mode'][k[2]] if isinstance(un['mode'], dict) else un['mode']), Z.shape[0]
    if isinstance(mode, dict):  # regime-level stacking of the predictives (student_hbs2, many groups only)
        return sum(w * _pred(un, Z, k, m) for m, w in zip(MODES, mode['stack']))
    if mode == 'flat':
        par = torch.tensor([P.params_for(un['flat'], k, E)] * Z.shape[1])
        return P.mix(Z, par[:, :E], par[:, E:])
    return mix_cell(Z, un['cells'][k], mode)


def _unseen(groups, fixed=None):
    un, info = fit_unseen(groups, fixed=fixed)
    un['flat'] = P.fit({k: v for k, v in groups.items() if not k[1]})
    return un, info


def select_mode(groups, tau):
    """Group-level model check (pool rows only): for every training group g, fit on the other groups and score g's unseen rows
    as a NEW group (taus fixed at the full-pool evidence optimum: 2 scalars, for fit time); the predictive form with the lowest mean per-group NLL (groups weighted equally) is used on the device."""
    uns = {k: v for k, v in groups.items() if not k[1]}
    per, js = {m: [] for m in MODES}, []
    for g in sorted({r[2] for v in uns.values() for r in v}):
        un, _ = _unseen({k: w for k, w in ((k, [r for r in v if r[2] != g]) for k, v in uns.items()) if w}, fixed=tau)
        rows = {k: [r for r in v if r[2] == g] for k, v in uns.items()}
        n = sum(len(v) for v in rows.values())
        js.append(sum(len(v) for k, v in rows.items() if k[2]) * 2 > n)  # the group's regime: structured (json) states or text
        acc = {}
        for m in MODES:
            tot = 0.0
            for k, rs in rows.items():
                if rs:
                    Z, y = _stack(rs)
                    pk = _pred(un, Z, k, m)
                    tot -= pk[torch.arange(len(y)), y].clamp(min=1e-15).log().sum().item()
                    acc.setdefault(m, {}).setdefault(k[0], []).extend((pk.argmax(1) == y).tolist())  # honest per-type accuracy (report)
            per[m].append(tot / n)
        print('hbs select', g, {m: round(v[-1], 4) for m, v in per.items()}, {m: {t: round(sum(v) / len(v), 3) for t, v in a.items()}
                                                                              for m, a in acc.items()}, file=sys.stderr, flush=True)
    if not BY_REGIME:
        score = {m: round(sum(v) / len(v), 4) for m, v in per.items()}
        return min(score, key=score.get), score
    out, score = {}, {}
    for j in (False, True):  # one check per regime (json groups are few: 3 TD workflows) -> the mode per regime
        sc = {m: round(sum(x for x, jj in zip(v, js) if jj == j) / max(1, sum(jj == j for jj in js)), 4) for m, v in per.items()}
        out[j], score[f'json={j}'] = (min(sc, key=sc.get) if any(jj == j for jj in js) else 'flat'), sc
    return out, score


def fit(groups, unit=None):
    """{'seen': student_pool dict for seen keys, 'unseen': hierarchical model + flat fit + chosen mode}, report.
    unit: {group: held-out block} when the pool has many groups (student_kf) -> block model check + regime stacking (student_hbs2)."""
    seen = {k: v for k, v in groups.items() if k[1]}
    if os.environ.get('EDGE_HBS_DUMP'):  # analysis only: the pool-calibration rows (pool data, no dev)
        torch.save(groups, os.environ['EDGE_HBS_DUMP'])
    un, info = _unseen(groups)
    sel = (lambda: HB2.select_blocks(groups, un['tau'], unit, _unseen, _pred, _stack, MODES)) if unit else (lambda: select_mode(groups, un['tau']))
    mode, score = sel() if MODE == 'select' else (MODE, {})
    un['mode'] = mode
    info.update({'mode': mode, 'logo_score': score})
    print('hbs', info, file=sys.stderr, flush=True)
    return {'seen': P.fit(seen), 'unseen': un}, info


def mix_rows(model, Z, keys, n_exp):
    """Pooled distribution per row; keys: list of (type, seen, json)."""
    out = torch.zeros(Z.shape[1:])
    for k in sorted(set(keys), key=str):
        r = torch.tensor([kk == k for kk in keys])
        if k[1]:
            par = torch.tensor([P.params_for(model['seen'], k, n_exp)] * int(r.sum()))
            out[r] = P.mix(Z[:, r], par[:, :n_exp], par[:, n_exp:])
        else:
            out[r] = _pred(model['unseen'], Z[:, r], k)
    return out


def sel_fit(groups, model, n_exp):
    """student_sel's correctness head with the pool replaced by this model in its 2-fold cross-fit (unseen: same taus)."""
    feats = {True: ([], [], []), False: ([], [], [])}
    for f in (0, 1):
        fit_g = {k: [r for n, r in enumerate(v) if n % 2 != f] for k, v in groups.items()}
        ev_g = {k: [r for n, r in enumerate(v) if n % 2 == f] for k, v in groups.items() if len(v) >= 2 * P.MIN_CAL}
        m = {'seen': P.fit({k: v for k, v in fit_g.items() if k[1]}), 'unseen': _unseen(fit_g, fixed=model['unseen']['tau'])[0]}
        m['unseen']['mode'] = model['unseen']['mode']
        for k, rows in sorted(ev_g.items(), key=lambda kv: str(kv[0])):
            Z, y = _stack(rows)
            p = mix_rows(m, Z, [k] * len(rows), n_exp)
            n = len(rows)
            phi = SEL.features(Z, p, torch.full((n,), k[0]), torch.full((n,), bool(k[1])), torch.full((n,), bool(k[2])))
            for lst, v in zip(feats[bool(k[1])], (phi, p, y)):
                lst.append(v)
    K = max(p.shape[1] for v in feats.values() for p in v[1])
    out, rep = {}, {}
    for s, (phis, ps, ys) in feats.items():
        if phis:
            out[s], rep['seen' if s else 'unseen'] = SEL._head(torch.cat([torch.nn.functional.pad(p, (0, K - p.shape[1])) for p in ps]),
                                                               torch.cat(phis), torch.cat(ys))
        else:
            out[s] = None
    return out, rep
