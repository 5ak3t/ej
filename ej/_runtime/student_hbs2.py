"""Many-group extensions of the hierarchical stacking: block-structured evidence and Hessian computations that scale with the
number of groups, a block-wise model check, and simplex stacking weights of the new-group predictives on the held-out log
score."""
import hashlib
import sys

import torch

import student_kf as KF

STACK_MIN = 12
CAP = 32
EM_ITERS = 500


def big(tree):
    return len({g for _, g, _ in tree.blocks}) > KF.MAX_LOGO


def cap(tree, th, t, j):
    """Group effects of cell (t, j) for the 'groups' predictive; at most CAP of them (deterministic by sha256 of the name)."""
    nodes = [(nd, i) for nd, i in tree.idx.items() if nd[0] == 'u' and nd[1:3] == (t, j)]
    if len(nodes) > CAP:
        nodes = sorted(nodes, key=lambda x: hashlib.sha256(x[0][3].encode()).hexdigest())[:CAP]
    return [th[i] for _, i in nodes]


def _block_hess(tree, phi_b, b, logp):
    sel = tree.bid == b
    Zb, yb = tree.Z[:, sel], tree.y[sel]
    hb = torch.autograd.functional.hessian(lambda v: -logp(Zb, yb, v.expand(len(yb), -1)).sum(), phi_b).double()
    return 0.5 * (hb + hb.T)


def evidence(tree, th, tc, tg, logp):
    """(Laplace log evidence as student_hbs.Tree.evidence, (top-node covariance (nT, D, nT, D), {node index: top position}))."""
    D, N = tree.E + 1, len(tree.nodes)
    top = [i for i, nd in enumerate(tree.nodes) if nd == 'm' or nd[0] != 'u']
    pos = {i: n for n, i in enumerate(top)}
    prec = tree.prec(tc, tg).double()
    nT = len(top)
    A = torch.diag(prec[top].repeat_interleave(D))
    phi = tree.Pm @ th
    I = torch.eye(D, dtype=torch.float64)
    ld_u = 0.0
    for b in range(len(tree.blocks)):
        hb = _block_hess(tree, phi[b], b, logp)
        on = torch.nonzero(tree.Pm[b]).flatten().tolist()
        u = next(i for i in on if i not in pos)
        e = torch.zeros(nT, 1, dtype=torch.float64)
        e[[pos[i] for i in on if i in pos]] = 1
        Eb = torch.kron(e, I)
        w, V = torch.linalg.eigh(hb + prec[u] * I)
        w = w.clamp(min=1e-10)
        ld_u += w.log().sum().item()
        C = Eb @ hb
        A = A + Eb @ hb @ Eb.T - (C @ V / w) @ (V.T @ C.T)
    A = 0.5 * (A + A.T)
    ws, Vs = torch.linalg.eigh(A)
    n_c = sum(nd != 'm' and nd[0] != 'u' for nd in tree.nodes)
    n_u = N - 1 - n_c
    ev = (-tree.neg_post(th, tc, tg).item() - D * (n_c * torch.log(torch.tensor(tc)).item() + n_u * torch.log(torch.tensor(tg)).item())
          - 0.5 * (ld_u + ws.clamp(min=1e-10).log().sum().item()))
    cov = ((Vs / ws.clamp(min=1e-6)) @ Vs.T).reshape(nT, D, nT, D)
    return ev, (cov, pos)


def stack_weights(P, wr, iters=EM_ITERS):
    """Simplex weights w maximising sum_r wr_r log sum_m w_m P[r, m] (EM; P: (n, M) predictive prob of the gold label)."""
    P = P.double().clamp(min=1e-15); wr = wr.double() / wr.sum()
    w = torch.full((P.shape[1],), 1.0 / P.shape[1], dtype=torch.float64)
    for _ in range(iters):
        q = P * w
        q = q / q.sum(1, keepdim=True)
        w = (wr[:, None] * q).sum(0)
    return [round(x, 6) for x in w.tolist()]


def select_blocks(groups, tau, unit, unseen, pred, stack, modes):
    """Block-out model check of the new-group predictive; per regime: stacked weights ({'stack': w}) or the argmin mode."""
    uns = {k: v for k, v in groups.items() if not k[1]}
    rec = {False: [], True: []}  # per held-out row: (group, type, [p_m(gold)], [correct_m])
    for B in sorted({unit[r[2]] for v in uns.values() for r in v}):
        un, _ = unseen({k: w for k, w in ((k, [r for r in v if unit[r[2]] != B]) for k, v in uns.items()) if w}, fixed=tau)
        for k, rs in sorted(uns.items(), key=lambda kv: str(kv[0])):
            rows = [r for r in rs if unit[r[2]] == B]
            if not rows:
                continue
            Z, y = stack(rows)
            P = [pred(un, Z, k, m) for m in modes]
            for n, r in enumerate(rows):
                rec[bool(k[2])].append((r[2], k[0], [p[n, y[n]].item() for p in P], [int(p[n].argmax() == y[n]) for p in P]))
        print('hbs block', B, file=sys.stderr, flush=True)
    out, score = {}, {}
    for j, rs in rec.items():
        if not rs:
            out[j] = 'flat'
            continue
        cnt = {}
        for g, *_ in rs:
            cnt[g] = cnt.get(g, 0) + 1
        wr = torch.tensor([1.0 / cnt[g] for g, *_ in rs])
        P = torch.tensor([r[2] for r in rs])
        sc = {m: round((-(wr * P[:, n].clamp(min=1e-15).log()).sum() / wr.sum()).item(), 4) for n, m in enumerate(modes)}
        acc = {t: {m: round(sum(r[3][n] for r in rs if r[1] == t) / max(1, sum(r[1] == t for r in rs)), 3) for n, m in enumerate(modes)}
               for t in sorted({r[1] for r in rs})}
        if len(cnt) >= STACK_MIN:
            w = stack_weights(P, wr)
            Pm = (P.double() * torch.tensor(w, dtype=torch.float64)).sum(1)
            sc['stack'] = round((-(wr * Pm.clamp(min=1e-15).log()).sum() / wr.sum()).item(), 4)
            out[j] = {'stack': w}
        else:
            out[j] = min(sc, key=sc.get)
        score[f'json={j}'] = {'nll': sc, 'acc_by_type': acc, 'groups': len(cnt)}
    print('hbs select_blocks', out, score, file=sys.stderr, flush=True)
    return out, score


def _test():
    P = torch.tensor([[0.9, 0.1], [0.8, 0.3], [0.7, 0.2]])
    w = stack_weights(P, torch.ones(3))
    assert abs(sum(w) - 1) < 1e-9 and w[0] > 0.99
    P2 = torch.tensor([[0.9, 0.05], [0.05, 0.9]])
    w2 = stack_weights(P2, torch.ones(2))
    assert abs(w2[0] - 0.5) < 1e-6
    import student_hbs as HBS  # arrow evidence == dense evidence; many-group fit end to end (synthetic rows)
    g = torch.Generator().manual_seed(0)
    rows = lambda n, k: [(torch.randn(3, k, generator=g), int(torch.randint(0, k, (1,), generator=g)), f's/{n % 14}') for n in range(n)]
    groups = {(t, False, j): rows(70, 3 + t) for t in (0, 2) for j in (False, True)}
    groups[(0, True, False)] = rows(40, 3)
    tree = HBS.Tree(groups)
    th = tree.fit(1.0, 0.3)
    ev_d, H = tree.evidence(th, 1.0, 0.3)
    ev_a, (covT, pos) = evidence(tree, th, 1.0, 0.3, HBS._logp)
    assert big(tree) and abs(ev_d - ev_a) < 1e-3 * max(1, abs(ev_d)), (ev_d, ev_a)
    D, N = tree.E + 1, len(tree.nodes)
    cov = torch.linalg.inv(H.double()).reshape(N, D, N, D)
    ii = [tree.idx['m'], tree.idx[('t', 0)]]
    assert torch.allclose(cov[ii][:, :, ii], covT[[pos[i] for i in ii]][:, :, [pos[i] for i in ii]], atol=1e-5)
    unit = {f's/{n}': f'kfold/{n % 8}' for n in range(14)}
    global STACK_MIN
    STACK_MIN, old = 5, STACK_MIN
    model, info = HBS.fit(groups, unit=unit)
    STACK_MIN = old
    assert isinstance(model['unseen']['mode'][False], dict) and abs(sum(model['unseen']['mode'][False]['stack']) - 1) < 1e-6
    Z = torch.randn(3, 4, 5)
    p = HBS.mix_rows(model, Z, [(2, False, False)] * 4, 3)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    print('student_hbs2 ok', ev_d, ev_a, info['mode'])


if __name__ == '__main__':
    _test()
