"""Fit side of the distilled NLI reader: teacher labels on a transfer set of training texts, pair heads (full and held-out
variants) memoised by a content hash, and device_features for prediction."""
import hashlib
import os
import sys

import numpy as np
import torch

import student_dn as DN
import student_nli as NLI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ("student_dn.py", "student_dnfit.py", "student_tok.py")


def _key(ts, Z, X):
    h = hashlib.sha256()
    for f in SRC:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update(repr([t[:3] for t in ts]).encode()); h.update(Z.numpy().tobytes()); h.update(X.numpy().tobytes())
    return h.hexdigest()[:16]


def heads(items, hold):
    """{'net': full head, 'pairs': transfer pairs, 'z': {name: (P, 3) predicted logits}} (memoised checkpoint)."""
    ts = DN.transfer_set(items)
    Z = torch.tensor(NLI.margins([(p, h) for p, h, _, _, _ in ts]))
    bank = DN.Bank(sorted({x[0] for x in ts}), sorted({(x[1], x[2]) for x in ts}))
    X = bank.x([(p, h, k) for p, h, k, _, _ in ts]).half()
    path = f'{ROOT}/dn-{_key(ts, Z, X)}.pt'
    if os.path.exists(path):
        print('dn: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    val = np.array([DN._salt(p, 'val') == 0 for p, _, _, _, _ in ts])
    pg, hg = np.array([x[3] for x in ts]), np.array([x[4] for x in ts])
    hs = {i['state'] for i, h in zip(items, hold) if h}
    hp = np.array([p in hs for p, _, _, _, _ in ts])
    out = {'pairs': [t[:3] for t in ts], 'z': {}, 'rep': {}}
    for name, keep in [('full', np.ones(len(ts), bool)), ('noho', ~hp)] + [(g, (pg != g) & (hg != g)) for g in sorted(set(pg))]:
        k = np.flatnonzero(keep)
        net, v = DN.train(X[k], Z[k], val[k])
        zp = DN.predict(net, X)
        o = torch.tensor(~keep)
        dm = lambda z: z[:, 1] - z[:, 0]
        out['rep'][name] = {'val': v, 'oof_r': round(float(np.corrcoef(dm(zp[o]).numpy(), dm(Z[o]).numpy())[0, 1]), 3) if o.any() else None}
        out['z'][name] = zp.half()
        print('dn head', name, out['rep'][name], file=sys.stderr, flush=True)
        if name == 'full':
            out['net'] = net
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    print('dn: computed ->', path, out['rep'], file=sys.stderr)
    return out


def _lookup(pairs, z):
    return {p: r.float() for p, r in zip(pairs, z)}


def fit_expert(items, M, y, hold, gid):
    """Same contract as student_nli.fit_all (ho / logo / full / rep) on distilled features; plus the device head."""
    hd = heads(items, hold)
    T = NLI.rows(items)
    Xf = DN.features(items, _lookup(hd['pairs'], hd['z']['full']))
    Xh = DN.features(items, _lookup(hd['pairs'], hd['z']['noho']))
    Xl = torch.zeros_like(Xf)
    for g in sorted(set(gid)):
        o = torch.tensor(np.asarray(gid) == g)
        if g in hd['z']:
            f = DN.features([it for it, m in zip(items, o.tolist()) if m], _lookup(hd['pairs'], hd['z'][g]))
            Xl[torch.nonzero(o).squeeze(1), :f.shape[1]] = f
    l2, rep = NLI.logo_l2(Xl, M, T, y, gid)
    tr, ho = torch.tensor(~hold), torch.tensor(hold)
    out = {'ho': NLI.logits(NLI.train(Xh[tr], M[tr], T[tr], y[tr], l2), Xh[ho], M[ho], T[ho]).masked_fill(~M[ho], 0), 'logo': {},
           'full': NLI.train(Xf, M, T, y, l2), 'rep': {'l2': l2, 'logo': rep, 'heads': hd['rep']}, 'net': hd['net']}
    for g in sorted(set(gid)):
        o = torch.tensor(np.asarray(gid) == g)
        out['logo'][g] = NLI.logits(NLI.train(Xl[~o], M[~o], T[~o], y[~o], l2), Xl[o], M[o], T[o]).masked_fill(~M[o], 0)
    return out


def device_features(items, net):
    """Predict-time features: one pair-head call per (state, hypothesis) on the shared e5 embeddings; no NLI model."""
    pairs = DN.device_pairs(items)
    if not pairs:
        return torch.zeros(len(items), max(len(i['opts']) for i in items), NLI.NF)
    bank = DN.Bank(sorted({p for p, _, _ in pairs}), sorted({(h, k) for _, h, k in pairs}))
    return DN.features(items, _lookup(pairs, DN.predict(net, bank.x(pairs))))
