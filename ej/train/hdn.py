"""Group-honest LOGO logits for the distilled NLI expert, part of the nested cross-fitting of the runtime's student_hcf.
Imported by the runtime's fit code under the bare name ``student_hdn`` (ej.train.env.register).

For the leave-one-group-out (LOGO) logits of group g:
  rows of g      <- the NLI pair head trained without g (runtime student_dnfit)
  rows of h != g <- a PAIR head trained without g AND h (every transfer pair whose premise or hypothesis comes from g or h is
                    dropped; plain-text groups only -- structured states have no NLI pairs, so their features are 0 everywhere)
  L2             <- inner leave-one-group-out over the groups != g on exactly these features (nested selection; never sees g).
The cross-encoder teacher is a public checkpoint (SNLI / MultiNLI); it saw no pool row. Memoised under <EDGE_CKPT>/hdn-*.pt
(files this training wrote itself; they hold frozenset keys, so they are read with weights_only=False)."""
import hashlib
import itertools
import os
import sys

import numpy as np
import torch

import student_dn as DN
import student_dnfit as DNF
import student_nli as NLI

HERE = os.path.dirname(DN.__file__)  # the runtime directory (the hashed sources live there)
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ('hdn-v1', 'student_dn.py', 'student_dnfit.py', 'student_tok.py', 'student_nli.py')


def pair_heads(items, hold):
    """{frozenset({g, h}): (P, 3) predicted logits of the head trained without g and h} for plain-text group pairs; memoised."""
    ts = DN.transfer_set(items)
    Z = torch.tensor(NLI.margins([(p, h) for p, h, _, _, _ in ts]))
    h_ = hashlib.sha256()
    for f in SRC:
        h_.update(open(os.path.join(HERE, f), 'rb').read() if f.endswith('.py') else f.encode())
    h_.update(repr([t for t in ts]).encode()); h_.update(Z.numpy().tobytes())
    path = f'{ROOT}/hdn-{h_.hexdigest()[:16]}.pt'
    if os.path.exists(path):
        print('hdn: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    bank = DN.Bank(sorted({x[0] for x in ts}), sorted({(x[1], x[2]) for x in ts}))
    X = bank.x([(p, h, k) for p, h, k, _, _ in ts]).half()
    val = np.array([DN._salt(p, 'val') == 0 for p, _, _, _, _ in ts])
    pg, hg = np.array([x[3] for x in ts]), np.array([x[4] for x in ts])
    out = {'pairs': [t[:3] for t in ts], 'z': {}, 'groups': {}}
    for g, h in itertools.combinations(sorted(set(pg) | set(hg)), 2):
        keep = ~np.isin(pg, [g, h]) & ~np.isin(hg, [g, h])
        k = np.flatnonzero(keep)
        net, v = DN.train(X[k], Z[k], val[k])
        out['z'][frozenset({g, h})] = DN.predict(net, X).half()
        out['groups'][frozenset({g, h})] = sorted(set(pg[k]) | set(hg[k]))
        print('hdn head', g, h, v, file=sys.stderr, flush=True)
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    return out


def logo(items, M, y, hold, gid):
    """{'logo': {g: logits}, 'l2': {g: per-row L2}, 'map': {g: audit entry}} -- honest LOGO logits of the NLI expert."""
    gid = np.asarray(gid)
    hd = DNF.heads(items, hold)
    ph = pair_heads(items, hold)
    T = NLI.rows(items)
    plain = set(hd['z']) - {'full', 'noho'}
    h_ = hashlib.sha256(b'hdn-logo-v1')
    for t in [ph['z'][k] for k in sorted(ph['z'], key=sorted)] + [hd['z'][k] for k in sorted(plain)] + [y, M, T]:
        h_.update(t.contiguous().float().numpy().tobytes())
    h_.update(np.asarray(hold).tobytes()); h_.update('\x1f'.join(gid).encode())
    path = f'{ROOT}/hdnlogo-{h_.hexdigest()[:16]}.pt'
    if os.path.exists(path):
        print('hdn: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    feats = {}

    def F(name_or_key):
        if name_or_key not in feats:
            src = hd['z'][name_or_key] if isinstance(name_or_key, str) else ph['z'][name_or_key]
            feats[name_or_key] = DN.features(items, DNF._lookup(hd['pairs'] if isinstance(name_or_key, str) else ph['pairs'], src))
        return feats[name_or_key]
    out = {'logo': {}, 'l2': {}, 'map': {}}
    for g in sorted(set(gid)):
        o = torch.tensor(gid == g)
        X = torch.zeros(len(items), M.shape[1], NLI.NF)
        heads = {}
        for h in sorted(set(gid)):
            r = torch.tensor(gid == h)
            if h not in plain:  # structured / non-NLI group: features are 0 (no pairs)
                continue
            if h == g:
                X[r] = F(g)[r]; heads[h] = ('dn.head-g', sorted(plain - {g}))
            elif g in plain:
                X[r] = F(frozenset({g, h}))[r]; heads[h] = ('hdn.pair', ph['groups'][frozenset({g, h})])
            else:
                X[r] = F(h)[r]; heads[h] = ('dn.head-h', sorted(plain - {h}))  # g has no NLI pairs: head without h never saw g
        rest = ~o
        l2, _ = NLI.logo_l2(X[rest], M[rest], T[rest], y[rest], list(gid[rest.numpy()]))
        W = NLI.train(X[rest], M[rest], T[rest], y[rest], l2)
        out['logo'][g] = NLI.logits(W, X[o], M[o], T[o]).masked_fill(~M[o], 0)
        out['l2'][g] = l2
        out['map'][g] = {'model': sorted(set(gid) - {g}), 'l2_from': sorted(set(gid) - {g}), 'features': heads}
        print('hdn logo', g, l2, round(torch.nn.functional.cross_entropy(out['logo'][g].masked_fill(~M[o], -1e9), y[o]).item(), 4),
              file=sys.stderr, flush=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    return out
