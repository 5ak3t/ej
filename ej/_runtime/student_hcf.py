"""Group-honest nested cross-fitting: recomputes the leave-one-group-out rows of the distilled experts with teachers that
never saw the held-out group, so the pool coefficients for unseen option slots are fitted on honest logits. Fit-time code."""
import hashlib
import os
import sys

import numpy as np
import torch

import student_dd as DD
import student_rich as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ('student_dd.py', 'student_rich.py', 'student_read.py')  # + VERSION: edits elsewhere in this file keep the memo
VERSION = 'v2'  # bump when logo_dd / logo_rich change
USE_PG = os.environ.get('EDGE_PG', '0') == '1'  # pseudo-groups (student_pg): pilot only (banking halves: slot-3 LOGO .80/.85, worse than source-level .75) -> off, not evaluated
DD_LOGO = 'student'  # slot-3 unseen-key evidence: 'student' = honest nested LOGO student; 'teacher' = honest fold-teacher OOF logits


def _sub(data, sel):
    return tuple(t[sel] for t in data)


def _pad(z, K):
    out = torch.full((z.shape[0], K), -1e9)
    out[:, :z.shape[1]] = z.float()
    return out


def teacher_groups(name, groups, folds):
    """Training groups of a decision-encoder teacher: 'merge.fold<k>' (train_merge) or 'hcf.pair<a><b>' (train_hcf)."""
    ex = set(folds[int(name[-1])]) if name.startswith('merge') else set(folds[int(name[-2])]) | set(folds[int(name[-1])])
    return sorted(set(groups) - ex)


def dec_targets(z_oof, pairs, M, gid, g, fold_of):
    """Teacher distribution for every row the LOGO student of g trains on + {group h: teacher name}. Rows of g's own fold keep
    the train_merge out-of-fold logits (that teacher excluded g's fold); rows of fold b get the pair {fold(g), b} teacher."""
    a, z, names = fold_of(g), z_oof.clone(), {}
    for h in sorted(set(gid) - {g}):
        b = fold_of(h)
        if b == a:
            names[h] = f'merge.fold{a}'
            continue
        key = (min(a, b), max(a, b)); p = pairs[key]
        rows = torch.tensor(np.flatnonzero(gid == h))
        pos = {int(r): n for n, r in enumerate(p['idx'].tolist())}
        z[rows] = _pad(p['z'][torch.tensor([pos[int(r)] for r in rows])], M.shape[1])
        names[h] = f'hcf.pair{key[0]}{key[1]}'
    pt = torch.softmax(z.float().masked_fill(~M, -1e9), -1).masked_fill(~M, 0)
    return pt, names


def _key(tag, tensors, hold, gid, extra=b''):
    h = hashlib.sha256((tag + VERSION).encode())
    for f in SRC:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    for t in tensors:
        h.update(t.detach().contiguous().float().numpy().tobytes())
    h.update(np.asarray(hold).tobytes()); h.update('\x1f'.join(gid).encode()); h.update(extra)
    return h.hexdigest()[:16]


def _memo(path, compute):
    if os.path.exists(path):
        print('hcf: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    out = compute()
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    print('hcf: computed ->', path, file=sys.stderr)
    return out


def logo_dd(data, y, z_oof, pairs, hold, gid, fold_of):
    """Honest LOGO logits of the slot-3 student: {'logo': {g: logits}, 'ep': {g: epochs}, 'teachers': {g: {h: teacher}}}."""
    gid = np.asarray(gid)
    extra = b''.join(pairs[k]['z'].float().numpy().tobytes() for k in sorted(pairs))

    base = f"{ROOT}/hcfddg-{_key('dd', tuple(data) + (y, z_oof), hold, list(gid), extra)}"
    G = sorted(set(gid)); os.makedirs(ROOT, exist_ok=True)

    def one(g):  # one LOGO student; several processes may share the work (per-group files, O_EXCL lock)
        pt, names = dec_targets(z_oof, pairs, data[2], gid, g, fold_of)
        o = torch.tensor(gid == g); trn, val = torch.tensor(~hold) & ~o, torch.tensor(hold) & ~o
        net, ep = DD.train(_sub(data, trn), y[trn], pt[trn], val=(_sub(data, val), y[val], pt[val]))  # early-stopped model itself
        r = {'logo': R.logits(net, _sub(data, o)), 'ep': ep, 'teachers': names}
        print('hcf dd', g, ep, round(torch.nn.functional.cross_entropy(r['logo'], y[o]).item(), 4), file=sys.stderr, flush=True)
        return r
    for k in [int(x) for x in os.environ.get('HCF_ORDER', ','.join(map(str, range(len(G))))).split(',')]:
        f = f'{base}-{k}.pt'
        if os.path.exists(f):
            continue
        try:
            os.close(os.open(f + '.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        torch.save(one(G[k]), f + '.tmp'); os.replace(f + '.tmp', f)
    out = {'logo': {}, 'ep': {}, 'teachers': {}}
    for k, g in enumerate(G):
        f = f'{base}-{k}.pt'
        if not os.path.exists(f):  # another process holds the lock: wait for it, else (stale lock) compute here
            import time
            for _ in range(720):
                if os.path.exists(f) or not os.path.exists(f + '.lock'):
                    break
                time.sleep(5)
            if not os.path.exists(f):
                torch.save(one(g), f + '.tmp'); os.replace(f + '.tmp', f)
        r = torch.load(f, weights_only=False)
        out['logo'][g], out['ep'][g], out['teachers'][g] = r['logo'], r['ep'], r['teachers']
    return out


def logo_rich(data, y, hold, gid):
    """Honest LOGO logits of the gold-only rich scorer (epochs early-stopped on the held-out slice minus g)."""
    gid = np.asarray(gid)

    def compute():
        out = {'logo': {}, 'ep': {}}
        for g in sorted(set(gid)):
            o = torch.tensor(gid == g); trn, val = torch.tensor(~hold) & ~o, torch.tensor(hold) & ~o
            net, ep = R.train(_sub(data, trn), y[trn], val=(_sub(data, val), y[val]))  # early-stopped model itself (one fit per g)
            out['logo'][g], out['ep'][g] = R.logits(net, _sub(data, o)), ep
            print('hcf rich', g, ep, round(torch.nn.functional.cross_entropy(out['logo'][g], y[o]).item(), 4), file=sys.stderr, flush=True)
        return out
    return _memo(f"{ROOT}/hcfrich-{_key('rich', tuple(data) + (y,), hold, list(gid))}.pt", compute)


def fold_map(groups, folds, dd, nli):
    """Auditable map {g: {expert: {'model': trained-on groups, 'targets': {h: (teacher, its training groups)}}}}."""
    out = {}
    for g in sorted(groups):
        rest = sorted(set(groups) - {g})
        out[g] = {'deep': {'model': rest}, 'centred': {'model': rest}, 'rich': {'model': rest, 'epochs_from': rest},
                  'dd': {'model': rest, 'epochs_from': rest,
                         'targets': {h: (t, teacher_groups(t, groups, folds)) for h, t in dd['teachers'][g].items()}},
                  'nli': nli[g]}
    return out


def audit(fmap):
    """Every model / teacher / head / selection behind g's LOGO logits must exclude g; returns (#checks, violations)."""
    n, bad = 0, []

    def walk(g, path, v):
        nonlocal n
        if isinstance(v, dict):
            for k, w in v.items():
                walk(g, path + (str(k),), w)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str) and '/' in v[0]:
            n += 1
            if g in v:
                bad.append((g,) + path)
        elif isinstance(v, (list, tuple)):
            for k, w in enumerate(v):
                walk(g, path + (str(k),), w)
    for g, m in fmap.items():
        walk(g, (), m)
    return n, bad


def honest(items, data, y, hold, gid, ck, recs, up, dd, nl, folds, fold_of):
    """Replace the LOGO (unseen-key) logits of slots 1, 3 and 9 in place by their group-honest versions; returns the audit report.
    deep / centred deep LOGO logits (student_up) are kept: convex, fixed L2, trained without g."""
    import student_hdn as HDN
    import train_hcf as TH
    gid = np.asarray(gid); K = data[2].shape[1]
    pairs = TH.load(recs) or {(int(p[1]), int(p[2])): TH.run_pair(p, recs) for p in TH.PAIRS}  # no checkpoint = train here
    z_oof = torch.full((len(items), K), -1e9)
    for k in range(len(folds)):
        z_oof[ck[f'fold{k}']['idx']] = _pad(ck[f'fold{k}']['z'], K)
    hd, hr, hn = logo_dd(data, y, z_oof, pairs, hold, gid, fold_of), logo_rich(data, y, hold, gid), HDN.logo(items, data[2], y, hold, gid)
    for g in sorted(set(gid)):
        zg, zgc, _ = up['logo'][g]
        up['logo'][g] = (zg, zgc, hr['logo'][g])
        dd['logo'][g], nl['logo'][g] = hd['logo'][g], hn['logo'][g]
        if DD_LOGO == 'teacher':  # the fold teacher the device student imitates on g (trained without g's whole fold)
            dd['logo'][g] = z_oof[torch.tensor(gid == g)]
    fmap = fold_map(set(gid), folds, hd, hn['map'])
    if USE_PG:  # gap closer: large label-space sources cross-fitted by pseudo-group (student_pg); same honesty, finer unit
        import student_pg as PG
        pgo, pg = PG.logo(items, data, y, hold, gid, z_oof, pairs, folds, fold_of)
        for s_, r in sorted(pgo.items()):
            g = s_.split('#')[0]; pos = torch.tensor(np.searchsorted(np.flatnonzero(gid == g), r['rows']))
            zg, zgc, rg = (t.clone() for t in up['logo'][g])
            zg[pos], zgc[pos], rg[pos] = r['deep'], r['centred'], r['rich']
            lab = {items[n]['opts'][items[n]['y']] for n in r['rows']}  # candidates restricted to s's (unseen) labels, as on a
            for p, n in zip(pos.tolist(), r['rows']):  # new label space; options seen in training (other pseudo-group) masked
                keep = torch.tensor([o in lab for o in items[n]['opts']] + [False] * (K - len(items[n]['opts'])))
                if keep.sum() >= 2:
                    zg[p] = zg[p].masked_fill(~keep, -1e9)
            up['logo'][g] = (zg, zgc, rg)
            dd['logo'][g] = dd['logo'][g].clone(); dd['logo'][g][pos] = r['dd']
            nl['logo'][g] = nl['logo'][g].clone(); nl['logo'][g][pos] = r['nli']
            fmap[g].setdefault('pseudo', {})[s_] = r['map']
    n, bad = audit(fmap)
    path = os.environ.get('EDGE_FOLDMAP')
    if path:
        import json
        json.dump({'fold_map': fmap, 'checks': n, 'violations': bad, 'ep_dd': hd['ep'], 'ep_rich': hr['ep'],
                   'l2_nli': {g: list(map(float, v)) for g, v in hn['l2'].items()}}, open(path, 'w'), indent=1, default=str)
    print('hcf audit', n, 'checks', len(bad), 'violations', bad, file=sys.stderr, flush=True)
    assert not bad, bad
    return {'checks': n, 'violations': len(bad), 'ep_dd': hd['ep'], 'ep_rich': hr['ep']}
