"""Group-honest pair teachers (nested cross-fitting of the decision-encoder teacher). Imported by the runtime's fit code under
the bare name ``train_hcf`` (ej.train.env.register).

The decision encoder (runtime train_merge) is trained once per group fold k WITHOUT fold k; its out-of-fold logits are the
distillation targets of the distilled decision-encoder expert. A leave-one-group-out (LOGO) row for group g (fold a) that
distils those targets on a row of fold b != a would learn from a teacher that SAW g. Here the missing teachers of the nested
scheme (double cross-fitting, Newey & Robins 2018; cluster cross-fitting, Chiang et al. 2022) are trained: one decision encoder
per fold PAIR {a, b}, fine-tuned WITHOUT both folds (same recipe, its checkpoint selected on the held-out slice of its own
training rows only), scored on the rows of folds a and b. Every target of the LOGO rows for g then comes from a teacher that
never saw g. Output: <EDGE_CKPT>/hcf-<sha256 of this file + the teacher code + the pool>/p<a><b>.pt (logits + row index only).
CPU training; ``python -m ej.train teachers`` runs every part, fit() trains missing parts in-process."""
import hashlib
import itertools
import os
import sys
import time

import numpy as np
import torch


def _rt():
    """(student_dec, train_merge) from the runtime (imported after ej.train.env.setup)."""
    import student_dec as DEC
    import train_merge as TM
    return DEC, TM


def pairs():
    """Names of the fold-pair parts: p01, p02, ..."""
    DEC, _ = _rt()
    return tuple(f'p{a}{b}' for a, b in itertools.combinations(range(len(DEC.FOLDS)), 2))


def __getattr__(name):
    if name == 'PAIRS':  # the runtime reads TH.PAIRS
        return pairs()
    raise AttributeError(name)


def ckpt_dir():
    """Directory of the pair teachers for this code and pool."""
    DEC, TM = _rt()
    h = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read())
    for f in TM.DEPS:
        h.update(open(os.path.join(TM.HERE, f), 'rb').read())
    h.update(open(TM.POOL, 'rb').read())
    return f'{TM.ROOT}/hcf-{h.hexdigest()[:16]}'


def run_pair(part, recs, log=lambda s: print(s, file=sys.stderr, flush=True)):
    """Train the teacher without folds a and b (part 'p<a><b>') on pool records `recs`, save and return its logits."""
    DEC, TM = _rt()
    t0, a, b = time.time(), int(part[1]), int(part[2])
    its = DEC.units(recs)
    hold = DEC.hold_mask([i[0] for i in its])
    tok, low = DEC.lower()
    S = DEC.states(tok, low, its)
    fold = np.array([DEC.fold_of(i[5]) for i in its])
    tr, ev = np.flatnonzero((fold != a) & (fold != b)), np.flatnonzero((fold == a) | (fold == b))
    torch.manual_seed(0)
    top, head = DEC.train([its[n] for n in tr], S, hold[tr])
    out = {'z': DEC.logits(top, head, S, [its[n] for n in ev]).half(), 'idx': torch.tensor(ev), 'items': TM._items_key(its),
           'excluded_folds': (a, b), 'trained_groups': sorted({its[n][5] for n in tr}), 'seconds': round(time.time() - t0, 1)}
    d = ckpt_dir()
    os.makedirs(d, exist_ok=True)
    torch.save(out, f'{d}/{part}.tmp')
    os.replace(f'{d}/{part}.tmp', f'{d}/{part}.pt')
    log(f'pair teacher {part}: {out["seconds"]} s -> {d}/{part}.pt (trained on {out["trained_groups"]})')
    return out


def load(recs):
    """{(a, b): dict} for every fold pair; None when any pair is missing (fit() then trains them)."""
    DEC, TM = _rt()
    d, key, out = ckpt_dir(), TM._items_key(DEC.units(recs)), {}
    for p in pairs():
        f = f'{d}/{p}.pt'
        if not os.path.exists(f):
            return None
        out[(int(p[1]), int(p[2]))] = torch.load(f, weights_only=True)  # tensors, ints, strings only
        assert out[(int(p[1]), int(p[2]))]['items'] == key, f'{f} was trained on other items'
    return out
