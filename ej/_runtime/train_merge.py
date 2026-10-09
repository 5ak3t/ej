#!/usr/bin/env python3
"""Training of the decision-encoder teacher (student_dec): trains on the training pool and writes checkpoint parts (full model
and per-fold models) into a directory keyed by a sha256 of the code and the pool; load() reads them. Fit-time code."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import student_dec as DEC  # noqa: E402

POOL = os.environ.get('EDGE_DATA', os.path.expanduser('~/.cache/ej/data')) + '/train/pool.jsonl'
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
DEPS = ('train_merge.py', 'student_dec.py', 'student_ft.py', 'student_x.py', 'student_enc.py')
PARTS = ('full',) + tuple(f'fold{k}' for k in range(len(DEC.FOLDS)))


def ckpt_dir():
    h = hashlib.sha256()
    for f in DEPS:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update(open(POOL, 'rb').read())
    return f'{ROOT}/{h.hexdigest()[:16]}'


def _items_key(its):
    return hashlib.sha256(json.dumps([(i[0], i[1], i[2], i[4]) for i in its]).encode()).hexdigest()


def run_part(part, recs, log=lambda s: print(s, file=sys.stderr, flush=True)):
    """Train one part on the pool records `recs` and save it; returns the saved dict."""
    t0 = time.time()
    its = DEC.units(recs); hold = DEC.hold_mask([i[0] for i in its])
    tok, low = DEC.lower(); S = DEC.states(tok, low, its)
    if part == 'full':
        top, head = DEC.train(its, S, hold)
        ho = [i for i, h in zip(its, hold) if h]
        out = {'top': {k: v.half() for k, v in top.state_dict().items()}, 'head': {k: v.half() for k, v in head.state_dict().items()},
               'z': DEC.logits(top, head, S, ho).half(), 'idx': torch.tensor(np.flatnonzero(hold))}
    else:
        k = int(part[4:]); fold = np.array([DEC.fold_of(i[5]) for i in its])
        tr = np.flatnonzero(fold != k); ev = np.flatnonzero(fold == k)
        top, head = DEC.train([its[n] for n in tr], S, hold[tr])
        out = {'z': DEC.logits(top, head, S, [its[n] for n in ev]).half(), 'idx': torch.tensor(ev)}
    out['items'] = _items_key(its); out['seconds'] = round(time.time() - t0, 1)
    d = ckpt_dir(); os.makedirs(d, exist_ok=True)
    torch.save(out, f'{d}/{part}.tmp'); os.replace(f'{d}/{part}.tmp', f'{d}/{part}.pt')
    log(f'train_merge {part}: {out["seconds"]} s -> {d}/{part}.pt')
    return out


def load(recs):
    """{part: dict} for every part, training missing parts in-process (slow: ~1 h CPU in total)."""
    d = ckpt_dir(); its = DEC.units(recs); key = _items_key(its); out = {}
    for part in PARTS:
        f = f'{d}/{part}.pt'
        out[part] = torch.load(f, weights_only=False) if os.path.exists(f) else run_part(part, recs)
        assert out[part]['items'] == key, f'checkpoint {f} was trained on other items'
    return out


if __name__ == '__main__':
    torch.manual_seed(0); torch.set_num_threads(int(os.environ.get('THREADS', '2')))
    pool = [json.loads(line) for line in open(POOL)]
    for p in sys.argv[1:]:
        assert p in PARTS, p
        run_part(p, pool)
