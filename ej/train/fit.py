"""The training pipeline behind ``python -m ej.train``: encoder -> teachers -> distilled heads -> fit -> weights directory.

Every step reads the training pool only and memoises its output under <work>/ckpt by a content hash of the code and the pool,
so an interrupted run resumes and a finished step is never repeated; ``fit`` trains whatever is missing in-process. The fit
itself is the released ej 0.0.1 path of the runtime (ej/_runtime/student.py, flags unchanged); this module only points it at the
work directory and at the encoder trained here, then writes the result pickle-free (ej.train.export)."""
import json
import os
import sys
import time

from . import env

LOG = lambda s: print(f'ej.train: {s}', file=sys.stderr, flush=True)  # noqa: E731


def _records():
    with open(env.pool_path()) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_pool(recs):
    """Raise ValueError unless every record is a valid ej record with a gold label per question and a 'source'."""
    from ..records import validate_record
    if not recs:
        raise ValueError('the training pool is empty')
    for i, r in enumerate(recs):
        validate_record(r, i)
        if not isinstance(r.get('id'), str) or not r.get('source'):
            raise ValueError(f'pool record {i}: needs a string "id" and a "source" (the held-out group unit)')
        for qid, q in r['questions'].items():
            g = (r.get('gold') or {}).get(qid)
            if not isinstance(g, dict) or not isinstance(g.get('label'), int) or not 0 <= g['label'] < len(q['options']):
                raise ValueError(f'pool record {r["id"]!r}: question {qid!r} needs gold.label (an option index)')
    return len(recs)


def encoder(variant='w23'):
    """The low-bit encoder for this pool: its checkpoint directory name, trained when missing."""
    from . import lowbit
    name = lowbit.ck_name()
    if not os.path.exists(os.path.join(os.environ['EDGE_CKPT'], name, f'{variant}.pt')):
        LOG(f'training the low-bit encoder {name} on {lowbit.DEV} (GPU strongly recommended)')
        lowbit.run(variant)
    return name


def teachers(parts=None):
    """Decision-encoder teachers (full model + group folds) and the group-honest pair teachers; missing ones are trained."""
    import torch
    import train_merge as TM
    from . import hcf
    torch.manual_seed(0)
    recs = _records()
    for p in parts or TM.PARTS + hcf.pairs():
        done = os.path.exists(os.path.join(TM.ckpt_dir() if p in TM.PARTS else hcf.ckpt_dir(), f'{p}.pt'))
        if done:
            LOG(f'teacher {p}: present')
        elif p in TM.PARTS:
            TM.run_part(p, recs)
        elif p in hcf.pairs():
            hcf.run_pair(p, recs)
        else:
            raise ValueError(f'unknown teacher part {p!r}; known: {TM.PARTS + hcf.pairs()}')


def fit(out, meta=None):
    """Fit the model on the pool with the encoder of this work directory and write the weights directory `out`."""
    import student as S
    import student_lb as LB
    from . import export
    recs = _records()
    LOG(f'pool: {check_pool(recs)} records')
    LB.CK_DIR = encoder()  # the runtime reads its encoder directory from this module constant at fit time
    t0 = time.time()
    state = S.fit(recs)
    LOG(f'fit: {time.time() - t0:.0f} s')
    cfg = export.write(state, out, env.pool_path(), dict(meta or {}, fit_seconds=round(time.time() - t0, 1)))
    LOG(f'weights: {out} (state {cfg["state_key"][:16]}, encoder {cfg["encoder"]})')
    return cfg
