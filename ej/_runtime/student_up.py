"""Upstream expert training: deep, rich and field-attention experts on the training pool with held-out and leave-one-group-out
logits, memoised as a checkpoint keyed by a hash of its inputs. Fit-time code."""
import hashlib
import os
import sys

import numpy as np
import torch

import student_attn as A
import student_deep as D
import student_rich as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ('student_up.py', 'student_deep.py', 'student_rich.py', 'student_attn.py', 'student_read.py')


def _sub(data, sel):
    return tuple(t[sel] for t in data)


def _key(data, fb, y, hold, gid):
    h = hashlib.sha256()
    for f in SRC:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    for t in tuple(data) + tuple(fb) + (y,):
        h.update(t.detach().contiguous().numpy().tobytes())
    h.update(np.asarray(hold).tobytes()); h.update('\x1f'.join(gid).encode())
    return h.hexdigest()[:16]


def _compute(data, fb, y, hold, gid):
    tr, ho = torch.tensor(~hold), torch.tensor(hold)
    th = D.train(_sub(data, tr), y[tr])
    out = {'zd_tr': D.logits(th, _sub(data, tr)), 'zd_ho': D.logits(th, _sub(data, ho)),
           'zc_ho': D.centred(th, _sub(data, ho), data[0][tr].mean(0))}
    net, ep = R.train(_sub(data, tr), y[tr], val=(_sub(data, ho), y[ho]))
    out['zr_ho'], out['ep'] = R.logits(net, _sub(data, ho)), ep
    na, ep_a = A.train(_sub(data, tr), A.sub(fb, tr), y[tr], val=(_sub(data, ho), A.sub(fb, ho), y[ho]))
    out['za_ho'], out['ep_a'] = A.logits(na, _sub(data, ho), A.sub(fb, ho)), ep_a
    out['logo'] = {}
    for g in sorted(set(gid)):  # zero-shot cross-fitting: experts trained without a whole source/workflow
        o = torch.tensor(np.asarray(gid) == g)
        thg = D.train(_sub(data, ~o), y[~o])
        out['logo'][g] = (D.logits(thg, _sub(data, o)), D.centred(thg, _sub(data, o), data[0][~o].mean(0)),
                          R.logits(R.train(_sub(data, ~o), y[~o], epochs=ep)[0], _sub(data, o)))
    out['deep'] = D.train(data, y)
    out['rich'], _ = R.train(data, y, epochs=ep)
    out['attn'], _ = A.train(data, fb, y, epochs=ep_a)
    return out


def upstream(data, fb, y, hold, gid):
    path = f'{ROOT}/up-{_key(data, fb, y, hold, gid)}.pt'
    if os.path.exists(path):
        print('upstream: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    out = _compute(data, fb, y, hold, gid)
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    print('upstream: computed ->', path, file=sys.stderr)
    return out
