"""Distilled decision-encoder expert: a rich scorer (student_rich form) trained to reproduce the out-of-fold distributions of
a fine-tuned decision encoder, so the device needs no second encoder pass. Fit-time code; at prediction time the scorer's
weights are part of the state."""
import hashlib
import os
import sys

import numpy as np
import torch

import student_rich as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ('student_dd.py', 'student_rich.py', 'student_read.py')
ALPHA = 0.5


def _sub(data, sel):
    return tuple(t[sel] for t in data)


def _loss(z, y, pt):
    lp = torch.log_softmax(z, -1)
    return (1 - ALPHA) * torch.nn.functional.nll_loss(lp, y) - ALPHA * (pt * lp.clamp(min=-1e4)).sum(-1).mean()


def train(data, y, pt, epochs=None, val=None, seed=0):
    """student_rich.Scorer on the distillation loss (AdamW, student_rich hyperparameters); early stopping on `val` = (data, y, pt)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = R.Scorer(data[0].shape[1], data[3].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=R.LR, weight_decay=R.WD)
    best, best_ep, state, bad = float('inf'), 0, None, 0
    for ep in range(1, (epochs or R.MAX_EP) + 1):
        net.train()
        order = rng.permutation(len(y))
        for b in range(0, len(y), R.BATCH):
            i = torch.tensor(order[b:b + R.BATCH])
            opt.zero_grad(); _loss(net(*[t[i] for t in data[:5]]), y[i], pt[i]).backward(); opt.step()
        if val is not None:
            v = _loss(R.logits(net, val[0]), val[1], val[2]).item()
            if v < best - 1e-4:
                best, best_ep, bad, state = v, ep, 0, {k: x.clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= R.PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(p.half().float())
    return net, best_ep or (epochs or R.MAX_EP)


def _key(data, y, pt, hold, gid):
    h = hashlib.sha256()
    for f in SRC:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    for t in tuple(data) + (y, pt):
        h.update(t.detach().contiguous().numpy().tobytes())
    h.update(np.asarray(hold).tobytes()); h.update('\x1f'.join(gid).encode())
    return h.hexdigest()[:16]


def fit(data, y, z_oof, hold, gid):
    """{'ho': held-out logits, 'logo': {group: logits}, 'full': net, 'ep': epochs}; z_oof = teacher out-of-fold logits (-1e9 pads)."""
    pt = torch.softmax(z_oof.float().masked_fill(~data[2], -1e9), -1).masked_fill(~data[2], 0)
    path = f'{ROOT}/dd-{_key(data, y, pt, hold, gid)}.pt'
    if os.path.exists(path):
        print('dd: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    tr, ho = torch.tensor(~hold), torch.tensor(hold)
    net, ep = train(_sub(data, tr), y[tr], pt[tr], val=(_sub(data, ho), y[ho], pt[ho]))
    out = {'ho': R.logits(net, _sub(data, ho)), 'ep': ep, 'logo': {}}
    for g in sorted(set(gid)):
        o = torch.tensor(np.asarray(gid) == g)
        out['logo'][g] = R.logits(train(_sub(data, ~o), y[~o], pt[~o], epochs=ep)[0], _sub(data, o))
    out['full'] = train(data, y, pt, epochs=ep)[0]
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    print('dd: computed ->', path, 'epochs', ep, file=sys.stderr)
    return out
