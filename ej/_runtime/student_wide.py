"""Wide expert (Wide & Deep, Cheng et al. 2016, arXiv:1606.07792): sparse crosses of state tokens x option slots that memorise
seen decision rules and stay silent on unseen slots, fitted as a convex residual on top of the deep scorer."""
import hashlib

import numpy as np
import torch

import student_feat as F

BIAS = '__bias__'
MIN_COUNT = 2
L2_GRID = (3e-4, 3e-3, 3e-2)


def slot(item, j):
    return hashlib.sha256(f"{item['instr']}\x1f{item['opts'][j]}".encode()).hexdigest()[:16]


def _toks(item, cache):
    if item['state'] not in cache:
        cache[item['state']] = [BIAS] + F.state_tokens(item['state'])
    return cache[item['state']]


def vocab(items):
    """Crosses (slot, token) seen in >= MIN_COUNT training questions -> column index."""
    cnt, cache = {}, {}
    for i in items:
        toks = _toks(i, cache)
        for j in range(len(i['opts'])):
            s = slot(i, j)
            for t in toks:
                cnt[(s, t)] = cnt.get((s, t), 0) + 1
    keys = sorted(k for k, c in cnt.items() if c >= MIN_COUNT)
    return {k: n for n, k in enumerate(keys)}


def bags(voc, items, K=None):
    """EmbeddingBag inputs over all (question, option) pairs in row-major (N, K) order + a seen-slot mask (N, K)."""
    K = K or max(len(i["opts"]) for i in items)
    idx, off, wts, cache = [], [], [], {}
    seen = np.zeros((len(items), K), dtype=bool)
    slots = {s for s, _ in voc}
    for n, i in enumerate(items):
        toks = _toks(i, cache)
        w = 1.0 / np.sqrt(len(toks))
        for j in range(K):
            off.append(len(idx))
            if j < len(i['opts']):
                s = slot(i, j)
                seen[n, j] = s in slots
                cols = [voc[(s, t)] for t in toks if (s, t) in voc]
                idx.extend(cols); wts.extend([w] * len(cols))
    return (torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long),
            torch.tensor(wts, dtype=torch.float32), torch.tensor(seen))


def logits(w, bag, shape):
    idx, off, wts, _ = bag
    if len(idx) == 0:
        return torch.zeros(shape)
    z = torch.nn.functional.embedding_bag(idx, w[:, None], off, mode='sum', per_sample_weights=wts)
    return z.reshape(shape)


def fit(nvoc, bag, offset, M, y, l2):
    """Convex: min CE(offset + wide) + l2 * |w|^2 over the cross weights (offset = frozen deep logits)."""
    w = torch.zeros(nvoc, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = (offset + logits(w, bag, offset.shape)).masked_fill(~M, -1e9)
        loss = torch.nn.functional.cross_entropy(z, y) + l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def size_mb(voc):
    """fp16 weight + 8-byte hashed key per kept cross."""
    return len(voc) * 10 / 2 ** 20
