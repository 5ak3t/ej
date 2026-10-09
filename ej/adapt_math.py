"""The mathematics of workflow adaptation (ej.adapt): per-(question id, option key) logit offsets on top of the zero-shot output.

For a question q of a workflow with option keys O_q, the adapted prediction is

    p_b(y | x) = softmax_y(log p0(y | x) + b[q, y]),        p0 = the zero-shot distribution of ej

and the offsets b are fitted on the workflow's own labelled records (the OPTION TILT): b ~ N(0, sb^2) per (q, y), maximum a
posteriori on the labelled questions (L-BFGS, ``fit_tilt``). It learns the workflow's label prior, which a zero-shot model
cannot know. The model is never fine-tuned."""
import math

import torch

DT = torch.float64


def usable(lp):
    """Boolean mask of rows whose log-probabilities hold no NaN / +inf and at least one finite entry."""
    return ~(torch.isnan(lp) | (lp == math.inf)).any(-1) & torch.isfinite(lp).any(-1)


def index(rows, K):
    """(names, idx (n, K)): the sorted (qid, key) names of rows and each entry's name index (len(names) = padding)."""
    names = sorted({(q, k) for q, keys in rows for k in keys})
    pos = {nm: i for i, nm in enumerate(names)}
    idx = torch.full((len(rows), K), len(names), dtype=torch.long)
    for n, (q, keys) in enumerate(rows):
        idx[n, :len(keys)] = torch.tensor([pos[(q, k)] for k in keys])
    return names, idx


def offsets(rows, b, K):
    """(n, K) offsets of rows from b {(qid, key): value} (0 for unknown names)."""
    off = torch.zeros(len(rows), K, dtype=DT)
    for n, (q, keys) in enumerate(rows):
        for j, k in enumerate(keys):
            off[n, j] = b.get((q, k), 0.0)
    return off


def fit_tilt(rows, y, lp, sb=0.5, iters=60):
    """Option tilt: MAP of b ~ N(0, sb^2) per (qid, key) given labelled rows, labels y (option index) and log-probs lp (n, K)."""
    K = lp.shape[1]
    names, idx = index(rows, K)
    valid = torch.isfinite(lp)
    base = lp.masked_fill(~valid, 0.0)
    y = torch.as_tensor(y, dtype=torch.long)
    b = torch.zeros(len(names) + 1, dtype=DT, requires_grad=True)  # last slot: padding (never valid, no prior)
    opt = torch.optim.LBFGS([b], max_iter=iters, tolerance_grad=1e-7, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = (base + b[idx]).masked_fill(~valid, -math.inf)
        loss = -torch.log_softmax(z, -1)[torch.arange(len(y)), y].sum() + 0.5 * (b[:-1] ** 2).sum() / sb ** 2
        loss.backward()
        return loss
    opt.step(closure)
    return {nm: float(b[i]) for i, nm in enumerate(names)}
