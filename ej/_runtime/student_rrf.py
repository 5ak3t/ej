"""Fit routine of the slot-free relational reader (student_rr)."""
import sys

import numpy as np
import torch

import student_rr as RR


def rr_fit(items, data, Ho, y, hold, gid):
    """Slot-free relational reader (student_rr): FULL reader (L2 by the iid held-out slice; seen-slot expert) and ZERO-SHOT
    VERIFIER (L2 by leave-one-workflow-out NLL; held-out + LOGO logits, so the pool trusts it on new tasks only as far as it
    transfers). Both read value-free relational facts of JSON states embedded once per schema; silent on plain text."""
    d = data[1].shape[-1]
    fb = RR.bank(items); mu = RR.kind_rates(fb).half().float()
    x = RR.inputs(items, fb, data[0][:, -d:], data[1], Ho, data[2], data[5], mu)
    xv, M = RR.verify(x), data[2]
    js = torch.tensor([i['json'] for i in items]); tr, ho = torch.tensor(~hold) & js, torch.tensor(hold)
    nll = lambda th, xx: torch.nn.functional.cross_entropy(RR.logits(th, RR.sub(xx, ho & js)), y[ho & js]).item()
    fits = {l2: RR.train(RR.sub(x, tr), y[tr], l2) for l2 in RR.L2_GRID}
    l2f = min(fits, key=lambda l2: nll(fits[l2], x))
    l2v, rep = RR.logo_l2(xv, y, list(gid), js)
    th_v = RR.train(RR.sub(xv, tr), y[tr], l2v)
    out = {'zf_ho': RR.logits(fits[l2f], RR.sub(x, ho)).masked_fill(~M[ho], 0), 'zv_ho': RR.logits(th_v, RR.sub(xv, ho)).masked_fill(~M[ho], 0),
           'logo': {}, 'mu': mu, 'rep': {'mu': [round(v, 3) for v in mu.tolist()], 'l2_full': l2f, 'ho_full': round(nll(fits[l2f], x), 4), 'l2_ver': l2v, 'logo_ver': rep}}
    for g in sorted({g for g, j in zip(gid, js.tolist()) if j}):
        o = torch.tensor(np.asarray(gid) == g)
        out['logo'][g] = RR.logits(RR.train(RR.sub(xv, js & ~o), y[js & ~o], l2v), RR.sub(xv, o)).masked_fill(~M[o], 0)
    out['full'], out['ver'] = RR.train(RR.sub(x, js), y[js], l2f), RR.train(RR.sub(xv, js), y[js], l2v)
    print('rr', out['rep'], file=sys.stderr)
    return out
