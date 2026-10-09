"""Distilled heads: builds, from the training pool only, the two memoised checkpoints fit() loads by content hash (when they
are missing, fit() trains them in-process, identically):
  <EDGE_CKPT>/dn-<hash>.pt  distilled NLI pair heads (runtime student_dnfit.heads; teacher = the cross-encoder
                            cross-encoder/nli-deberta-v3-xsmall on an augmented transfer set of pool texts; full, no-held-out and
                            leave-one-group-out heads)
  <EDGE_CKPT>/dd-<hash>.pt  distilled decision encoder (runtime student_dd.fit; teacher = the out-of-fold logits of the
                            decision-encoder teachers, runtime train_merge)
Needs the decision-encoder teachers (``python -m ej.train teachers``; trained here when missing). CPU."""
import json
import sys
import time

import torch


def run(log=lambda s: print(s, file=sys.stderr, flush=True)):
    """Build both checkpoints for the pool of the current training environment (ej.train.env.setup)."""
    import student as S
    import student_dd as DD
    import student_dec as DEC
    import student_dnfit as DNF
    import student_feat as F
    import train_merge as TM
    t0 = time.time()
    torch.manual_seed(0)
    with open(TM.POOL) as f:
        pool = [json.loads(line) for line in f]
    items = S.questions(pool)
    hold = DEC.hold_mask([i['state'] for i in items])
    DNF.heads(items, hold)
    log(f'distil: NLI pair heads done ({time.time() - t0:.0f} s)')
    texts = sorted({i['state'] for i in items} | {i['instr'] for i in items} | {o for i in items for o in i['opts']})
    data, _, _ = S.tensors(items, F.Tfidf(texts))
    y, K = torch.tensor([i['y'] for i in items]), data[2].shape[1]
    ck = TM.load(pool)
    z = torch.full((len(items), K), -1e9)
    for k in range(len(DEC.FOLDS)):
        z[ck[f'fold{k}']['idx']] = S._pad(ck[f'fold{k}']['z'], K)
    DD.fit(data, y, z, hold, [i['group'] for i in items])
    log(f'distil: distilled decision encoder done ({time.time() - t0:.0f} s)')
