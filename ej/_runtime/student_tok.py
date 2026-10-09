"""Token-level late interaction for the distilled NLI reader: per-text token states (disk-cached) and MaxSim alignment scores
between hypothesis and state tokens."""
import hashlib
import os

import torch

import student_enc as E

MODEL, BITS, MAX_LEN, NT = 'intfloat/e5-small-v2', 4, 64, 6
_MEM = {}


def _path():
    return f"{E.CACHE}/tok_{MODEL.replace('/', '_')}#q{BITS}#{MAX_LEN}.pt"


def states(texts):
    """{text: (n, d) fp16 normalised token states}; texts carry their e5 prefix ('query: ' / 'passage: ')."""
    if not _MEM and os.path.exists(_path()):
        _MEM.update(torch.load(_path()))
    key = lambda t: hashlib.sha256(t.encode()).hexdigest()[:24]
    todo = sorted({t for t in texts if key(t) not in _MEM}, key=len)
    if todo:
        tok, enc = E._ENC[E.load(MODEL, BITS)]
        with torch.no_grad():
            for b in range(0, len(todo), 64):
                ch = todo[b:b + 64]
                x = tok(ch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
                hs = torch.nn.functional.normalize(enc(**x).last_hidden_state, dim=-1)
                for t, h, m in zip(ch, hs, x['attention_mask']):
                    n = int(m.sum())
                    _MEM[key(t)] = h[3:n - 1].half() if n > 4 else h[1:n - 1].half()  # drop [CLS], 'query'/'passage', ':', [SEP]
        if len(todo) > 200:
            os.makedirs(E.CACHE, exist_ok=True)
            torch.save(_MEM, _path() + f'.{os.getpid()}'); os.replace(_path() + f'.{os.getpid()}', _path())
    return {t: _MEM[key(t)] for t in set(texts)}


def maxsim(pairs, P, H):
    """(n, NT) late-interaction scalars for [(premise text, hypothesis text)] given token-state dicts P, H."""
    out = torch.zeros(len(pairs), NT)
    for b in range(0, len(pairs), 2048):
        ch = pairs[b:b + 2048]
        Lp, Lh = max(len(P[p]) for p, _ in ch), max(len(H[h]) for _, h in ch)
        A, B = torch.zeros(len(ch), Lp, P[ch[0][0]].shape[1]), torch.zeros(len(ch), Lh, P[ch[0][0]].shape[1])
        ma, mb = torch.zeros(len(ch), Lp, dtype=torch.bool), torch.zeros(len(ch), Lh, dtype=torch.bool)
        for r, (p, h) in enumerate(ch):
            A[r, :len(P[p])], B[r, :len(H[h])] = P[p].float(), H[h].float(); ma[r, :len(P[p])], mb[r, :len(H[h])] = True, True
        S = (B @ A.transpose(1, 2)).masked_fill(~ma[:, None, :], -2.0)  # (n, Lh, Lp)
        m = S.amax(-1)  # support of each hypothesis token
        nb = mb.sum(1).clamp(min=1).float()
        mean = (m * mb).sum(1) / nb
        mn = m.masked_fill(~mb, 2.0).amin(1)
        soft = -0.1 * torch.logsumexp((-m / 0.1).masked_fill(~mb, -1e9), 1) + 0.1 * nb.log()
        mx = m.masked_fill(~mb, -2.0).amax(1)
        rev = (S.masked_fill(~mb[:, :, None], -2.0).amax(1) * ma).sum(1) / ma.sum(1).clamp(min=1)
        out[b:b + len(ch)] = torch.stack([mean, mn, soft, mx, rev, nb.log()], 1)
    return out
