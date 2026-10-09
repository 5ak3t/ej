"""Cold-run switch: when the environment variable EDGE_COLD is '1' (read at call time), every prediction-time cache (sentence
embeddings, token states) is bypassed, so latency can be measured without warm caches. install() wires the switch into the
encoder functions."""
import os

import numpy as np
import torch

import student_enc as E
import student_tok as TOK

_ORIG = {}


def cold():
    """True when the uncached (cold) device path is requested; read at call time."""
    return os.environ.get('EDGE_COLD') == '1'


def _encode(key, texts):
    tok, enc = E._ENC[key]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), E.BATCH):
            b = tok(texts[i:i + E.BATCH], padding=True, truncation=True, max_length=E.MAX_LEN, return_tensors='pt')
            hs = enc(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((hs * m).sum(1) / m.sum(1), dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def embed(model, bits, texts):
    if not cold():
        return _ORIG['embed'](model, bits, texts)
    key = E.load(model, bits)
    uniq = list(dict.fromkeys(texts)); pos = {t: k for k, t in enumerate(uniq)}
    if not uniq:
        return np.zeros((0, E._ENC[key][1].config.hidden_size), np.float32)
    return _encode(key, uniq)[[pos[t] for t in texts]]


def states(texts):
    if not cold():
        return _ORIG['states'](texts)
    tok, enc = E._ENC[E.load(TOK.MODEL, TOK.BITS)]
    todo, out = sorted(set(texts), key=len), {}
    with torch.no_grad():
        for b in range(0, len(todo), 64):
            ch = todo[b:b + 64]
            x = tok(ch, padding=True, truncation=True, max_length=TOK.MAX_LEN, return_tensors='pt')
            hs = torch.nn.functional.normalize(enc(**x).last_hidden_state, dim=-1)
            for t, h, m in zip(ch, hs, x['attention_mask']):
                n = int(m.sum())
                out[t] = h[3:n - 1].half() if n > 4 else h[1:n - 1].half()  # same slicing as student_tok.states
    return out


def install():
    """Wrap the two predict-time caches (idempotent). Callers use E.embed / TOK.states by attribute, so the wrappers apply."""
    if not _ORIG:
        _ORIG['embed'], _ORIG['states'] = E.embed, TOK.states
        E.embed, TOK.states = embed, states


def _test():
    """Cold and cached paths agree (same encoder, same slicing)."""
    install()
    t = ['query: the invoice total is 12', 'passage: refund']
    a = E.embed('intfloat/e5-small-v2', 4, t)
    os.environ['EDGE_COLD'] = '1'
    try:
        b = E.embed('intfloat/e5-small-v2', 4, t); s = TOK.states(t)
    finally:
        os.environ.pop('EDGE_COLD')
    s0 = TOK.states(t)
    assert np.abs(a - b).max() < 1e-4 and all((s[k].float() - s0[k].float()).abs().max() < 1e-2 for k in t)
    print('cold ok', float(np.abs(a - b).max()))


if __name__ == '__main__':
    _test()
