"""Device encoder runtime for prediction: every distinct text of a prediction call is encoded once, in length-sorted
token-budget batches, and the pass returns both the mean-pooled embedding and the token states (token states of short texts
are sliced from the same pass); a process-memory memo is kept unless EDGE_COLD=1. install() routes the experts' encoder
calls through it."""
import numpy as np
import torch

import student_cold as COLD
import student_enc as E
import student_tok as TOK

BUDGET, MAXB = 1024, 32  # tokens per batch (padded), texts per batch (smaller batches ran faster on this CPU: label-free timing)
_ST = {'on': False, 'enc': None, 'memo': {}, 'warm': {}}
_PREV = {}


def _run(enc_fn, tok, texts, max_len, keep=10 ** 9):
    """{text: (pooled (d,), token states (n, d) unnormalised, or None if n > keep)}: unique texts, length-sorted budget batches."""
    ids = tok(texts, truncation=True, max_length=max_len)['input_ids']
    order = sorted(range(len(texts)), key=lambda j: (len(ids[j]), texts[j]))
    out, b = {}, 0
    with torch.no_grad():
        while b < len(order):
            e = b + 1
            while e < len(order) and e - b < MAXB and len(ids[order[e]]) * (e - b + 1) <= BUDGET:
                e += 1
            ch = order[b:e]
            x = tok.pad({'input_ids': [ids[j] for j in ch]}, return_tensors='pt')
            hs = enc_fn(x['input_ids'], x['attention_mask'])
            for r, j in enumerate(ch):
                n = len(ids[j]); h = hs[r, :n]
                out[texts[j]] = (torch.nn.functional.normalize(h.mean(0), dim=-1), h.clone() if n <= keep else None)
            b = e
    return out


def _full(enc):
    return lambda i, m: enc(input_ids=i, attention_mask=m).last_hidden_state


def begin(encoder=None):
    """Enter device mode for one predict call; encoder = (tokenizer, fn(input_ids, mask) -> last-layer-equivalent states, tag)."""
    if encoder is None:
        tok, enc = E._ENC[E.load(TOK.MODEL, TOK.BITS)]
        encoder = (tok, _full(enc), 'full')
    _ST.update(on=True, enc=encoder, memo={} if COLD.cold() else _ST['warm'].setdefault(encoder[2], {}))


def end():
    _ST.update(on=False, memo={})


def _get(texts, max_len, tag):
    memo = _ST['memo']; tok, fn, _ = _ST['enc']
    todo = sorted({t for t in texts if (tag, t) not in memo})
    if todo:  # token states kept only where student_tok can reuse them (memory: the fit-time memo holds the whole pool)
        for t, v in _run(fn, tok, todo, max_len, TOK.MAX_LEN).items():
            memo[(tag, t)] = v
    return [memo[(tag, t)] for t in texts]


def embed(model, bits, texts):
    if not _ST['on']:
        return _PREV['embed'](model, bits, texts)
    if not texts:
        return np.zeros((0, E._ENC[E.load(model, bits)][1].config.hidden_size), np.float32)
    return torch.stack([p for p, _ in _get(list(texts), E.MAX_LEN, 'E')]).numpy().astype(np.float32)


def states(texts):
    if not _ST['on']:
        return _PREV['states'](texts)
    tok = _ST['enc'][0]; uniq = sorted(set(texts))
    n = [len(x) for x in tok(uniq, truncation=True, max_length=E.MAX_LEN)['input_ids']]
    short = [t for t, k in zip(uniq, n) if k <= TOK.MAX_LEN]  # same input ids as a MAX_LEN-truncated pass -> reuse the states
    long_ = [t for t, k in zip(uniq, n) if k > TOK.MAX_LEN]
    hs = dict(zip(short, [h for _, h in _get(short, E.MAX_LEN, 'E')]))
    hs.update(zip(long_, [h for _, h in _get(long_, TOK.MAX_LEN, 'T')]))
    out = {}
    for t, h in hs.items():
        h = torch.nn.functional.normalize(h, dim=-1); k = len(h)
        out[t] = h[3:k - 1].half() if k > 4 else h[1:k - 1].half()  # same slicing as student_tok.states
    return out


def install():
    """Wrap E.embed / TOK.states on top of student_cold's wrappers (idempotent); active only between begin() and end()."""
    COLD.install()
    if not _PREV:
        _PREV['embed'], _PREV['states'] = E.embed, TOK.states
        E.embed, TOK.states = embed, states


def device_fit(shallow_of, encoder_of):
    """Decorator for fit(train): ck = shallow_of(train); when ck is not None the whole fit runs in device mode on encoder_of(ck)
    (every head is fit on the features the device computes; Poor Man's BERT re-fit), and the state carries ck as 'shallow'."""
    def wrap(fn):
        def run(train):
            ck = shallow_of(train)
            if ck is None:
                return fn(train)
            begin(encoder_of(ck))
            try:
                state = fn(train)
            finally:
                end()
            state['shallow'] = ck
            return state
        run.__doc__ = fn.__doc__
        return run
    return wrap


def device(encoder_of):
    """Decorator for predict(state, recs): run it in device mode with encoder_of(state) (None = the full 12-layer encoder)."""
    def wrap(fn):
        def run(state, recs):
            begin(encoder_of(state))
            try:
                return fn(state, recs)
            finally:
                end()
        run.__doc__ = fn.__doc__
        return run
    return wrap
