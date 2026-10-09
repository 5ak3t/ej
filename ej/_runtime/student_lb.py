"""Low-bit shared encoder and int8 heads: loads the compact encoder checkpoint (cached per process in _CK), serves it to every
encoder consumer, compacts the fitted heads to int8 (per-row matrices, hashed cross weights) and computes the counted model
size."""
import copy
import hashlib
import os

import torch

import student_lbq as Q

CK_DIR = 'lowbit-9c15e5f8f8e3b927'  # QAT attempt 3 (train_lowbit.py as committed); attempts 1-2: lowbit-d64806ae4e5a51bf / -337c1f4fe32d3af3 (weights deleted)
QB = 64
_CK = {}


def ck(variant):
    """fit side: the device-encoder record of `variant` (None when '' / 0, i.e. the full 4-bit encoder)."""
    if not variant:
        return None
    assert variant in Q.VARIANTS, f'USE_LB must be "" or one of {sorted(Q.VARIANTS)}'
    return {'lb': variant, 'dir': CK_DIR}


def checkpoint(c):
    """The compact checkpoint of record c, loaded once per process from $EDGE_CKPT/<dir>/<variant>.pt."""
    path = os.path.join(os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt')), c['dir'], f"{c['lb']}.pt")
    if path not in _CK:
        _CK.clear(); _CK[path] = torch.load(path, map_location='cpu', weights_only=False)  # noqa: E702
    return _CK[path]


def encoder(c, shallow=None):
    """(tokenizer, fn, tag) for student_fast.begin; a non-low-bit record goes to shallow.encoder (USE_SHALLOW path)."""
    return Q.encoder(checkpoint(c)) if 'lb' in c else shallow.encoder(c)


def predict_encoder(st, shallow):
    """predict side: the device encoder recorded in the fitted state (None = full encoder, exactly as before)."""
    c = st.get('shallow')
    return encoder(c, shallow) if c else None


def _row_int8(w):
    s = (w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127).half().float()
    return torch.round(w / s).clamp(-127, 127) * s


def net(m):
    """Copy of a head module with int8 matrices (per-row fp16 scale; Jacob et al. 2018) and fp16 vectors."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(_row_int8(p.reshape(p.shape[0], -1)).reshape(p.shape) if p.dim() >= 2 else p.half().float())
    return m


def net_mb(m):
    if m is None:
        return 0.0
    return sum(p.numel() + 2 * p.shape[0] if p.dim() >= 2 else 2 * p.numel() for p in m.parameters()) / 2 ** 20


def cross(w):
    """int8 cross weights with an fp16 scale per block of QB (in key order)."""
    n = w.numel(); pad = (-n) % QB
    g = torch.cat([w.float(), torch.zeros(pad)]).reshape(-1, QB)
    s = (g.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127).half().float()
    return (torch.round(g / s).clamp(-127, 127) * s).reshape(-1)[:n]


def cross_mb(voc):
    """Device key per cross = 40-bit hash of (slot, token) (5 B; checked collision-free here) + int8 weight + block scales; 8 B per slot."""
    keys = {hashlib.sha256(repr(k).encode()).digest()[:5] for k in voc}
    assert len(keys) == len(voc), '40-bit key collision'
    return (len(voc) * (5 + 1 + 2 / QB) + 8 * len({s for s, _ in voc})) / 2 ** 20


def compact(state):
    """Quantise the heads of a fitted state in place (returns it); marks it so size_mb uses the compact accounting."""
    for k in ('rich', 'attn', 'dd', 'dn'):
        if state.get(k) is not None:
            state[k] = net(state[k])
    state['w'], state['rel'] = cross(state['w']), cross(state['rel'])
    state['compact'] = True
    return state


def compacted(variant):
    """Decorator for student.fit: with a low-bit variant the fitted state's heads are compacted; '' returns fit unchanged."""
    if not variant:
        return lambda fn: fn

    def wrap(fn):
        def run(train):
            return compact(fn(train))
        run.__doc__ = fn.__doc__
        return run
    return wrap


def parts(state, D, RR):
    """MB per part of a compacted state: encoder, vocab, heads (D = student_deep, RR = student_rr for parameter counts)."""
    sp = Q.size_parts(checkpoint(state['shallow']))
    heads = (sum(net_mb(state.get(k)) for k in ('rich', 'attn', 'dd', 'dn')) + cross_mb(state['voc']) + cross_mb(state['rvoc'])
             + (D.n_params(state['deep']) + RR.n_params(state['rr']) + RR.n_params(state['rv']) + state['nli'].numel()
                + state['ubar'].numel()) * 2 / 2 ** 20 + state['tf'].size_mb())
    return {'encoder': sp['encoder'], 'vocab': sp['vocab'], 'heads': heads}


def size_mb(state, D, RR):
    return sum(parts(state, D, RR).values())


def _test():
    """Off-switch passthrough, int8 compaction error bounds and accounting."""
    assert ck('') is None and ck(0) is None and predict_encoder({}, None) is None
    f = lambda t: t
    assert compacted('')(f) is f
    w = torch.randn(300)
    assert (cross(w) - w).abs().max() < w.abs().max() / 127
    m = net(torch.nn.Linear(8, 4))
    assert abs(net_mb(m) - (32 + 8 + 8) / 2 ** 20) < 1e-12
    print('student_lb ok')


if __name__ == '__main__':
    _test()
