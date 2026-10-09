"""State-once read: the state is encoded once per record (the whole state plus one unit per JSON field), and every question
reads it through a fixed cosine attention over the field units, giving the query vectors the experts score against."""
import torch

import student_enc as E
import student_feat as F

MODEL, BITS = 'intfloat/e5-small-v2', 4
TEMP = 0.05
R = 3  # blocks of u_q: state, attended fields, question


def _emb(texts):
    keys = sorted(set(texts)); pos = {t: k for k, t in enumerate(keys)}
    return torch.tensor(E.embed(MODEL, BITS, keys)), pos


def fields(items):
    """{state: (n_f, d) field-unit embeddings}; each distinct state is encoded once."""
    ch = {i['state']: F.chunks(i['state']) for i in items}
    Ec, pc = _emb([f'query: {c}' for cs in ch.values() for c in cs])
    return {s: Ec[[pc[f'query: {c}'] for c in cs]] for s, cs in ch.items()}


def read(items, Ef):
    """(N, R*d) question reads u_q = [s ; a_q ; q] and (N, d) whole-state embeddings s."""
    Hs, ps = _emb([f"query: {i['state']}" for i in items])
    Q, pq = _emb([f"query: {i['instr']}" for i in items])
    s = Hs[[ps[f"query: {i['state']}"] for i in items]]
    q = Q[[pq[f"query: {i['instr']}"] for i in items]]
    a = torch.zeros_like(s)
    for n, i in enumerate(items):
        C = Ef[i['state']]
        a[n] = torch.softmax((C @ q[n]) / TEMP, 0) @ C
    return torch.cat([s, torch.nn.functional.normalize(a, dim=-1), q], -1), s


def fold(U, d):
    """Sum of the d-blocks of U (the read as one d-vector)."""
    return U.view(*U.shape[:-1], U.shape[-1] // d, d).sum(-2)


def tile(V, U):
    """Repeat option vectors to the block width of U (for experts with element-wise u * v features)."""
    return V.repeat(*([1] * (V.dim() - 1)), U.shape[-1] // V.shape[-1])


def late(items, Ef, V, Ho):
    """(N, K, 2) late interaction: max over the state's field units of cos(field, option) and cos(field, hypothesis)."""
    out = torch.zeros(V.shape[0], V.shape[1], 2)
    for n, i in enumerate(items):
        C = Ef[i['state']]
        k = len(i['opts'])
        out[n, :k, 0] = (C @ V[n, :k].T).amax(0); out[n, :k, 1] = (C @ Ho[n, :k].T).amax(0)
    return out

