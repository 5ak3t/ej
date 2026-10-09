"""Distilled NLI reader: a small pair head on the shared encoder's embeddings that predicts the entailment logits of an NLI
cross-encoder (the cross-encoder is a fit-time teacher only). Builds per-option hypothesis texts, trains the head and
computes its device features."""
import hashlib

import numpy as np
import torch

import student_enc as E
import student_feat as F
import student_nli as NLI
import student_tok as TOK

MODEL, BITS = 'intfloat/e5-small-v2', 4
HID, LR, WD, BATCH, MAX_EP, PATIENCE, TEMP = 256, 1e-3, 1e-4, 256, 30, 3, 0.05
USE_TOK = True  # token-level MaxSim scalars (student_tok) join the pair features
N_AUG_OPT, N_AUG_NN, NK, MAXC = 2, 2, 5, 12  # hypothesis kinds: option, yes/no statement, instr+option, aug option, aug state


def _salt(s, salt='', mod=10):
    return int(hashlib.sha256((salt + s).encode()).hexdigest()[:8], 16) % mod


def item_hyps(i):
    """[(hypothesis text, kind)] the NLI expert reads for one plain-text question (student_nli.features order)."""
    return [(o, 0) for o in i['opts']] + ([(i['instr'], 1)] if i['type'] == 1 else []) + [(h, 2) for h in NLI.io_hyps(i)]


def _prefix(text, kind):
    return f'query: {text}' if kind in (1, 4) else f'passage: {text}'


def _emb(texts):
    keys = sorted(set(texts)); pos = {t: k for k, t in enumerate(keys)}
    return torch.tensor(E.embed(MODEL, BITS, keys)), pos


class Bank:
    """Embedding tables of premises (state + sentence chunks) and hypotheses; pairs index into them."""

    def __init__(self, states, hyps):
        self.S, self.ps = _emb([f'query: {s}' for s in states])
        ch = {s: F.chunks(s)[:MAXC] for s in states}
        Ec, pc = _emb([f'query: {c}' for cs in ch.values() for c in cs])
        self.C = torch.zeros(len(self.ps), MAXC, self.S.shape[1]); self.CM = torch.zeros(len(self.ps), MAXC, dtype=torch.bool)
        for s, cs in ch.items():
            r = self.ps[f'query: {s}']; self.C[r, :len(cs)] = Ec[[pc[f'query: {c}'] for c in cs]]; self.CM[r, :len(cs)] = True
        self.H, self.ph = _emb([_prefix(t, k) for t, k in hyps])
        self.Pt, self.Ht = (TOK.states(list(self.ps)), TOK.states(list(self.ph))) if USE_TOK else ({}, {})  # token states

    def x(self, pairs):
        """(n, 4d + 2 + TOK.NT + NK) pair features for [(state, hyp, kind)]."""
        out = []
        for b in range(0, len(pairs), 4096):
            ch = pairs[b:b + 4096]
            pi = torch.tensor([self.ps[f'query: {p}'] for p, _, _ in ch]); hi = torch.tensor([self.ph[_prefix(h, k)] for _, h, k in ch])
            s, h, C, CM = self.S[pi], self.H[hi], self.C[pi], self.CM[pi]
            sim = (C @ h[:, :, None]).squeeze(-1).masked_fill(~CM, -1e9)
            a = torch.nn.functional.normalize((torch.softmax(sim / TEMP, -1)[..., None] * C).sum(1), dim=-1)
            kind = torch.nn.functional.one_hot(torch.tensor([k for _, _, k in ch]), NK).float()
            li = TOK.maxsim([(f'query: {p}', _prefix(h, k)) for p, h, k in ch], self.Pt, self.Ht) if USE_TOK else torch.zeros(len(ch), 0)
            out.append(torch.cat([s * h, a * h, (s - h).abs(), h, (s * h).sum(-1, keepdim=True), sim.amax(-1, keepdim=True), li, kind], 1))
        return torch.cat(out)


def transfer_set(items):
    """[(state, hyp, kind, premise group, hypothesis group)] over plain-text pool states: expert pairs + augmentation."""
    plain = [i for i in items if not i['json'] and i['state']]
    out = {(i['state'], h, k): (i['group'], i['group']) for i in plain for h, k in item_hyps(i)}
    opt_g = sorted({(o, i['group']) for i in plain for o in i['opts']})
    states = sorted({i['state']: i['group'] for i in plain}.items())
    own = {}
    for i in plain:
        own.setdefault(i['state'], set()).update(i['opts'])
    S, ps = _emb([f'query: {s}' for s, _ in states])
    S = S[[ps[f'query: {s}'] for s, _ in states]]
    for b in range(0, len(states), 2048):
        sim = S[b:b + 2048] @ S.T
        sim[torch.arange(sim.shape[0]), torch.arange(b, b + sim.shape[0])] = -2  # not itself
        nn = sim.topk(N_AUG_NN + 4, -1).indices
        for r in range(sim.shape[0]):
            s, g = states[b + r]
            rng = np.random.default_rng(_salt(s, 'aug', 2 ** 31))
            cand = [opt_g[j] for j in rng.choice(len(opt_g), 4 * N_AUG_OPT, replace=False) if opt_g[j][0] not in own[s]]
            for o, og in cand[:N_AUG_OPT]:
                out.setdefault((s, o, 3), (g, og))
            nb = [states[j] for j in nn[r].tolist() if states[j][0] != s][:N_AUG_NN]
            for t, tg in nb:
                out.setdefault((s, t, 4), (g, tg))
    return [(k[0], k[1], k[2], v[0], v[1]) for k, v in sorted(out.items())]


class Head(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lin = torch.nn.Linear(n, 3)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n, HID), torch.nn.GELU(), torch.nn.Linear(HID, 3))

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


def _loss(z, t):
    return ((z - t) ** 2).mean() + (((z[:, 1] - z[:, 0]) - (t[:, 1] - t[:, 0])) ** 2).mean()


def train(X, T, val, seed=0):
    """Adam on the distillation loss; early stopping on the `val` rows (patience PATIENCE); weights stored fp16."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = np.flatnonzero(~val); va = torch.tensor(np.flatnonzero(val))
    net = Head(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best, state, bad = float('inf'), None, 0
    for _ in range(MAX_EP):
        net.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), BATCH):
            i = torch.tensor(order[b:b + BATCH])
            opt.zero_grad(); _loss(net(X[i].float()), T[i]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = _loss(predict(net, X[va]), T[va]).item()
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(p.half().float())
    return net, round(best, 4)


def predict(net, X):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[b:b + 8192].float()) for b in range(0, len(X), 8192)])


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def features(items, zd):
    """(N, K, NLI.NF) NLI-expert features (student_nli.features, text view) from predicted logits zd[(state, hyp, kind)]."""
    K = max(len(i['opts']) for i in items)
    out = torch.zeros(len(items), K, NLI.NF)
    c = lambda v: v - v.mean()
    for n, i in enumerate(items):
        if i['json'] or not i['state']:
            continue
        k, st = len(i['opts']), i['state']
        pol = torch.tensor([-1.0, 1.0]) if i['type'] == 1 and k == 2 else torch.zeros(k)
        Zo = torch.stack([zd[(st, o, 0)] for o in i['opts']])
        t = NLI._m(zd[(st, i['instr'], 1)]) if i['type'] == 1 else torch.tensor(0.0)
        out[n, :k, 0:4] = torch.stack([c(NLI._m(Zo)), c(torch.log_softmax(Zo, -1)[:, 1]), pol * t, pol * torch.tanh(t / 4)], 1)
        if NLI.io_hyps(i):
            zio = torch.stack([zd[(st, h, 2)] for h in NLI.io_hyps(i)])
            out[n, :k, 7:9] = torch.stack([c(NLI._m(zio)), c(torch.log_softmax(zio, -1)[:, 1])], 1)
    return out


def device_pairs(items):
    """The (state, hyp, kind) pairs the device reads for `items` (plain-text states only)."""
    return sorted({(i['state'], h, k) for i in items if not i['json'] and i['state'] for h, k in item_hyps(i)})
