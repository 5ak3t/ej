"""Field-attention expert: learned key-aware attention over the field units of a JSON state. Several heads score how relevant
each field is to a question (a learned bilinear term on top of a fixed cosine prior), and the attended field summaries are
scored against each option."""
import numpy as np
import torch

import student_read as RD
import student_rich as R

HEADS, RANK = 2, 32


def bank(Ef, items):
    """(field bank (n_states, L, d), mask (n_states, L), per-item state index (N,))."""
    states = sorted({i['state'] for i in items}); sp = {s: k for k, s in enumerate(states)}
    L, d = max(len(Ef[s]) for s in states), Ef[states[0]].shape[1]
    E = torch.zeros(len(states), L, d); EM = torch.zeros(len(states), L, dtype=torch.bool)
    for s, k in sp.items():
        E[k, :len(Ef[s])] = Ef[s]; EM[k, :len(Ef[s])] = True
    return E, EM, torch.tensor([sp[i['state']] for i in items])


class FieldScorer(torch.nn.Module):
    """H learned field-attention heads in front of the rich scorer."""

    def __init__(self, du, d, ns):
        super().__init__()
        self.d = d
        self.Wq = torch.nn.Linear(d, HEADS * RANK, bias=False); self.Wk = torch.nn.Linear(d, HEADS * RANK, bias=False)
        self.scorer = R.Scorer(du + HEADS * d, ns)

    def forward(self, U, V, M, S, pos, E, EM, ids):
        e, m = E[ids], EM[ids]; q = U[:, -self.d:]  # last block of the read = question embedding
        lg = torch.einsum('nhr,nlhr->nhl', self.Wq(q).view(-1, HEADS, RANK), self.Wk(e).view(*e.shape[:2], HEADS, RANK)) / RANK ** .5
        lg = lg + (e @ q[:, :, None]).squeeze(-1)[:, None, :] / RD.TEMP
        b = torch.nn.functional.normalize(torch.softmax(lg.masked_fill(~m[:, None, :], -1e9), -1) @ e, dim=-1)
        return self.scorer(torch.cat([U, b.reshape(len(U), -1)], -1), V, M, S, pos)


def logits(net, data, fb):
    E, EM, ids = fb
    net.eval()
    with torch.no_grad():
        return torch.cat([net(*[t[b:b + 1024] for t in data[:5]], E, EM, ids[b:b + 1024]) for b in range(0, len(ids), 1024)])


def train(data, fb, y, epochs=None, val=None, seed=0):
    """AdamW on the conditional CE (rich-expert recipe); with `val` = (data, fb, y): early stopping, returns (net, epochs)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    E, EM, ids = fb
    net = FieldScorer(data[0].shape[1], data[1].shape[-1], data[3].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=R.LR, weight_decay=R.WD)
    best, best_ep, state, bad = float('inf'), 0, None, 0
    for ep in range(1, (epochs or R.MAX_EP) + 1):
        net.train()
        order = rng.permutation(len(y))
        for b in range(0, len(y), R.BATCH):
            i = torch.tensor(order[b:b + R.BATCH])
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(*[t[i] for t in data[:5]], E, EM, ids[i]), y[i]).backward(); opt.step()
        if val is not None:
            nll = torch.nn.functional.cross_entropy(logits(net, val[0], val[1]), val[2]).item()
            if nll < best - 1e-4:
                best, best_ep, bad = nll, ep, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= R.PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    with torch.no_grad():
        for p in net.parameters():  # stored fp16
            p.copy_(p.half().float())
    return net, best_ep or (epochs or R.MAX_EP)


def sub(fb, sel):
    E, EM, ids = fb
    return E, EM, ids[sel]
