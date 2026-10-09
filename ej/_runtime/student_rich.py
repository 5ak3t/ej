"""Rich expert: a structured state-option scorer (diagonal and low-rank bilinear interactions, an ordinal axis for score
questions and label-agnostic scalars) used as one expert of the calibrated pool."""
import numpy as np
import torch

from student_read import tile

RANK, HID, ORD = 32, 64, 16
LR, WD, BATCH, MAX_EP, PATIENCE = 2e-3, 1e-4, 256, 40, 4


class Scorer(torch.nn.Module):
    """s_j = a . phi_j + MLP(phi_j); masked over the question's options."""

    def __init__(self, d, ns):  # d = width of the state-once read u_q (option vectors are tiled to it)
        super().__init__()
        self.P = torch.nn.Linear(d, RANK, bias=False); self.Q = torch.nn.Linear(d, RANK, bias=False)
        self.R = torch.nn.Linear(d, ORD, bias=False)
        n = 2 * d + RANK + ORD + ns
        self.lin = torch.nn.Linear(n, 1)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n, HID), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(HID, 1))
        torch.nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, U, V, M, S, pos):
        m = M.float()[..., None]
        V = tile(V, U)
        C = (V - (V * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
        u = U[:, None, :]
        phi = torch.cat([u * V, u * C, self.P(U)[:, None, :] * self.Q(C), pos[..., None] * self.R(U)[:, None, :], S], -1)
        return (self.lin(phi) + self.mlp(phi)).squeeze(-1).masked_fill(~M, -1e9)


def logits(net, data):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(*[t[b:b + 2048] for t in data[:5]]) for b in range(0, len(data[0]), 2048)])


def train(data, y, epochs=None, val=None, seed=0):
    """AdamW on the conditional CE; with `val` = (data, y): early stopping, returns (net, best epoch count)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Scorer(data[0].shape[1], data[3].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best, best_ep, state, bad = float('inf'), 0, None, 0
    for ep in range(1, (epochs or MAX_EP) + 1):
        net.train()
        order = rng.permutation(len(y))
        for b in range(0, len(y), BATCH):
            i = torch.tensor(order[b:b + BATCH])
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(*[t[i] for t in data[:5]]), y[i]).backward(); opt.step()
        if val is not None:
            nll = torch.nn.functional.cross_entropy(logits(net, val[0]), val[1]).item()
            if nll < best - 1e-4:
                best, best_ep, bad = nll, ep, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    with torch.no_grad():
        for p in net.parameters():  # stored fp16
            p.copy_(p.half().float())
    return net, best_ep or (epochs or MAX_EP)


def n_params(net):
    return sum(p.numel() for p in net.parameters())
