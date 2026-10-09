"""Deep expert: a convex state-option scorer over frozen sentence embeddings (state and query, option, option-set context and
statement-as-hypothesis views) with label-agnostic scalar features, so it transfers to option sets not seen in training.
Also provides the centred (prior-free) variant."""
import torch

from student_read import fold

L2 = 1e-3
NS = 13


def scalars(U, V, Hs, Ho, M, lex, late, T):
    """(N, K, NS) option-varying scalars; masked. U: (N, d) unit read (folded state-once read)."""
    m = M.float()
    mean = lambda X: (X * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
    cosv = (U[:, None, :] * V).sum(-1)
    hyp = (Hs[:, None, :] * Ho).sum(-1)
    L = torch.as_tensor(lex)
    k = M.sum(1, keepdim=True).float()
    j = torch.arange(V.shape[1]).float()[None, :]
    pos = torch.where(T[:, None] == 2, j / (k - 1).clamp(min=1) - 0.5, torch.zeros_like(j))
    cols = [cosv, cosv - mean(cosv), hyp, hyp - mean(hyp), L[..., 0], L[..., 0] - mean(L[..., 0]), L[..., 1],
            L[..., 1] - mean(L[..., 1]), pos, late[..., 0], late[..., 0] - mean(late[..., 0]), late[..., 1],
            late[..., 1] - mean(late[..., 1])]
    return torch.stack(cols, -1) * m[..., None], pos * m


def init(d):
    return {'w1': torch.zeros(d), 'w2': torch.zeros(d), 'w3': torch.zeros(d), 'ws': torch.zeros(3, NS)}


def logits(theta, data):
    U, V, M, S, pos, T = data
    m = M.float()[..., None]
    C = (V - (V * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    d = V.shape[-1]  # U = state-once read [s ; a_q ; q] (student_read): w . (u_q * v) = v . sum_r w_r * u_r
    z = (V * fold(U * theta['w1'], d)[:, None, :]).sum(-1) + (C * fold(U * theta['w2'], d)[:, None, :]).sum(-1)
    z = z + pos * (U @ theta['w3'])[:, None] + (S * theta['ws'][T][:, None, :]).sum(-1)
    return z.masked_fill(~M, -1e9)


def train(data, y):
    """Conditional logit with L2 by L-BFGS (convex)."""
    theta = {k: v.requires_grad_() for k, v in init(data[0].shape[1]).items()}
    opt = torch.optim.LBFGS(list(theta.values()), max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits(theta, data), y) + L2 * sum((p * p).sum() for p in theta.values())
        loss.backward()
        return loss
    opt.step(closure)
    return {k: v.detach().half().float() for k, v in theta.items()}  # stored fp16


def n_params(theta):
    return sum(v.numel() for v in theta.values())


def centred(theta, data, ubar):
    """Prior-free logits: z(u) minus the u-linear part evaluated at the mean training query u_bar, i.e. the score
    component that depends on the option text alone (a label marginal of the training tasks; label shift on a new task)."""
    U, V, M, S, pos, T = data
    m = M.float()[..., None]
    C = (V - (V * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    d = V.shape[-1]
    cb = V @ torch.nn.functional.normalize(fold(ubar, d), dim=-1)
    cbc = cb - (cb * M.float()).sum(1, keepdim=True) / M.float().sum(1, keepdim=True)
    ws = theta['ws'][T]
    prior = V @ fold(ubar * theta['w1'], d) + C @ fold(ubar * theta['w2'], d) + pos * (ubar @ theta['w3'])
    prior = prior + ws[:, None, 0] * cb + ws[:, None, 1] * cbc
    return (logits(theta, data) - prior * M.float()).masked_fill(~M, -1e9)
