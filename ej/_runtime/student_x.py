"""Decision stack: the top layers of e5-small fine-tuned as a decision model over segmented inputs (the state is read once per
record; question and option segments attend to it), with batching and scoring helpers. Used at fit time by the decision
encoder."""
import copy
import sys

import numpy as np
import torch

F = torch.nn.functional


def _heads(sa, x):
    n, length, _ = x.shape
    return x.view(n, length, sa.num_attention_heads, sa.attention_head_size).transpose(1, 2)


def _kv(layer, x):
    sa = layer.attention.self
    return _heads(sa, sa.key(x)), _heads(sa, sa.value(x))


def _layer(layer, x, mask, ctx=(), kv=None):
    """One BERT layer: x attends to [ctx_1[i_1] ; ... ; x]. ctx = ((K, V, M, i), ...); masks bool (N, L), True = token."""
    sa = layer.attention.self
    k, v = kv if kv is not None else _kv(layer, x)
    if ctx:
        k = torch.cat([c[0][c[3]] for c in ctx] + [k], 2); v = torch.cat([c[1][c[3]] for c in ctx] + [v], 2)
        mask = torch.cat([c[2][c[3]] for c in ctx] + [mask], 1)
    a = F.scaled_dot_product_attention(_heads(sa, sa.query(x)), k, v, attn_mask=mask[:, None, None, :],
                                       dropout_p=sa.dropout.p if layer.training else 0.0)
    a = layer.attention.output(a.transpose(1, 2).reshape(x.shape), x)
    return layer.output(layer.intermediate(a), a)


def _pad(states):
    L, d = max(s.shape[0] for s in states), states[0].shape[1]
    H = torch.zeros(len(states), L, d); M = torch.zeros(len(states), L, dtype=torch.bool)
    for i, s in enumerate(states):
        H[i, :len(s)] = s.float(); M[i, :len(s)] = True
    return H, M


def _sum(h, m):
    return (h * m.unsqueeze(-1).float()).sum(1), m.sum(1, keepdim=True).float()


class Head(torch.nn.Module):
    """Decision head over (u, v_j, o_j)."""

    def __init__(self, w, b):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone()); self.b = torch.nn.Parameter(b.clone())
        self.r = torch.nn.Parameter(torch.zeros(w.shape[0] - 1)); self.c = torch.nn.Parameter(torch.zeros(()))


def batch_logits(top, head, S, items, sel, types, read=True):
    """(B, K) logits for the questions `sel`; -1e9 on padded option slots."""
    states = sorted({items[n][0] for n in sel}); sp = {s: k for k, s in enumerate(states)}
    sidx = torch.tensor([sp[items[n][0]] for n in sel])
    Hs, Ms = _pad([S[s] for s in states]); Hi, Mi = _pad([S[items[n][1]] for n in sel])
    opts = sorted({o for n in sel for o in items[n][2]}); pos = {o: k for k, o in enumerate(opts)}
    Hv, Mv = _pad([S[o] for o in opts])
    pairs = [(r, j, o) for r, n in enumerate(sel) for j, o in enumerate(items[n][2])]
    idx = torch.tensor([p[0] for p in pairs]); col = torch.tensor([p[1] for p in pairs])
    Ho, Mo = _pad([S[o] for _, _, o in pairs]) if read else (None, None)
    for layer in top.encoder.layer:
        ks, ki = _kv(layer, Hs), _kv(layer, Hi)
        cs = (ks[0], ks[1], Ms, sidx)
        if read:
            Ho = _layer(layer, Ho, Mo, ((ks[0], ks[1], Ms, sidx[idx]), (ki[0], ki[1], Mi, idx)))
        Hi = _layer(layer, Hi, Mi, (cs,), kv=ki); Hs = _layer(layer, Hs, Ms, kv=ks); Hv = _layer(layer, Hv, Mv)
    (ss, ns), (si, ni) = _sum(Hs, Ms), _sum(Hi, Mi)
    u = F.normalize((ss[sidx] + si) / (ns[sidx] + ni), dim=-1)[idx]
    v = F.normalize(_sum(Hv, Mv)[0], dim=-1)[[pos[p[2]] for p in pairs]]
    P = u * v; T = torch.tensor([types.index(items[n][3]) for n in sel])
    zp = torch.cat([P, P.sum(-1, keepdim=True)], -1) @ head.w + head.b[T][idx]
    if read:
        o = F.normalize(_sum(Ho, Mo)[0], dim=-1)
        zp = zp + o @ head.r + head.c * (u * o).sum(-1)
    z = torch.full((len(sel), max(len(items[n][2]) for n in sel)), -1e9)
    return z.index_put((idx, col), zp)


def batches(S, items, order, size):
    """Groups of whole records (consecutive by state length) with about `size` questions each."""
    by = {}
    for n in order:
        by.setdefault(items[n][0], []).append(n)
    out, cur = [], []
    for s in sorted(by, key=lambda s: S[s].shape[0]):
        cur += by[s]
        if len(cur) >= size:
            out.append(cur); cur = []
    return out + ([cur] if cur else [])


def score_items(top, head, S, items, types, read=True, chunk=48):
    """No-grad logits for all items; padded to the max option count."""
    top.eval(); out = torch.full((len(items), max(len(i[2]) for i in items)), -1e9)
    with torch.no_grad():
        for sel in batches(S, items, range(len(items)), chunk):
            z = batch_logits(top, head, S, items, sel, types, read); out[torch.tensor(sel), :z.shape[1]] = z
    return out


def finetune(top, head, S, items, hold, types, cfg, log=lambda s: print(s, file=sys.stderr, flush=True)):
    """AdamW on (top stack, head), conditional-logit CE over options; held-out NLL every half epoch, best kept."""
    rng = np.random.default_rng(cfg['seed']); torch.manual_seed(cfg['seed']); read = cfg['read']
    opt = torch.optim.AdamW([{'params': list(top.parameters()), 'lr': cfg['lr'], 'weight_decay': 0.01},
                             {'params': list(head.parameters()), 'lr': cfg['lr_head'], 'weight_decay': 0.0}])
    bs = batches(S, items, [n for n in range(len(items)) if not hold[n]], cfg['batch'])
    ho = [items[n] for n in range(len(items)) if hold[n]]; y_ho = torch.tensor([i[4] for i in ho])
    total = cfg['epochs'] * len(bs); warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    snap = lambda: (copy.deepcopy(top.state_dict()), copy.deepcopy(head.state_dict()))
    best = (F.cross_entropy(score_items(top, head, S, ho, types, read), y_ho).item(), snap())
    log(f'ft start held-out nll {best[0]:.4f} ({len(bs)} batches/epoch)')
    checks = {len(bs) // 2, len(bs)}
    for ep in range(cfg['epochs']):
        for j, bi in enumerate(rng.permutation(len(bs))):
            top.train(); sel = bs[bi]; y = torch.tensor([items[n][4] for n in sel])
            loss = F.cross_entropy(batch_logits(top, head, S, items, sel, types, read), y) + cfg['l2'] * (head.w ** 2).sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(top.parameters(), 1.0); opt.step(); sched.step()
            if j + 1 in checks:
                nll = F.cross_entropy(score_items(top, head, S, ho, types, read), y_ho).item()
                log(f'ft epoch {ep} batch {j + 1}/{len(bs)} held-out nll {nll:.4f}')
                if nll < best[0]:
                    best = (nll, snap())
    top.load_state_dict({k: v.half().float() for k, v in best[1][0].items()})  # top layers stored fp16 on device
    head.load_state_dict(best[1][1]); top.eval()
