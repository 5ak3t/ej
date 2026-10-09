"""Decision-trained encoder (fit-time teacher): e5-small-v2 with frozen lower layers and fine-tuned top layers that score
option texts with a conditional-logit loss; also provides the group folds and held-out masks used by cross-fitting. Not a
device expert in the released configuration; its out-of-fold outputs train student_dd."""
import hashlib

import numpy as np
import torch

import student_enc as E
import student_ft as FT
import student_x as XS

MODEL, BITS = 'intfloat/e5-small-v2', 4
TYPES = ('choice', 'noul', 'score')
L2 = 1e-3
K_TOP, LEN_STATE, LEN_SHORT = 3, 320, 64
CFG = {'seed': 0, 'lr': 5e-5, 'lr_head': 1e-3, 'batch': 32, 'epochs': 1, 'l2': 1e-4, 'read': True}
# group folds for zero-shot cross-fitting: one TD workflow and/or one intent source per fold (a priori, by task kind)
FOLDS = ({'typed-decisions/agent_trace_observability', 'banking77/None'},
         {'typed-decisions/customer_service', 'clinc150/None'},
         {'typed-decisions/invoice_processing', 'counterfactual/None'},
         {'goemotions/None', 'tickets/None'})


def units(recs):
    """Segmented items: (state, 'query: '+instructions, ['passage: '+option], type, label, group)."""
    return [(r['state'], f"query: {q['instructions']}", [f"passage: {o['text']}" for o in q['options']], q['type'],
             r.get('gold', {}).get(qid, {}).get('label'), f"{r.get('source')}/{r.get('workflow')}")
            for r in recs for qid, q in r['questions'].items()]


def hold_mask(states):
    """10% held-out slice by STATE (all questions of a record together); shared by every expert of the merged student."""
    return np.array([int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10 == 0 for s in states])


def fold_of(group):
    """Fixed fold for the original groups; any other group (data v4: new text workflows) gets a fold by a stable hash of its name."""
    return next((k for k, f in enumerate(FOLDS) if group in f), int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % len(FOLDS))


def _probe(its):
    """Linear probe on the frozen encoder (LP of LP-FT): (w, b) over [u ⊙ v_j, u·v_j] + b_type."""
    U = torch.tensor(E.embed(MODEL, BITS, [f"{i[1]}\n{i[0]}" for i in its]))
    opts = sorted({o for i in its for o in i[2]}); pos = {o: k for k, o in enumerate(opts)}
    Vo = torch.tensor(E.embed(MODEL, BITS, opts))
    K = max(len(i[2]) for i in its)
    idx = np.full((len(its), K), -1)
    for n, i in enumerate(its):
        idx[n, :len(i[2])] = [pos[o] for o in i[2]]
    M = torch.tensor(idx >= 0)
    Fx = U[:, None, :] * Vo[np.maximum(idx, 0)]
    X = torch.cat([Fx, Fx.sum(-1, keepdim=True)], -1); T = torch.tensor([TYPES.index(i[3]) for i in its])
    y = torch.tensor([i[4] for i in its])
    w = torch.zeros(X.shape[-1], requires_grad=True); b = torch.zeros(len(TYPES), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy((X @ w + b[T][:, None]).masked_fill(~M, -1e9), y) + L2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def states(tok, lower, its):
    """Lower-stack token states for every state / instruction / option text of `its` (disk-cached)."""
    tag = f"{MODEL.replace('/', '_')}#q{BITS}#L{12 - K_TOP}"
    S = FT.lower_states(tok, lower, [i[0] for i in its], LEN_STATE, f'{tag}#{LEN_STATE}')
    S.update(FT.lower_states(tok, lower, [i[1] for i in its] + [o for i in its for o in i[2]], LEN_SHORT, f'{tag}#{LEN_SHORT}'))
    return S


def train(its, S, hold):
    """LP-FT on the non-held-out items of `its`; the held-out slice selects the checkpoint. Returns (top, head)."""
    head0 = _probe([i for i, h in zip(its, hold) if not h])
    _, _, top = FT.split(MODEL, BITS, K_TOP)
    head = XS.Head(*head0)
    XS.finetune(top, head, S, its, hold, TYPES, CFG)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(p.half().float())
    return top, head


def logits(top, head, S, its):
    """(n, K) logits; -1e9 on padded option slots."""
    return XS.score_items(top, head, S, its, TYPES, CFG['read'])


def lower():
    tok, low, _ = FT.split(MODEL, BITS, K_TOP)
    return tok, low


def size_extra_mb(top, head):
    """Device size added on top of the shared 4-bit encoder: fp16 top stack minus its 4-bit copy, plus the head."""
    q4, f16 = FT.size_top_mb(top, BITS)
    return f16 - q4 + sum(p.numel() for p in head.parameters()) * 2 / 2 ** 20
