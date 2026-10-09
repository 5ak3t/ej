"""Slot-free relational reader: relational facts between JSON fields are rendered as text, embedded and scored against each
option without option-slot keys, so the evidence transfers to new workflows; also a zero-shot verifier variant."""
import itertools

import torch

import student_enc as E
import student_feat as F
import student_rel as REL

MODEL, BITS = 'intfloat/e5-small-v2', 4
TEMP = 0.05
MAX_FACTS = 96
RELS = tuple(k + t for k in ('member', 'zero', 'neg', 'eq', 'same') for t in '+-')
NR = len(RELS)
NS = 4 + 3 * NR + 4
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GENERIC = {'id', 'ids', 'sum', 'the', 'of', 'a'}


def _name(path):
    """Readable name of a (possibly derived) leaf path: 'invoice.lines[].qty*unit_usd#sum' -> 'the sum of invoice lines qty times unit
    usd'."""
    agg = path.endswith('#sum')
    p = path[:-4] if agg else path
    w = ' '.join(F.words(p.replace('*', ' times ')))
    return f'the sum of {w}' if agg else w


def _unit(path):
    """Words of the last segment (what the quantity measures), used to keep only comparable numeric pairs."""
    last = (path[:-4] if path.endswith('#sum') else path).split('.')[-1].split('*')[-1]  # product: unit of its last factor
    return set(F.words(last)) - GENERIC


def facts(state):
    """[(true text, kind+/-, false text)] value-free CANONICAL fact pairs of a JSON state ([] for plain text). Which pairs are emitted,
    and their order, depends on the SCHEMA only (paths), never on which side holds: the truth decides only the sign. (Round-4 probe:
    emitting facts conditional on their truth, e.g. 'id appears in list' only on a hit, gave the contrast a question-specific level
    that did not transfer.) Pairs: list membership, zero, negative, near-equality of comparable quantities (shared unit word),
    equality of string fields sharing a word."""
    obj = F.parse_json(state)
    if obj is None:
        return []
    num, strs, lists, objs = {}, {}, {}, []
    REL._walk(obj, '', num, strs, lists, objs)
    num.update(REL._derived(objs))
    keys = sorted(num)[:REL.MAX_NUM]
    out = []

    def pair(kind, holds, a_text, b_text):
        out.append((a_text, kind + '+', b_text) if holds else (b_text, kind + '-', a_text))
    for b, vals in sorted(lists.items()):
        hit = any(strs[a] in vals for a in strs if a != b) or any(REL._numstr(num[a]) in vals for a in keys if '#' not in a)
        pair('member', hit, f'{_name(b)} contains the value of another field', f'{_name(b)} contains none of the other values')
    base = [a for a in keys if not a.endswith('#sum') and '*' not in a]
    for a in base:
        pair('zero', num[a] == 0, f'{_name(a)} is zero', f'{_name(a)} is not zero')
    for a in base:
        pair('neg', num[a] < 0, f'{_name(a)} is negative', f'{_name(a)} is not negative')
    near, far = [], []
    for a, b in itertools.combinations(keys, 2):
        ua, ub = _unit(a), _unit(b)
        if b == a + '#sum' or a == b + '#sum' or not (ua & ub):
            continue
        (near if ua <= ub or ub <= ua else far).append((a, b))
    for a, b in near + far:
        r = REL._cmp(num[a], num[b])
        pair('eq', r == 'eq' or r.endswith('tiny'), f'{_name(a)} matches {_name(b)}', f'{_name(a)} differs from {_name(b)}')
    for a, b in itertools.combinations(sorted(strs), 2):
        if _unit(a) & _unit(b):
            pair('same', strs[a] == strs[b], f'{_name(a)} is the same as {_name(b)}', f'{_name(a)} is different from {_name(b)}')
    return out[:MAX_FACTS]


def bank(items):
    """{state: (fact embeddings (m, d), counterfactual embeddings (m, d), relation one-hot (m, NR))} for every distinct JSON
    state; the value-free fact texts are embedded once (cached per schema)."""
    fs = {i['state']: facts(i['state']) for i in items}
    texts = sorted({f'query: {t}' for v in fs.values() for f in v for t in (f[0], f[2])})
    if not texts:
        return {}
    Em = torch.tensor(E.embed(MODEL, BITS, texts)); pos = {t: k for k, t in enumerate(texts)}
    out = {}
    for s, v in fs.items():
        if v:
            R = torch.zeros(len(v), NR); R[torch.arange(len(v)), torch.tensor([RELS.index(f[1]) for f in v])] = 1
            out[s] = (Em[[pos[f'query: {f[0]}'] for f in v]], Em[[pos[f'query: {f[2]}'] for f in v]], R)
    return out


def kind_rates(fb):
    """mu_k = mean sign of each fact kind over the training states' facts (+1 = the 'a' side holds): how USUAL each side is. A fact
    on its usual side ('days until due is not negative') carries little evidence, a surprising one ('... is negative') much --
    surprise weighting by kind, a label-agnostic prior that transfers (pool probe: unweighted contrasts had a question-specific level
    set by the common facts, e.g. sup(h_last) - sup(h_first) < 0 for both classes)."""
    R = torch.cat([v[2] for v in fb.values()]) if fb else torch.zeros(1, NR)
    c = R.sum(0).view(-1, 2)
    return (c[:, 0] - c[:, 1]) / c.sum(1).clamp(min=1)


def inputs(items, fb, q, V, Ho, M, T, mu):
    """Per-question tensors (r (N, d), Vc (N, K, d), Hc (N, K, d), S (N, K, NS), M, T); all zero where a state has no facts."""
    N, K, d = V.shape
    m = M.float()
    mean = lambda X: (X * m[..., None]).sum(1, keepdim=True) / m.sum(1)[:, None, None]
    C, Hc = (V - mean(V)) * m[..., None], (Ho - mean(Ho)) * m[..., None]
    r, S = torch.zeros(N, d), torch.zeros(N, K, NS)
    k = M.sum(1).float()
    j = torch.arange(K).float()[None, :]
    pol = torch.where(T[:, None] == 2, j / (k[:, None] - 1).clamp(min=1) - 0.5, torch.where(T[:, None] == 1, j - 0.5,
                                                                                         torch.zeros_like(j)))
    for n, i in enumerate(items):
        if i['state'] not in fb:
            continue
        Ef, Ecf, R = fb[i['state']]
        kk = len(i['opts'])
        a = torch.softmax(Ef @ q[n] / TEMP, 0)
        r[n] = torch.nn.functional.normalize(a @ Ef, dim=0)
        cv, ch = V[n, :kk] @ Ef.T, Ho[n, :kk] @ Ef.T  # (k, m)
        pj = torch.softmax(ch / TEMP, -1) @ R  # option-view relation profile (k, NR)
        pjc = pj - pj.mean(0, keepdim=True)
        lv, lh = cv.amax(1), ch.amax(1)
        sg = R.view(-1, NR // 2, 2) @ torch.tensor([1.0, -1.0])  # (m, kinds): sign of each fact in its kind column
        om = 1 - (sg * mu).sum(1)  # surprise weight 1 - s * mu_kind
        sq, sv, sh = _support(q[n:n + 1], Ef, Ecf, om), _support(V[n, :kk], Ef, Ecf, om), _support(Ho[n, :kk], Ef, Ecf, om)
        sv, sh, pl = sv - sv.mean(), sh - sh.mean(), pol[n, :kk]
        S[n, :kk] = torch.cat([torch.stack([lv, lv - lv.mean(), lh, lh - lh.mean()], 1), pjc, pol[n, :kk, None] * (a @ R)[None],
                               pol[n, :kk, None] * pjc, torch.stack([pl * sq, sv, sh, pl * sh], 1)], 1)
    return r, C, Hc, S * m[..., None], M, T, V


def _support(H, Ef, Ecf, om):
    """Verification by counterfactual contrast: sum_f rel_f (cos(H, fact_f) - cos(H, counterfactual_f)) / T, where the relevance
    rel = softmax_f(mean of both cosines / T) does not depend on which version holds -> > 0 when H reads like what IS true."""
    ct, cc = H @ Ef.T, H @ Ecf.T
    return (torch.softmax((ct + cc) / (2 * TEMP), -1) * (ct - cc) * om).sum(-1) / TEMP


def sub(x, sel):
    return tuple(t[sel] for t in x)


def logits(th, x):
    r, C, Hc, S, M, T, V = x
    z = (V * (r * th['w1'])[:, None]).sum(-1) + (C * (r * th['w2'])[:, None]).sum(-1) + (Hc * (r * th['w3'])[:, None]).sum(-1)
    return (z + (S * th['ws'][T][:, None, :]).sum(-1)).masked_fill(~M, -1e9)


def train(x, y, l2):
    """Convex conditional logit by L-BFGS with L2 on STANDARDISED scalars (the contrast scores are ~1e-2, an unscaled L2 would erase
    them); the scale is folded back into ws (deterministic; stored fp16)."""
    d = x[0].shape[1]
    live = x[4]
    sd = x[3][live].std(0).clamp(min=1e-6)
    x = x[:3] + (x[3] / sd,) + x[4:]
    th = {'w1': torch.zeros(d), 'w2': torch.zeros(d), 'w3': torch.zeros(d), 'ws': torch.zeros(3, NS)}
    th = {k: v.requires_grad_() for k, v in th.items()}
    opt = torch.optim.LBFGS(list(th.values()), max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits(th, x), y) + l2 * sum((p * p).sum() for p in th.values())
        loss.backward()
        return loss
    opt.step(closure)
    th = {k: v.detach() for k, v in th.items()}
    th['ws'] = th['ws'] / sd
    return {k: v.half().float() for k, v in th.items()}


def logo_l2(x, y, gid, js):
    """L2 by leave-one-workflow-out NLL over the structured-state groups (transfer, decided inside fit); returns (l2, {l2: nll})."""
    gs = sorted({g for g, j in zip(gid, js.tolist()) if j})
    rep = {}
    for l2 in L2_GRID:
        tot, n = 0.0, 0
        for g in gs:
            o = torch.tensor([gg == g for gg in gid])
            tr = js & ~o
            z = logits(train(sub(x, tr), y[tr], l2), sub(x, o))
            tot += torch.nn.functional.cross_entropy(z, y[o], reduction='sum').item(); n += int(o.sum())
        rep[l2] = round(tot / max(n, 1), 4)
    return min(rep, key=rep.get), rep


def verify(x):
    """The ZERO-SHOT VERIFIER view of the inputs: only the scalars whose sign is a property of reading, not of a label space -- the
    option-view counterfactual support (centred; polarity-weighted) of the option and of the hypothesis and the centred late maximum
    (pool-only probe: sup(h_true) - sup(h_false) ranks the gold side with AUC .84-.96 on 5 of the 8 non-choice Typed Decisions
    questions with ZERO parameters, while the question-view support and the relation profiles flip sign between workflows)."""
    keep = torch.zeros(NS); keep[[3, NS - 3, NS - 2, NS - 1]] = 1
    return (torch.zeros_like(x[0]), x[1], x[2], x[3] * keep) + tuple(x[4:])


def n_params(th):
    return sum(v.numel() for v in th.values())
