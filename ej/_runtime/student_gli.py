"""Single-pass multi-option reader: one encoder pass per record over the state together with every question and option,
scoring options from their in-context token representations. Off in the released configuration."""
import hashlib
import os
import sys

import numpy as np
import torch

import student_cold as COLD
import student_enc as E

MODEL, BITS, MAX_LEN, MAX_OPT, MAX_INS = 'intfloat/e5-small-v2', 4, 512, 16, 48
KINDS = ('mlp', 'bil')
HID, RANK, LR, WD, DROP, BATCH, MAX_EP, PATIENCE = 128, 32, 1e-3, 1e-2, 0.1, 128, 40, 3
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
_MEM = {}


def _ids(tok, text, cap):
    return tok(text, add_special_tokens=False)['input_ids'][:cap]


def layout(tok, rec):
    """(token ids, spans): spans = per question (instr span, [(marker pos, option span)]), state span; options never truncated."""
    qs = list(rec['questions'].values())
    ins = [_ids(tok, q['instructions'], MAX_INS) for q in qs]
    opt = [[_ids(tok, o['text'], MAX_OPT) or [tok.unk_token_id] for o in q['options']] for q in qs]
    cap = MAX_OPT
    while 3 + sum(len(a) + sum(len(o) + 1 for o in os_) for a, os_ in zip(ins, opt)) > MAX_LEN - 34 and cap > 4:
        cap //= 2; ins = [a[:cap] for a in ins]; opt = [[o[:cap] for o in os_] for os_ in opt]
    bar = tok.convert_tokens_to_ids('|')
    ids, spans = [tok.cls_token_id] + _ids(tok, 'query:', 4), []
    for a, os_ in zip(ins, opt):
        s0 = len(ids); ids += a; o_sp = []
        for o in os_:
            mk = len(ids); ids += [bar] + o; o_sp.append((mk, mk + 1, len(ids)))
        spans.append(((s0, max(s0 + len(a), s0 + 1)), o_sp))
    ids.append(tok.sep_token_id)
    st = _ids(tok, rec['state'] or '-', MAX_LEN - len(ids) - 1)
    t0 = len(ids); ids += st + [tok.sep_token_id]
    return ids, spans, (t0, t0 + max(len(st), 1))


def _pool(H, spans, ts):
    t = H[ts[0]:ts[1]].mean(0)
    return [(torch.stack([H[a:b].mean(0) for _, a, b in o]).half(), H[[m for m, _, _ in o]].half(), H[i0:i1].mean(0).half(), t.half())
            for (i0, i1), o in spans]


def read(recs, batch=16):
    """Per question (record order, then question order -- as student.questions): (O (k, d), Mk (k, d), q (d), t (d)) fp16."""
    tok, enc = E._ENC[E.load(MODEL, BITS)]
    cold = COLD.cold()
    mem = {} if cold else _MEM
    lay = [layout(tok, r) for r in recs]
    keys = [hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()[:24] for ids, _, _ in lay]
    path = f"{E.CACHE}/gli_{MODEL.replace('/', '_')}#q{BITS}#{MAX_LEN}.pt"
    if not cold and not _MEM and os.path.exists(path):
        _MEM.update(torch.load(path))
    todo = sorted({k: n for n, k in enumerate(keys) if k not in mem}.values(), key=lambda n: len(lay[n][0]))
    with torch.no_grad():
        for b in range(0, len(todo), batch):
            ch = todo[b:b + batch]; L = max(len(lay[n][0]) for n in ch)
            x = torch.zeros(len(ch), L, dtype=torch.long); am = torch.zeros(len(ch), L, dtype=torch.long)
            for r, n in enumerate(ch):
                x[r, :len(lay[n][0])] = torch.tensor(lay[n][0]); am[r, :len(lay[n][0])] = 1
            H = enc(input_ids=x, attention_mask=am, token_type_ids=torch.zeros_like(x)).last_hidden_state
            for r, n in enumerate(ch):
                mem[keys[n]] = _pool(H[r], lay[n][1], lay[n][2])
            if len(todo) > 2000 and b % (batch * 100) == 0:
                print('gli read', b, len(todo), file=sys.stderr, flush=True)
    if len(todo) > 200 and not cold:
        os.makedirs(E.CACHE, exist_ok=True)
        torch.save(_MEM, path + f'.{os.getpid()}'); os.replace(path + f'.{os.getpid()}', path)
    return [qv for k in keys for qv in mem[k]]


def tensors(qv, types):
    """Padded (O (N, K, d), Mk (N, K, d), q (N, d), t (N, d), M (N, K), T (N)) float32."""
    N, K, d = len(qv), max(v[0].shape[0] for v in qv), qv[0][2].shape[0]
    O, Mk, M = torch.zeros(N, K, d), torch.zeros(N, K, d), torch.zeros(N, K, dtype=torch.bool)
    for n, (o, m, _, _) in enumerate(qv):
        O[n, :len(o)], Mk[n, :len(o)], M[n, :len(o)] = o.float(), m.float(), True
    return O, Mk, torch.stack([v[2].float() for v in qv]), torch.stack([v[3].float() for v in qv]), M, torch.tensor(types)


class Reader(torch.nn.Module):
    """Option scorer over the in-context pooled vectors (shared across options, questions and tasks). kind 'mlp': MLP on the
    centred option / marker vectors and their products with t, q; kind 'bil': GLiClass-style low-rank dot products only,
    z_j = <P a_j, Qt t> + <P a_j, Qq q> + <P b_j, Qt t> (no raw option features, so it cannot memorise label identities)."""

    def __init__(self, d, kind='mlp'):
        super().__init__()
        self.kind = kind
        if kind == 'mlp':
            self.mlp = torch.nn.Sequential(torch.nn.Dropout(DROP), torch.nn.Linear(5 * d, HID), torch.nn.GELU(), torch.nn.Linear(HID, 1))
        else:
            self.P, self.Qt, self.Qq = (torch.nn.Linear(d, RANK, bias=False) for _ in range(3))
        self.lin = torch.nn.Linear(6, 1)

    def forward(self, O, Mk, q, t, M, T):
        m = M.float()[..., None]; k = m.sum(1, keepdim=True).clamp(min=1)
        a, b = (O - (O * m).sum(1, keepdim=True) / k) * m, (Mk - (Mk * m).sum(1, keepdim=True) / k) * m
        if self.kind == 'mlp':
            x = self.mlp(torch.cat([a, b, a * t[:, None], a * q[:, None], O * t[:, None]], -1)).squeeze(-1)
        else:
            ht, hq = self.Qt(t)[:, None], self.Qq(q)[:, None]
            x = (self.P(a) * (ht + hq)).sum(-1) + (self.P(b) * ht).sum(-1)
        nt = lambda v: torch.nn.functional.normalize(v, dim=-1)
        c = lambda v: (v - (v * M).sum(1, keepdim=True) / k[..., 0]) * M
        cs = [c((nt(O) * nt(t)[:, None]).sum(-1)), c((nt(Mk) * nt(t)[:, None]).sum(-1)), c((nt(O) * nt(q)[:, None]).sum(-1))]
        j = torch.arange(M.shape[1]).float()[None]; pos = j / (k[..., 0] - 1).clamp(min=1) - 0.5
        s = torch.stack(cs + [pos * (T[:, None] == 2), pos * (T[:, None] == 1), pos * (T[:, None] == 0)], -1)
        return (x + self.lin(s).squeeze(-1)).masked_fill(~M, -1e9)


def _sub(X, sel):
    return tuple(v[sel] for v in X)


def logits(net, X):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(*_sub(X, slice(b, b + 4096))) for b in range(0, len(X[0]), 4096)])


def train(X, y, epochs=None, val=None, seed=0, kind='mlp'):
    """AdamW on the conditional logit; early stopping on `val` = (X, y) (patience PATIENCE); weights stored fp16."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Reader(X[0].shape[-1], kind); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best, best_ep, state, bad = float('inf'), 0, None, 0
    for ep in range(1, (epochs or MAX_EP) + 1):
        net.train(); order = rng.permutation(len(y))
        for b in range(0, len(y), BATCH):
            i = torch.tensor(order[b:b + BATCH])
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(*_sub(X, i)), y[i]).backward(); opt.step()
        if val is not None:
            v = torch.nn.functional.cross_entropy(logits(net, val[0]), val[1]).item()
            if v < best - 1e-4:
                best, best_ep, bad, state = v, ep, 0, {k: x.clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(p.half().float())
    return net, best_ep or (epochs or MAX_EP), round(best, 4)


def fit(recs, items, y, hold, gid):
    """{'ho': held-out logits, 'logo': {group: logits}, 'full': device reader, 'rep'} (memoised under ROOT/gli-<hash>.pt)."""
    X = tensors(read(recs), [i['type'] for i in items])
    h = hashlib.sha256(open(os.path.join(HERE, 'student_gli.py'), 'rb').read())
    for v in X + (y,):
        h.update(v.contiguous().numpy().tobytes())
    h.update(np.asarray(hold).tobytes()); h.update('\x1f'.join(gid).encode())
    path = f'{ROOT}/gli-{h.hexdigest()[:16]}.pt'
    if os.path.exists(path):
        print('gli: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    tr, ho = torch.tensor(~np.asarray(hold)), torch.tensor(np.asarray(hold))
    outs = []
    for kind in KINDS:  # scorer family chosen inside fit by mean leave-one-group-out NLL (zero-shot transfer), never on dev
        net, ep, v = train(_sub(X, tr), y[tr], val=(_sub(X, ho), y[ho]), kind=kind)
        out = {'ho': logits(net, _sub(X, ho)).masked_fill(~X[4][ho], 0), 'logo': {}, 'rep': {'kind': kind, 'epochs': ep, 'ho_nll': v, 'logo': {}}}
        for g in sorted(set(gid)):
            o = torch.tensor(np.asarray(gid) == g)
            z = logits(train(_sub(X, ~o), y[~o], epochs=ep, kind=kind)[0], _sub(X, o))
            out['rep']['logo'][g] = round(torch.nn.functional.cross_entropy(z, y[o]).item(), 4)
            out['logo'][g] = z.masked_fill(~X[4][o], 0)
        out['rep']['logo_mean'] = round(float(np.mean(list(out['rep']['logo'].values()))), 4)
        print('gli', out['rep'], file=sys.stderr, flush=True)
        outs.append(out)
    out = min(outs, key=lambda o: o['rep']['logo_mean'])
    out['full'] = train(X, y, epochs=out['rep']['epochs'], kind=out['rep']['kind'])[0]
    os.makedirs(ROOT, exist_ok=True)
    torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    print('gli: computed ->', path, out['rep'], file=sys.stderr)
    return out


def device_logits(net, recs, types):
    """Predict time: one joint pass per record, then the reader head."""
    return logits(net, tensors(read(recs), types))


def n_params(net):
    return sum(p.numel() for p in net.parameters())
