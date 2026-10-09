"""Shallow device encoder: a distilled variant of the quantised encoder with 8 of its 12 layers, trained to reproduce the full
model's token states. Off in the released configuration."""
import copy
import hashlib
import os
import random
import sys
import time

import torch
import torch.nn.utils.parametrize as PZ

import student_enc as E
import student_fast as FA

MODEL, BITS = 'intfloat/e5-small-v2', 4
DROP = (4, 5, 6, 7)
N_TEXTS, EPOCHS, LR, BS, SEED = 20000, 3, 1e-4, 32, 0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EDGE_CKPT', os.path.expanduser('~/.cache/ej/ckpt'))
SRC = ('student_shallow.py', 'student_enc.py')


class _STE(torch.nn.Module):
    """4-bit fake quantisation with a straight-through gradient (forward = student_enc._quant)."""

    def forward(self, w):
        return w + (E._quant(w, BITS) - w).detach()


def texts_of(items):
    """Pool texts the device encodes (deterministic sample of N_TEXTS distinct texts)."""
    import student_dn as DN
    import student_feat as F
    import student_rr as RR
    T = set()
    for i in items:
        T.add(f"query: {i['state']}"); T.add(f"query: {i['instr']}")
        T.update(f'query: {c}' for c in F.chunks(i['state']))
        T.update(f'passage: {o}' for o in i['opts']); T.update(f"passage: {i['instr']} {o}" for o in i['opts'])
        if not i['json']:
            T.update(DN._prefix(h, k) for h, k in DN.item_hyps(i))
    for s in sorted({i['state'] for i in items if i['json']}):
        T.update(f'query: {t}' for f in RR.facts(s) for t in (f[0], f[2]))
    T = sorted(T)
    return sorted(random.Random(SEED).sample(T, min(N_TEXTS, len(T))))


def _key(texts):
    h = hashlib.sha256()
    for f in SRC:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update('\n'.join(texts).encode())
    return h.hexdigest()[:16]


def build(sd=None):
    """Student encoder: a copy of the 4-bit e5 without the DROP layers (optionally loading a trained state dict)."""
    tok, enc = E._ENC[E.load(MODEL, BITS)]
    stu = copy.deepcopy(enc)
    stu.encoder.layer = torch.nn.ModuleList([m for k, m in enumerate(stu.encoder.layer) if k not in DROP])
    if sd is not None:
        stu.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    return tok, stu.eval()


def _train(texts):
    torch.manual_seed(SEED); rng = random.Random(SEED)
    tok, enc = E._ENC[E.load(MODEL, BITS)]
    ho = [t for t in texts if hashlib.sha256(t.encode()).digest()[0] < 13]  # ~5% held-out (by text hash) for early stopping
    hs_ = set(ho); tr = [t for t in texts if t not in hs_]
    teach = FA._run(lambda i, m: enc(input_ids=i, attention_mask=m).last_hidden_state, tok, texts, E.MAX_LEN)
    _, stu = build(); stu.train()
    for n, p in stu.named_parameters():
        p.requires_grad = 'embeddings' not in n and 'pooler' not in n
    for m in stu.encoder.modules():
        if isinstance(m, torch.nn.Linear):
            PZ.register_parametrization(m, 'weight', _STE())
    opt = torch.optim.Adam([p for p in stu.parameters() if p.requires_grad], lr=LR)

    def loss(ch):
        b = tok(ch, padding=True, truncation=True, max_length=E.MAX_LEN, return_tensors='pt')
        hs = stu(**b).last_hidden_state; m = b['attention_mask'].bool()
        Y = torch.nn.utils.rnn.pad_sequence([teach[t][1] for t in ch], batch_first=True)[:, :hs.shape[1]]
        tc = (1 - torch.nn.functional.cosine_similarity(hs, Y, dim=-1))[m].mean()
        mm = m.unsqueeze(-1).float()
        pc = 1 - torch.nn.functional.cosine_similarity((hs * mm).sum(1), (Y * mm).sum(1), dim=-1).mean()
        return tc + pc

    def held():
        with torch.no_grad():
            s = sorted(ho, key=len)
            return sum(loss(s[b:b + BS]).item() * len(s[b:b + BS]) for b in range(0, len(s), BS)) / len(s)
    s = sorted(tr, key=len); bl = [s[b:b + BS] for b in range(0, len(s), BS)]
    steps, t0 = EPOCHS * len(bl), time.time()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 100) * 0.5 * (1 + torch.cos(torch.tensor(k / steps * 3.14159)).item()))
    rep, best, bad = [round(held(), 5)], None, 0
    for ep in range(EPOCHS):
        rng.shuffle(bl)
        for ch in bl:
            stu.train(); opt.zero_grad(); loss(ch).backward(); opt.step(); sched.step()
        stu.eval(); rep.append(round(held(), 5))
        print('shallow ep', ep + 1, 'held-out loss', rep[-1], round(time.time() - t0), 's', file=sys.stderr, flush=True)
        if best is None or rep[-1] < best[0]:
            best, bad = (rep[-1], {k: v.detach().clone() for k, v in stu.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= 3 or rep[-1] != rep[-1]:
                break
    stu.load_state_dict(best[1])  # best held-out epoch
    for m in stu.encoder.modules():
        if isinstance(m, torch.nn.Linear) and PZ.is_parametrized(m):
            PZ.remove_parametrizations(m, 'weight', leave_parametrized=True)  # weights = their 4-bit values
    sd = {k: v for k, v in stu.state_dict().items() if k.startswith('encoder.')}
    return {'sd': sd, 'rep': {'held_loss': rep, 'n_texts': len(texts), 'n_ho': len(ho), 'drop': DROP, 'sec': round(time.time() - t0)}}


def fit(items):
    """Memoised: load the checkpoint whose hash matches the pool texts + code, else train in-process."""
    texts = texts_of(items)
    path = f'{ROOT}/shallow-{_key(texts)}.pt'
    if os.path.exists(path):
        print('shallow: cache', path, file=sys.stderr)
        return torch.load(path, weights_only=False)
    out = _train(texts)
    os.makedirs(ROOT, exist_ok=True); torch.save(out, path + '.tmp'); os.replace(path + '.tmp', path)
    return out


_BUILT = {}


def encoder(ck):
    """(tokenizer, fn(input_ids, mask) -> token states, tag) for student_fast.begin; the model is built once per checkpoint."""
    key = id(ck['sd'])
    if key not in _BUILT:
        _BUILT.clear(); _BUILT[key] = build(ck['sd'])
    tok, stu = _BUILT[key]
    return tok, (lambda i, m: stu(input_ids=i, attention_mask=m).last_hidden_state), f'shallow{DROP}'


def size_mb(ck):
    """Full-encoder size minus the dropped layers (4-bit Linear weights + fp32 rest), as student_enc.size_mb counts it."""
    _, enc = E._ENC[E.load(MODEL, BITS)]
    lay = [enc.encoder.layer[k] for k in DROP]
    qn = sum(m.weight.numel() for L in lay for m in L.modules() if isinstance(m, torch.nn.Linear))
    tot = sum(p.numel() for L in lay for p in L.parameters())
    return E.size_mb(MODEL, BITS) - (qn * (BITS + 16 / E.GROUP) + 32 * (tot - qn)) / 8 / 2 ** 20
