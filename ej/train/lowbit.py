"""Quantisation-aware distillation of the LOW-BIT shared encoder (runtime student_lbq): ``python -m ej.train encoder``.

Recipe (adapted from BitDistill, arXiv:2510.13998):
  0. trimmed vocabulary (student_lbq.keep_ids) over ALL pool texts the device encodes (student_shallow.texts_of, every text);
  1. init: MSE-clipped per-group (s, z) for every matrix, then GPTQ (Frantar et al., arXiv:2210.17323) on the 72 encoder Linears,
     layer by layer, on CAL pool calibration texts (round-to-nearest at 2-3 bits damages e5 badly);
  2. WARM-UP (continued training; e5 has no LM head, so the signal on pool texts is the teacher's hidden states): WARM epochs of
     layer-wise normalised hidden-state MSE over all 13 hidden states (TernaryBERT, arXiv:2009.12812) + the output terms of 3;
  3. DISTILLATION FINE-TUNING: output terms = token-state (1 - cos) over real tokens + POOL_W x (1 - cos) of the mean-pooled
     vectors (exactly what the device reads) + AR_W x MiniLM-v2 multi-head self-attention RELATION distillation at the last layer
     (Q-Q, K-K, V-V scaled-dot relations, AR_HEADS relation heads, KL(teacher || student) over real query tokens; off by
     default, AR_W = 0) + HID_LATE x the hidden-state term.
  Teacher = the 4-bit e5 of the runtime (student_enc), frozen. Trained: latent weights, log-steps, offsets (AdamW, warm-up +
  cosine, no dropout: eval mode). Fixed-length cosine schedule; the best epoch by the 'stop' slice's output loss is kept.
  Fidelity = token / pooled cosine to the teacher's last layer on the 'val' slice (mean, p05, min, share >= .99, long/short) for
  the MSE-clipped round-to-nearest init, the GPTQ init and the trained model. Text split by sha256: 5% val, 5% stop, 90% train.
Data: pool texts only. Deterministic seeds (GPU float order aside). Device: EJ_DEVICE or auto (a GPU is strongly recommended:
30 epochs over about 21k texts). Output: <EDGE_CKPT>/lowbit-<key>/<variant>.pt (student_lbq.pack) + report-<variant>.json;
key = sha256 of runtime student_lbq.py + this file + runtime student_enc.py + the pool bytes. EJ_LOWBIT_SMOKE=1: a tiny smoke
run (writes lowbit-<key>-smoke/)."""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

from .env import device

SMOKE = os.environ.get('EJ_LOWBIT_SMOKE') == '1'
EPOCHS, WARM, PATIENCE = (2, 1, 2) if SMOKE else (30, 10, 10 ** 9)  # fixed-length cosine, no early stop, best epoch by stop slice
BUDGET, MAXB, SEED, CAL = 8192, 256, 0, 32 if SMOKE else 1024
LR, LR_S, LR_Z = 1e-4, 1e-3, 5e-4
POOL_W, HID_LATE, AR_W, AR_HEADS = 2.0, 1.0, 0.0, 12
DEV = torch.device(device())
E = Q = None  # runtime student_enc / student_lbq, bound by _rt() after ej.train.env.setup


def _rt():
    global E, Q
    import student_enc
    import student_lbq
    E, Q = student_enc, student_lbq


def _pool():
    return os.environ['EDGE_DATA'] + '/train/pool.jsonl'


def key():
    """16-hex key of the encoder checkpoint: runtime quantiser + this recipe + runtime e5 wrapper + pool bytes."""
    _rt()
    h = hashlib.sha256()
    for f in (Q.__file__, os.path.abspath(__file__), E.__file__):
        h.update(open(f, 'rb').read())
    with open(_pool(), 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def ck_name():
    """Directory name of the checkpoint (the value fit() gives the runtime's student_lb.CK_DIR)."""
    return f'lowbit-{key()}' + ('-smoke' if SMOKE else '')


def ckpt_dir():
    return os.path.join(os.environ['EDGE_CKPT'], ck_name())


def split(t):
    d = hashlib.sha256(t.encode()).digest()[0]
    return 'val' if d < 13 else 'stop' if d < 26 else 'train'


def pool_texts():
    import student as S
    import student_shallow as SH
    SH.N_TEXTS = 10 ** 9  # every distinct pool text the device encodes
    with open(_pool()) as f:
        return SH.texts_of(S.questions([json.loads(line) for line in f]))


def batches(lens, idx, rng=None):
    order = sorted(idx, key=lambda j: (lens[j], j))
    out, b = [], 0
    while b < len(order):
        e = b + 1
        while e < len(order) and e - b < MAXB and lens[order[e]] * (e - b + 1) <= BUDGET:
            e += 1
        out.append(order[b:e]); b = e
    if rng is not None:
        rng.shuffle(out)
    return out


def pad(seqs, ch):
    L = max(len(seqs[j]) for j in ch)
    x, m = torch.zeros(len(ch), L, dtype=torch.long), torch.zeros(len(ch), L, dtype=torch.long)
    for r, j in enumerate(ch):
        x[r, :len(seqs[j])] = torch.as_tensor(seqs[j]); m[r, :len(seqs[j])] = 1
    return x.to(DEV), m.to(DEV)  # [PAD] is id 0 in both vocabularies (specials are kept, ids stay sorted)


def D(h, Y, m, per=False):
    """(token 1 - cos over real tokens, pooled 1 - cos); per=True: per-text cosines instead."""
    f = m.unsqueeze(-1).to(h.dtype)
    tc = 1 - torch.nn.functional.cosine_similarity(h, Y, dim=-1)
    pc = 1 - torch.nn.functional.cosine_similarity((h * f).sum(1), (Y * f).sum(1), dim=-1)
    if per:
        return 1 - (tc * f[..., 0]).sum(1) / f[..., 0].sum(1), 1 - pc
    return (tc * f[..., 0]).sum() / f.sum(), pc.mean()


class Rel:
    """Forward hooks keeping the last layer's query / key / value projections (MiniLM-v2 relation distillation)."""

    def __init__(self, model):
        self.out = {}
        sa = model.encoder.layer[-1].attention.self
        for nm in ('query', 'key', 'value'):
            getattr(sa, nm).register_forward_hook(lambda mod, i, o, nm=nm: self.out.__setitem__(nm, o))


def rel_loss(ot, os_, m):
    B, T, Dm = ot['query'].shape
    H, dh = AR_HEADS, Dm // AR_HEADS
    bias = (1.0 - m[:, None, None, :].float()) * -1e4
    tot = 0.0
    for nm in ('query', 'key', 'value'):
        xt, xs = (o[nm].reshape(B, T, H, dh).transpose(1, 2) for o in (ot, os_))
        lt = torch.log_softmax(xt @ xt.transpose(-1, -2) / dh ** .5 + bias, -1)
        ls = torch.log_softmax(xs @ xs.transpose(-1, -2) / dh ** .5 + bias, -1)
        kl = torch.nn.functional.kl_div(ls, lt, log_target=True, reduction='none').sum(-1)
        tot = tot + (kl * m[:, None, :]).sum() / (m.sum() * H)
    return tot / 3


def forward(stu, teach, rs, rt, xt, xs, m, ar=False):
    """(output loss terms (tok, pool), hidden-state term, relation term (0 unless ar)) of one batch."""
    with torch.no_grad():
        Ht = teach(input_ids=xt, attention_mask=m, output_hidden_states=True).hidden_states
    Hs = stu(input_ids=xs, attention_mask=m, output_hidden_states=True).hidden_states
    mb = m.bool()
    hid = sum((hs - ht)[mb].pow(2).mean() / ht[mb].pow(2).mean() for hs, ht in zip(Hs, Ht)) / len(Hs)
    return D(Hs[-1], Ht[-1], m), hid, rel_loss(rt.out, rs.out, m) if ar else torch.zeros((), device=m.device)


def fidelity(stu, teach, full, new, idx):
    acc = {'tok': [], 'pool': [], 'n': []}
    with torch.no_grad():
        for ch in batches([len(x) for x in full], idx):
            xt, m = pad(full, ch); xs, _ = pad(new, ch)
            t, p = D(stu(input_ids=xs, attention_mask=m).last_hidden_state,
                     teach(input_ids=xt, attention_mask=m).last_hidden_state, m, per=True)
            acc['tok'] += t.tolist(); acc['pool'] += p.tolist(); acc['n'] += m.sum(1).tolist()
    pc, n = np.array(acc['pool']), np.array(acc['n'])
    r = lambda v: round(float(v), 5)
    return {'tok': r(np.mean(acc['tok'])), 'pool': r(pc.mean()), 'pool_p05': r(np.quantile(pc, .05)), 'pool_min': r(pc.min()),
            'frac_pool_ge_99': r((pc >= .99).mean()), 'pool_long': r(pc[n > 64].mean()) if (n > 64).any() else None,
            'pool_short': r(pc[n <= 64].mean()) if (n <= 64).any() else None, 'n': len(pc)}


def run(variant='w23'):
    """Train one variant ('w23' is the one the model reads) and write its checkpoint; returns the checkpoint directory."""
    _rt()
    t0 = time.time()
    log = lambda s: print(f'[{time.strftime("%H:%M:%S")}] {variant} {s}', file=sys.stderr, flush=True)
    torch.manual_seed(SEED); rng = random.Random(SEED)  # noqa: E702
    texts = pool_texts()
    tok = E._ENC[E.load(Q.MODEL, Q.BITS)][0]
    full = tok(texts, truncation=True, max_length=E.MAX_LEN)['input_ids']
    ids = Q.keep_ids(tok, texts); pos = {o: k for k, o in enumerate(ids)}
    new = [[pos[i] for i in x] for x in full]
    chk = Q.trimmed_tokenizer(tok, ids)(texts, truncation=True, max_length=E.MAX_LEN)['input_ids']
    sp = [split(t) for t in texts]
    tr, st, va = ([j for j, s in enumerate(sp) if s == w] for w in ('train', 'stop', 'val'))
    if SMOKE:
        tr, st, va = tr[:96], st[:32], va[:32]
    lens = [len(x) for x in full]
    rep = {'variant': variant, 'bits': Q.VARIANTS[variant], 'group': Q.GROUP, 'key': key(), 'n_texts': len(texts), 'n_vocab': len(ids),
           'tokenizer_mismatch': sum(a != b for a, b in zip(chk, new)), 'n_train': len(tr), 'n_stop': len(st), 'n_val': len(va),
           'tokens_train': int(sum(lens[j] for j in tr)), 'device': str(DEV), 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'hp': {'epochs': EPOCHS, 'warm': WARM, 'lr': [LR, LR_S, LR_Z], 'pool_w': POOL_W, 'hid_late': HID_LATE, 'ar_w': AR_W,
                  'budget': BUDGET, 'cal': CAL}, 'fid_val': {}}
    log(json.dumps(rep))
    teach = E._ENC[E.load(Q.MODEL, Q.BITS)][1].to(DEV).eval()
    stu = Q.build(ids, variant).to(DEV).eval()  # eval: no dropout; gradients still flow
    rt, rs = Rel(teach), Rel(stu)

    def stop_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for ch in batches(lens, st):
                xt, m = pad(full, ch); xs, _ = pad(new, ch)
                (tc, pc), _, _ = forward(stu, teach, rs, rt, xt, xs, m)
                tot += (tc + pc).item() * len(ch); n += len(ch)
        return round(tot / max(n, 1), 6)
    rep['fid_val']['rtn'] = fidelity(stu, teach, full, new, va); log(f'rtn {rep["fid_val"]["rtn"]}')
    snap, pre = {n: v.detach().cpu().clone() for n, v in stu.state_dict().items()}, stop_loss()
    cal = sorted(random.Random(SEED + 1).sample(tr, min(CAL, len(tr))))
    Q.gptq(stu, [pad(new, ch) for ch in batches(lens, cal)])
    rep['fid_val']['gptq'] = fidelity(stu, teach, full, new, va); log(f'gptq {rep["fid_val"]["gptq"]} {time.time() - t0:.0f}s')
    rep['init'] = {'rtn_stop_loss': pre, 'gptq_stop_loss': stop_loss()}
    rep['init']['used'] = 'gptq' if rep['init']['gptq_stop_loss'] <= pre else 'rtn'  # guard: keep the better init (stop slice)
    if rep['init']['used'] == 'rtn':
        stu.load_state_dict(snap)
    del snap
    isq = lambda n: n.endswith('.log_s') or n.endswith('.z')
    groups = [{'params': [p for n, p in stu.named_parameters() if not isq(n)], 'lr': LR},
              {'params': [p for n, p in stu.named_parameters() if n.endswith('.log_s')], 'lr': LR_S},
              {'params': [p for n, p in stu.named_parameters() if n.endswith('.z')], 'lr': LR_Z}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = EPOCHS * len(batches(lens, tr)); warm = min(200, steps // 10 + 1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))

    rep['stop_loss'] = [stop_loss()]
    best, bad, k = (rep['stop_loss'][0], {n: v.detach().cpu().clone() for n, v in stu.state_dict().items()}), 0, 0
    for ep in range(EPOCHS):
        late = ep >= WARM
        for ch in batches(lens, tr, rng):
            xt, m = pad(full, ch); xs, _ = pad(new, ch)
            (tc, pc), hid, ar = forward(stu, teach, rs, rt, xt, xs, m, late and AR_W > 0)
            loss = tc + POOL_W * pc + (HID_LATE if late else 1.0) * hid + (AR_W * ar if late else 0.0)
            opt.zero_grad(); loss.backward()  # noqa: E702
            torch.nn.utils.clip_grad_norm_(stu.parameters(), 1.0)
            opt.step(); sch.step(); k += 1  # noqa: E702
            if k % 100 == 0:
                log(f'ep {ep + 1} step {k}/{steps} loss {loss.item():.5f} tok {tc.item():.5f} pool {pc.item():.5f} hid {hid.item():.4f} '
                    f'ar {float(ar):.5f} {time.time() - t0:.0f}s')
        rep['stop_loss'].append(stop_loss()); log(f'ep {ep + 1} stop loss {rep["stop_loss"][-1]} {time.time() - t0:.0f}s')
        if not np.isfinite(rep['stop_loss'][-1]):
            break
        if rep['stop_loss'][-1] < best[0]:
            best, bad = (rep['stop_loss'][-1], {n: v.detach().cpu().clone() for n, v in stu.state_dict().items()}), 0
        elif late:
            bad += 1
            if bad >= PATIENCE:
                break
    stu.load_state_dict(best[1])
    rep['fid_val']['final'] = fidelity(stu, teach, full, new, va)
    rep.update(best_stop_loss=best[0], epochs_run=len(rep['stop_loss']) - 1, sec=round(time.time() - t0))
    ck = Q.pack(stu, ids, variant, rep)
    rep['size_mb'] = {k: round(v, 3) for k, v in Q.size_parts(ck).items()}; ck['rep'] = rep
    d = ckpt_dir(); os.makedirs(d, exist_ok=True)
    torch.save(ck, f'{d}/{variant}.pt.tmp'); os.replace(f'{d}/{variant}.pt.tmp', f'{d}/{variant}.pt')  # noqa: E702
    json.dump(rep, open(f'{d}/report-{variant}.json', 'w'), indent=1)
    log(f'final {json.dumps(rep)}')
    return d
