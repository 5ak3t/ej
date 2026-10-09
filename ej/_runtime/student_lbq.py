"""Low-bit encoder, model side: e5-small-v2 over a trimmed vocabulary (dropped pieces are re-split, never mapped to unknown)
with 2- or 3-bit grouped weights (learned step and offset per group, GPTQ initialisation); building, packing and unpacking,
and the trimmed tokenizer, whose vocabulary is written to $EDGE_CACHE."""
import hashlib
import os

import torch

import student_enc as E

MODEL, BITS = 'intfloat/e5-small-v2', 4  # the teacher = the shipped 4-bit encoder (student_enc)
GROUP, S_MIN, N_GENERAL = 128, 1e-4, 5000
VARIANTS = {'w2': {'emb': 2, 'attn': 2, 'ffn': 2}, 'w23': {'emb': 2, 'attn': 3, 'ffn': 2}}
_BUILT = {}


def keep_ids(tok, texts):
    """A-priori kept vocabulary: specials, single characters (+ '##' forms), first N_GENERAL ids, every id of the pool texts."""
    v = tok.get_vocab()
    keep = {i for t, i in v.items() if t in tok.all_special_tokens or len(t) == 1 or (t.startswith('##') and len(t) == 3)}
    keep |= {i for i in range(N_GENERAL) if not tok.convert_ids_to_tokens(i).startswith('[unused')}
    for b in range(0, len(texts), 4096):
        keep |= {i for x in tok(texts[b:b + 4096], truncation=True, max_length=E.MAX_LEN)['input_ids'] for i in x}
    return sorted(keep)


def trimmed_tokenizer(tok, ids):
    """BertTokenizerFast over the kept pieces (original order; new id k = old id ids[k])."""
    from transformers import BertTokenizerFast
    d = f'{E.CACHE}/lb-vocab-{hashlib.sha256(str(ids).encode()).hexdigest()[:12]}'
    os.makedirs(d, exist_ok=True)
    inv = {i: t for t, i in tok.get_vocab().items()}
    if not os.path.exists(f'{d}/vocab.txt'):
        with open(f'{d}/vocab.txt.{os.getpid()}', 'w') as f:
            f.write('\n'.join(inv[i] for i in ids) + '\n')
        os.replace(f'{d}/vocab.txt.{os.getpid()}', f'{d}/vocab.txt')
    return BertTokenizerFast(f'{d}/vocab.txt', do_lower_case=True)


def quantised(name, mod):
    return isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and 'token_type' not in name


def bits_of(name, variant):
    v = VARIANTS[variant]
    return v['emb'] if 'embeddings' in name else v['attn'] if 'attention' in name else v['ffn']


def step(log_s):
    return log_s.exp().clamp(min=S_MIN).half().float()


def init_sz(w, bits):
    """(s, z) per group by an MSE-optimal clipping search of the group range (init only)."""
    g = w.detach().reshape(-1, GROUP).float(); lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
    best = None
    for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5):
        s = ((hi - lo) * c / (2 ** bits - 1)).clamp(min=S_MIN).half().float()
        z = torch.round(-((lo + hi) / 2 - c * (hi - lo) / 2) / s)
        e = ((torch.clamp(torch.round(g / s + z), 0, 2 ** bits - 1) - z) * s - g).pow(2).sum(1, keepdim=True)
        best = (e, s, z) if best is None else (torch.minimum(best[0], e), torch.where(e < best[0], s, best[1]),
                                               torch.where(e < best[0], z, best[2]))
    return best[1], best[2]


class FQ(torch.nn.Module):
    """Learnable asymmetric per-group fake quantiser (a parametrisation of a weight); straight-through round on w and z."""

    def __init__(self, w, bits):
        super().__init__()
        self.bits = bits
        s, z = init_sz(w, bits)
        self.log_s, self.z = torch.nn.Parameter(s.log()), torch.nn.Parameter(z)

    def set_sz(self, s, z):
        with torch.no_grad():
            self.log_s.copy_(s.clamp(min=S_MIN).log()); self.z.copy_(z)  # noqa: E702

    def codes(self, w):
        """(uint8 codes (groups, GROUP), s fp16 (groups, 1), round(z) fp16) -- exactly what forward() uses."""
        s, z = step(self.log_s), torch.round(self.z)
        q = torch.clamp(torch.round(w.detach().reshape(-1, GROUP) / s + z), 0, 2 ** self.bits - 1)
        return q.to(torch.uint8), s.half(), z.half()

    def forward(self, w):
        g, s = w.reshape(-1, GROUP), step(self.log_s)
        z = self.z + (torch.round(self.z) - self.z).detach()
        x = g / s + z
        q = torch.clamp(x + (torch.round(x) - x).detach(), 0, 2 ** self.bits - 1)
        return ((q - z) * s).reshape(w.shape)


class _Stop(Exception):
    pass


def _gptq(W, H, bits, damp=0.01):
    """GPTQ of one (rows, n) matrix with per-group (s, z) from init_sz on the error-updated weights: (latent, s, z)."""
    G = GROUP
    W = W.detach().clone().float(); n = W.shape[1]; qmax = 2 ** bits - 1
    H = H.clone(); dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    S, Z, lat = torch.zeros(W.shape[0], n // G, device=W.device), torch.zeros(W.shape[0], n // G, device=W.device), torch.zeros_like(W)
    for i in range(n):
        if i % G == 0:
            s, z = init_sz(W[:, i:i + G].contiguous(), bits)
            S[:, i // G], Z[:, i // G] = s[:, 0], z[:, 0]
        s, z, w = S[:, i // G], Z[:, i // G], W[:, i]
        q = (torch.clamp(torch.round(w / s + z), 0, qmax) - z) * s
        lat[:, i] = w
        W[:, i + 1:] -= ((w - q) / Hinv[i, i])[:, None] * Hinv[i, i + 1:][None, :]
    return lat, S.reshape(-1, 1), Z.reshape(-1, 1)


def gptq(stu, cal):
    """Quantise every encoder Linear of stu in place, layer by layer (inputs from the already-quantised layers below)."""
    mask = {}
    for L in stu.encoder.layer:
        lin = [(n, m) for n, m in L.named_modules() if isinstance(m, torch.nn.Linear)]
        H = {n: torch.zeros(m.in_features, m.in_features, device=m.parametrizations.weight.original.device) for n, m in lin}
        cnt = {n: 0 for n, _ in lin}

        def hook(n, x):
            x = x[mask['m']].float(); H[n] += 2 * x.T @ x; cnt[n] += len(x)  # noqa: E702

        def stop(mod, i, o):
            raise _Stop
        hs = [m.register_forward_hook(lambda mod, i, o, n=n: hook(n, i[0])) for n, m in lin] + [L.register_forward_hook(stop)]
        with torch.no_grad():
            for x, m in cal:
                mask['m'] = m.bool()
                try:
                    stu(input_ids=x, attention_mask=m)
                except _Stop:
                    pass
        for h in hs:
            h.remove()
        with torch.no_grad():
            for n, m in lin:
                p = m.parametrizations.weight
                lat, s, z = _gptq(p.original, H[n] / cnt[n], p[0].bits)
                p.original.copy_(lat); p[0].set_sz(s, z)  # noqa: E702


def _base(ids):
    """fp32 e5-small-v2 (no pooler) with the word embedding cut to the kept ids."""
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL, add_pooling_layer=False).float()
    w = m.embeddings.word_embeddings.weight.detach()[torch.as_tensor(ids)].clone()
    m.embeddings.word_embeddings = torch.nn.Embedding.from_pretrained(w, freeze=False)
    return m


def build(ids, variant):
    """Training student: the original fp32 weights over the trimmed vocabulary, every quantised weight parametrised by FQ."""
    import torch.nn.utils.parametrize as PZ
    m = _base(ids)
    for n, mod in list(m.named_modules()):
        if quantised(n, mod):
            PZ.register_parametrization(mod, 'weight', FQ(mod.weight, bits_of(n, variant)))
    return m


def pack(stu, ids, variant, rep):
    """Compact checkpoint of a trained student (see the module docstring)."""
    import torch.nn.utils.parametrize as PZ
    codes, other = {}, {}
    for n, mod in stu.named_modules():
        if isinstance(mod, (FQ, PZ.ParametrizationList)):
            continue
        if PZ.is_parametrized(mod, 'weight'):
            q, s, z = mod.parametrizations.weight[0].codes(mod.parametrizations.weight.original)
            codes[f'{n}.weight'] = {'q': q.cpu(), 's': s.cpu(), 'z': z.cpu(), 'bits': mod.parametrizations.weight[0].bits,
                                    'shape': tuple(mod.parametrizations.weight.original.shape)}
        for pn, p in mod.named_parameters(recurse=False):
            other[f'{n}.{pn}' if n else pn] = p.detach().half().cpu()
    return {'variant': variant, 'group': GROUP, 'ids': list(ids), 'codes': codes, 'other': other, 'rep': rep}


def dequant(c):
    return ((c['q'].float() - c['z'].float()) * c['s'].float()).reshape(c['shape'])


def unpack(ck):
    """The device encoder of a compact checkpoint (eval mode, fp32 compute on the dequantised weights)."""
    m = _base(ck['ids'])
    names = {n for n, _ in m.named_parameters()}
    assert names == set(ck['codes']) | set(ck['other']), sorted(names ^ (set(ck['codes']) | set(ck['other'])))[:5]
    with torch.no_grad():
        for n, c in ck['codes'].items():
            m.get_parameter(n).copy_(dequant(c))
        for n, v in ck['other'].items():
            m.get_parameter(n).copy_(v.float())
    for p in m.parameters():
        p.requires_grad = False
    return m.eval()


def tag(ck):
    return f"lb-{ck['variant']}-{ck['rep'].get('key', '?')}"


def encoder(ck):
    """(trimmed tokenizer, fn(input_ids, mask) -> last-layer token states, tag) for student_fast.begin; built once per process."""
    t = tag(ck)
    if t not in _BUILT:
        _BUILT.clear()
        _BUILT[t] = (trimmed_tokenizer(E._ENC[E.load(MODEL, BITS)][0], ck['ids']), unpack(ck))
    tok, m = _BUILT[t]

    def fn(i, a):
        dev = next(m.parameters()).device
        with torch.no_grad():
            return m(input_ids=i.to(dev), attention_mask=a.to(dev)).last_hidden_state.float().cpu()
    return tok, fn, t


def size_parts(ck):
    """{'encoder': MB of weights, 'vocab': MB of the kept vocabulary strings}."""
    bits = sum(c['q'].numel() * (c['bits'] + 32 / GROUP) for c in ck['codes'].values())
    bits += sum(v.numel() * 16 for v in ck['other'].values())
    inv = {i: t for t, i in E._ENC[E.load(MODEL, BITS)][0].get_vocab().items()}
    return {'encoder': bits / 8 / 2 ** 20, 'vocab': sum(len(inv[i].encode()) + 1 for i in ck['ids']) / 2 ** 20}


def _test():
    """Pack/unpack round trip reproduces the fake-quantised forward exactly; trimmed tokenizer = full one on kept-piece texts."""
    tok = E._ENC[E.load(MODEL, BITS)][0]
    texts = ['query: {"invoice_id": "A-17", "total": 12.5}', 'passage: refund the customer', 'query: hi there']
    ids = keep_ids(tok, texts)
    for v in VARIANTS:
        stu = build(ids, v).eval()
        ck = pack(stu, ids, v, {'key': 'test'})
        dev = unpack(ck)
        ttok = trimmed_tokenizer(tok, ids)
        b = ttok(texts, padding=True, return_tensors='pt')
        full = tok(texts, padding=True, return_tensors='pt')['input_ids']
        assert torch.equal(torch.as_tensor(ids)[b['input_ids']], full), 'trimmed tokenizer differs on pool-piece texts'
        with torch.no_grad():
            a, c = stu(**b).last_hidden_state, dev(**b).last_hidden_state
        sp = size_parts(ck)
        print(v, 'max |stu - device|', float((a - c).abs().max()), 'MB', {k: round(x, 3) for k, x in sp.items()}, len(ids), 'ids')
        assert float((a - c).abs().max()) < 1e-3
    print('student_lbq ok')


if __name__ == '__main__':
    _test()
