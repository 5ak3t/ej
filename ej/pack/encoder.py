"""ENCODER BUILD from the package alone: the eval-mode low-bit e5 encoder of a checkpoint, with the BERT config shipped in the
pack (no pretrained weight is read, nothing is downloaded).

The shell is BertModel(config) (no pooler) created on the meta device (no allocation, no random init, no RNG use) with the word
embedding cut to the kept vocabulary; the fp16 'other' parameters (LayerNorms, biases, token-type embedding) are set as fp32,
and every quantised weight is either
  'fp32'    dequantised once at load (the same values as student_lbq.dequant, so the same fp32 weights as the safetensors path),
  'stream'  kept as its 2/3-bit codes inside a lowmem.StreamLinear / StreamEmbedding and dequantised inside forward.
Both modes compute the same fp32 weights with the same arithmetic, so predictions are bit-identical (tests/test_pack.py)."""
import torch

from . import lowmem as LM

MODES = ('fp32', 'stream')


class Infer(torch.nn.Module):
    """Runs the wrapped encoder under torch.inference_mode (exposes .config like the wrapped BertModel)."""

    def __init__(self, m):
        super().__init__()
        self.m, self.config = m, m.config

    def forward(self, **kw):
        with torch.inference_mode():
            return self.m(**kw)


def meta_shell(cfg):
    """BertModel(cfg) (no pooler) with every tensor on the meta device: no allocation, no RNG use."""
    from transformers import BertModel
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        return BertModel(cfg, add_pooling_layer=False)


def _set_param(m, name, value):
    mod, attr = name.rsplit('.', 1)
    m.get_submodule(mod)._parameters[attr] = torch.nn.Parameter(value, requires_grad=False)


def _set_module(m, name, new):
    parent, attr = name.rsplit('.', 1)
    setattr(m.get_submodule(parent), attr, new)


def _buffers(m, cfg):
    """The two non-persistent BertEmbeddings buffers, exactly as its __init__ creates them."""
    e = m.embeddings
    e.position_ids = torch.arange(cfg.max_position_embeddings).expand((1, -1))
    e.token_type_ids = torch.zeros(e.position_ids.size(), dtype=torch.long)


def build(ck, cfg, mode='fp32', lru=0, policy='pin'):
    """The eval-mode encoder of checkpoint ck (code entries with 'q' = unpacked uint8 codes, or 'raw' = packed codes from
    codes.decode(unpack=False)) under BertConfig cfg. mode 'fp32': resident dequantised weights; 'stream': stream modules
    holding the packed codes (lru: dequantised Linear weights kept, 0 = none)."""
    if mode not in MODES:
        raise ValueError(f'encoder mode must be one of {MODES}, got {mode!r}')
    m = meta_shell(cfg)
    with torch.device('meta'):
        m.embeddings.word_embeddings = torch.nn.Embedding(len(ck['ids']), cfg.hidden_size)  # no padding_idx, as trained
    names = {n for n, _ in m.named_parameters()}
    have = set(ck['codes']) | set(ck['other'])
    if names != have:
        raise ValueError(f'encoder checkpoint does not match the config: {sorted(names ^ have)[:5]}')
    for n, v in ck['other'].items():
        _set_param(m, n, v.float())
    cache = LM.LRU(lru, policy) if mode == 'stream' else None
    for n, c in ck['codes'].items():
        raw = c['raw'] if 'raw' in c else LM.pack_q(c['q'], c['bits'])
        mod = n.rsplit('.', 1)[0]
        old = m.get_submodule(mod)
        kind = 'embedding' if isinstance(old, torch.nn.Embedding) else 'linear'
        new = LM.from_entry(c, raw, ck['group'], kind, bias=getattr(old, 'bias', None), lru=cache)
        if mode == 'fp32':
            _set_param(m, n, new.deq().detach())
        else:
            _set_module(m, mod, new)
    _buffers(m, cfg)
    m.lru = cache
    left = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f'meta tensors left after the encoder build: {left[:5]}')
    return m.eval()
