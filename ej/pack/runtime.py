"""INSTALL a read pack into ej's prediction runtime (ej/_runtime, unchanged): the runtime's predict path finds everything it
would otherwise load from files, so no pickle and no Hugging Face file is read.

  - student_lb._CK is pre-seeded with the encoder checkpoint at the exact path student_lb.checkpoint computes, as ej.safe.install
    does for a weights directory; the checkpoint is then slimmed to the fields the runtime still reads (variant / group / ids /
    rep), so the codes are not held twice.
  - student_lbq._BUILT is pre-seeded with (trimmed tokenizer, encoder): the encoder is built now from the pack (encoder.build;
    mode 'fp32' = default, 'stream' = low_memory=True) and runs under torch.inference_mode; the tokenizer is the pack's vocab.txt.
  - student_enc._ENC gets a stand-in for the base model whose only usable part is its config (the encoder width, which the
    runtime asks for when it embeds an empty list); any other use raises instead of downloading the base model.
  - the exact int8 head Linears of the state are swapped for dequantise-on-use ones (heads.py).
  - trim() hands freed heap pages back to the OS (glibc malloc_trim) after load and after each predict call."""
import os

from . import encoder as EN
from . import heads as HD

SLIM_KEYS = ('variant', 'group', 'ids', 'rep')


class NoBaseModel:
    """Stand-in for the base model in student_enc._ENC: exposes .config only; anything else raises."""

    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        raise RuntimeError(f'the packed ej model has no base model (asked for {name!r}); this code path is not part of '
                           'prediction from a pack')

    def __call__(self, *a, **k):
        self.__getattr__('__call__')


def base_key():
    """student_enc's cache key of the base model the runtime would load."""
    import student_lbq as Q
    return f'{Q.MODEL}#q{Q.BITS}'


def drop_standin():
    """Remove the base-model stand-in (before a weights directory, which builds from the real base model, is loaded)."""
    import student_enc as E
    v = E._ENC.get(base_key())
    if v is not None and isinstance(v[1], NoBaseModel):
        del E._ENC[base_key()]


def install(state, ck, tok_dir, ckpt_dir, mode='fp32', lru=0):
    """Make the runtime predict with this pack (module docstring). Returns the built encoder module."""
    from transformers import AutoConfig, BertTokenizerFast

    import student_enc as E
    import student_lb as LB
    import student_lbq as Q
    c = state['shallow']
    LB._CK.clear()
    LB._CK[os.path.join(os.path.abspath(ckpt_dir), c['dir'], f"{c['lb']}.pt")] = ck
    if LB.checkpoint(c) is not ck:
        raise RuntimeError('student_lb.checkpoint did not return the pre-seeded encoder')
    cfg = AutoConfig.from_pretrained(tok_dir)  # the pack's config.json (a local directory: nothing is downloaded)
    m = EN.build(ck, cfg, mode, lru)
    tok = BertTokenizerFast(os.path.join(tok_dir, 'vocab.txt'), do_lower_case=True)
    Q._BUILT.clear()
    Q._BUILT[Q.tag(ck)] = (tok, EN.Infer(m))
    if base_key() not in E._ENC:
        E._ENC[base_key()] = (None, NoBaseModel(cfg))
    for k in [k for k in ck if k not in SLIM_KEYS]:
        del ck[k]
    HD.install(state)
    return m


def trim():
    """glibc malloc_trim(0): hand freed heap pages (load-time buffers, last call's activations) back to the OS; False if absent."""
    try:
        import ctypes
        return bool(ctypes.CDLL('libc.so.6').malloc_trim(0))
    except (OSError, AttributeError):
        return False
