"""ejpack v1: the weights format of ej -- ONE file, model.ejpack, holding the fitted state, the 2/3-bit encoder codes, the int8
heads, the hashed cross vocabularies and the trimmed tokenizer + encoder config. Pickle-free; the header and every section are
sha256-checked before anything is decoded; nothing is downloaded at load (no base model).

  ej.load('model.ejpack')                      what users call (ej.pack.load.load_pack)
  python -m ej.train export --weights DIR --out model.ejpack     writes a pack from a weights directory (write.from_weights_dir)

Modules: container (file layout, digests), bits (bit packing), quant (exact int8 / fp16 recovery), codes (encoder codes), hvocab
(hashed cross keys), table (object graph through ej's pickle-free codec + per-tensor storage), tok (tokenizer files), write
(writer), read (pickle-free reader), encoder (encoder build from the pack), lowmem (streaming dequantisation, low_memory=True),
heads (int8 heads dequantised on use), runtime (install into ej/_runtime), load (verify + read + install)."""
from .read import NAME, is_pack, locate  # noqa: F401
