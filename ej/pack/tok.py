"""TOKENIZER FILES inside ejpack v1: the trimmed vocab.txt and the BERT config.json of the low-bit encoder, so loading and
predicting need NO Hugging Face file (no base-model download).

Pack side: files_from_base(ids) reads the pinned e5-small-v2 tokenizer + config (the only step that touches the base model's
files; no weights are read) and writes the kept pieces in checkpoint order; they ship as sections tok.vocab.txt.gz (gzip,
mtime 0) and tok.config.json. Load side: materialise() writes them to a directory under ej's cache named after their sha256
(re-used only when its files still hash right) and returns it."""
import gzip
import hashlib
import os
import tempfile

FILES = ('vocab.txt', 'config.json')
SECTIONS = ('tok.vocab.txt.gz', 'tok.config.json')


def files_from_base(ids, model, revision):
    """{'vocab.txt': bytes, 'config.json': bytes} for the kept token ids (pack side; reads the base tokenizer and config)."""
    from transformers import AutoConfig, AutoTokenizer
    inv = {i: t for t, i in AutoTokenizer.from_pretrained(model, revision=revision).get_vocab().items()}
    vocab = ('\n'.join(inv[i] for i in ids) + '\n').encode()
    with tempfile.TemporaryDirectory() as d:
        AutoConfig.from_pretrained(model, revision=revision).save_pretrained(d)
        with open(os.path.join(d, 'config.json'), 'rb') as f:
            config = f.read()
    return {'vocab.txt': vocab, 'config.json': config}


def sections(fs, n_ids):
    """The tok.* sections of files fs; refuses a vocabulary whose line count differs from the kept id count."""
    n = fs['vocab.txt'].count(b'\n')
    if n != n_ids:
        raise ValueError(f'vocab.txt has {n} lines for {n_ids} kept ids')
    return {'tok.vocab.txt.gz': gzip.compress(fs['vocab.txt'], 9, mtime=0), 'tok.config.json': fs['config.json']}


def files(secs):
    """{'vocab.txt': bytes, 'config.json': bytes} from the tok.* sections."""
    return {'vocab.txt': gzip.decompress(bytes(secs['tok.vocab.txt.gz'])), 'config.json': bytes(secs['tok.config.json'])}


def _read(p):
    with open(p, 'rb') as f:
        return f.read()


def _ok(d, fs):
    return all(os.path.exists(os.path.join(d, n)) and _read(os.path.join(d, n)) == b for n, b in fs.items())


def materialise(secs, root):
    """Directory <root>/ejpack-tok-<sha> holding vocab.txt + config.json of the package; written atomically."""
    fs = files(secs)
    tag = hashlib.sha256(b''.join(hashlib.sha256(fs[n]).digest() for n in FILES)).hexdigest()[:16]
    d = os.path.join(root, f'ejpack-tok-{tag}')
    if _ok(d, fs):
        return d
    os.makedirs(d, exist_ok=True)
    for n, b in fs.items():
        tmp = os.path.join(d, f'.{n}.{os.getpid()}')
        with open(tmp, 'wb') as f:
            f.write(b)
        os.replace(tmp, os.path.join(d, n))
    if not _ok(d, fs):
        raise OSError(f'{d}: tokenizer files did not verify after writing')
    return d
