"""A SMALL synthetic ejpack (no weights needed): a 1-layer BERT encoder with random 2/3-bit codes, int8-exact head and cross
tensors, hashed vocabularies and a toy tokenizer. Used by tests/test_pack.py; the file is about 0.1 MB."""
import json

import torch

WORDS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list('abcdefghijklmnopqrstuvwxyz') + [
    'the', 'order', 'refund', 'late', 'charge', 'box', 'item', 'help', 'please']
SLOTS = ('0123456789abcdef', '89abcdef01234567', 'fedcba9876543210')
GROUP = 128


def config():
    """A tiny BertConfig over the toy vocabulary."""
    from transformers import BertConfig
    return BertConfig(vocab_size=len(WORDS), hidden_size=128, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=256, max_position_embeddings=32)


def _int8_exact(rows, cols, g):
    """A float32 matrix that is exactly int8 * fp16 row scale (every row reaches |q| = 127)."""
    q = torch.randint(-127, 128, (rows, cols), generator=g).float()
    q[:, 0] = 127.0
    s = (torch.rand(rows, generator=g) * 0.01 + 1e-3).half().float()
    return q * s[:, None]


def checkpoint(seed=0):
    """A low-bit encoder checkpoint for config(): codes for every Linear / word + position embedding, fp16 'other'."""
    from ej.pack.encoder import meta_shell
    g = torch.Generator().manual_seed(seed)
    cfg = config()
    m = meta_shell(cfg)
    codes, other = {}, {}
    for n, p in m.named_parameters():
        quantised = n.endswith('.weight') and p.dim() == 2 and 'token_type' not in n
        if quantised:
            bits = 3 if 'attention' in n and 'LayerNorm' not in n else 2
            groups = p.numel() // GROUP
            codes[n] = {'q': torch.randint(0, 1 << bits, (groups, GROUP), generator=g, dtype=torch.uint8),
                        's': (torch.rand(groups, 1, generator=g) * 0.02 + 1e-3).half(),
                        'z': torch.randint(0, 1 << bits, (groups, 1), generator=g).half(), 'bits': bits, 'shape': tuple(p.shape)}
        else:
            other[n] = (torch.randn(p.shape, generator=g) * 0.02 + (1.0 if 'LayerNorm.weight' in n else 0.0)).half()
    return {'variant': 'w23', 'group': GROUP, 'ids': list(range(len(WORDS))), 'codes': codes, 'other': other,
            'rep': {'key': 'synthetic'}}


def vocab(n_per_slot, offset=0):
    """A (slot, token) -> column dict with contiguous slots (as the runtime builds them: sorted keys)."""
    keys = sorted((s, f'tok{offset + i}') for s in SLOTS for i in range(n_per_slot))
    return {k: i for i, k in enumerate(keys)}


def state(seed=0):
    """A small 'fitted state': hashed vocabularies, an int8-exact head, an int8-exact cross vector, misc tensors."""
    g = torch.Generator().manual_seed(seed + 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)  # the biases (the global RNG is restored afterwards)
        head = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.GELU(), torch.nn.Linear(8, 3))
    with torch.no_grad():
        head[0].weight.copy_(_int8_exact(8, 16, g))
        head[2].weight.copy_(_int8_exact(3, 8, g))
    w = _int8_exact(3, 64, g).reshape(-1)[:150].contiguous()
    return {'shallow': {'lb': 'w23', 'dir': 'lowbit-synthetic'}, 'voc': vocab(40), 'rvoc': vocab(10, 100), 'rich': head,
            'w': w, 'misc': {'f32': torch.randn(5, generator=g), 'f16': torch.randn(70, generator=g).half().float(),
                             'ints': torch.arange(7), 'flag': True, 'name': 'synthetic'}}


def payload(key='5e5e' * 16):
    """The state metadata a pack carries (no 'state')."""
    return {'version': 'synthetic-state-v1', 'key': key, 'meta': {'saved': 'synthetic'}, 'dropped': [], 'notes': {}}


def tok_files():
    """vocab.txt + config.json of the toy tokenizer / encoder."""
    cfg = json.loads(config().to_json_string())
    return {'vocab.txt': ('\n'.join(WORDS) + '\n').encode(), 'config.json': json.dumps(cfg, indent=2).encode()}


def write(path, seed=0, key=None):
    """Write the synthetic pack to path; returns (writer report, state, ck)."""
    from ej.pack import write as W
    st, ck = state(seed), checkpoint(seed)
    rep = W.write(str(path), payload(key) if key else payload(), st, ck, tok_files())
    return rep, st, ck
