"""HASHED CROSS VOCABULARY for ejpack v1: the (slot, token) -> column dicts state['voc'] / state['rvoc'] as binary keys.

Stored (per vocabulary): the key hash h(k) = sha256(repr(k).encode())[:W] (the hash student_lb counts for the model size) of
every key in column order (n x W bytes), with W the smallest width >= 5 bytes (40 bits) that is collision-free over the keys
(checked at pack time), plus the slot table: the distinct slots (16-hex strings -> 8 raw bytes each) in column order and the
uint32 run length of each (the dicts are built from sorted keys, so a slot's columns are contiguous; checked). Token strings
are NOT stored. Lookup (HashedVocab, what student_wide.bags / student_rel.bags call: `k in voc`, `voc[k]`, `{s for s, _ in
voc}`): hash the queried key, find the column (sorted uint64 keys + int32 columns, 12 bytes per key), and accept it only if
that column's slot equals the queried slot. A key outside the training vocabulary is mistaken for one inside only if its
W-byte hash equals the hash of a key with the same slot: probability (keys of that slot) / 2**(8W) per lookup
(false_hit_bound(); about 7e-9 for the largest 0.0.1 vocabulary)."""
import hashlib
import struct

import numpy as np

MIN_WIDTH = 5


def key_hash(k, width):
    """The first width bytes of sha256(repr(k)) (student_lb.cross_mb's 40-bit key at width 5)."""
    return hashlib.sha256(repr(k).encode()).digest()[:width]


def width(keys, lo=MIN_WIDTH):
    """Smallest width >= lo bytes whose hashes are pairwise distinct over keys."""
    full = [hashlib.sha256(repr(k).encode()).digest() for k in keys]
    for w in range(lo, 33):
        if len({d[:w] for d in full}) == len(full):
            return w
    raise ValueError('sha256 collision over the vocabulary')


def _check(voc):
    keys = list(voc)
    if [voc[k] for k in keys] != list(range(len(keys))):
        raise ValueError('vocabulary values are not 0..n-1 in key order')
    for k in keys:
        if not (type(k) is tuple and len(k) == 2 and type(k[0]) is str and type(k[1]) is str and len(k[0]) == 16):
            raise ValueError(f'unsupported key {k!r}')
        bytes.fromhex(k[0])
    return keys


def encode(voc):
    """{'hash', 'slots', 'runs': bytes, 'width', 'n'} of a (slot, token) -> column dict (see the module docstring)."""
    keys = _check(voc)
    w = width(keys)
    slots, runs = [], []
    for s, _ in keys:
        if slots and slots[-1] == s:
            runs[-1] += 1
        else:
            slots.append(s)
            runs.append(1)
    if len(set(slots)) != len(slots):
        raise ValueError('slot columns are not contiguous')
    return {'hash': b''.join(key_hash(k, w) for k in keys), 'slots': b''.join(bytes.fromhex(s) for s in slots),
            'runs': struct.pack(f'<{len(runs)}I', *runs), 'width': w, 'n': len(keys)}


class HashedVocab:
    """Read-only stand-in for a (slot, token) -> column dict, backed by the binary sections of encode()."""

    def __init__(self, hashes, slots, runs, width, n):
        self.width, self.n = int(width), int(n)
        hashes = bytes(hashes)
        if len(hashes) != self.width * self.n:
            raise ValueError('hash section length does not match width x n')
        self.slot_names = [bytes(slots[i:i + 8]).hex() for i in range(0, len(slots), 8)]
        self.runs = list(struct.unpack(f'<{len(runs) // 4}I', bytes(runs)))
        if len(self.runs) != len(self.slot_names) or sum(self.runs) != self.n:
            raise ValueError('slot table does not cover the vocabulary')
        self._slot_code = {s: i for i, s in enumerate(self.slot_names)}
        self._slot_of = np.repeat(np.arange(len(self.runs), dtype=np.int32), self.runs)
        w = self.width
        if w > 8:
            raise ValueError(f'key width {w} > 8 bytes')
        b = np.frombuffer(hashes, dtype=np.uint8).reshape(self.n, w).astype(np.uint64)
        keys = np.zeros(self.n, dtype=np.uint64)
        for j in range(w):  # big-endian integer of each key's w bytes
            keys = (keys << np.uint64(8)) | b[:, j]
        order = np.argsort(keys, kind='stable')
        self._keys, self._cols = keys[order], order.astype(np.int32)  # sorted keys + their columns (12 B / key; a dict is ~100)
        if self.n > 1 and bool((self._keys[1:] == self._keys[:-1]).any()):
            raise ValueError('duplicate key hashes')
        self._last = (None, None)

    def _column(self, h):
        """Column whose key hash is the integer h, or None."""
        h = np.uint64(h)
        i = int(np.searchsorted(self._keys, h))
        return int(self._cols[i]) if i < self.n and self._keys[i] == h else None

    def find(self, k):
        """Column of key k = (slot, token), or None when k is not in the vocabulary."""
        if self._last[0] == k:
            return self._last[1]
        code = self._slot_code.get(k[0]) if type(k) is tuple and len(k) == 2 else None
        col = None
        if code is not None:
            c = self._column(int.from_bytes(key_hash(k, self.width), 'big'))
            col = c if c is not None and self._slot_of[c] == code else None
        self._last = (k, col)
        return col

    def __contains__(self, k):
        return self.find(k) is not None

    def __getitem__(self, k):
        c = self.find(k)
        if c is None:
            raise KeyError(k)
        return c

    def get(self, k, default=None):
        """Column of k, or default."""
        c = self.find(k)
        return default if c is None else c

    def __len__(self):
        return self.n

    def __iter__(self):
        """(slot, column) per column, in column order (the token is not stored; `{s for s, _ in voc}` gives the slot set)."""
        col = 0
        for s, r in zip(self.slot_names, self.runs):
            for _ in range(r):
                yield (s, col)
                col += 1

    def slots(self):
        """The set of slots."""
        return set(self.slot_names)

    def false_hit_bound(self):
        """Upper bound on P(an out-of-vocabulary key of a known slot is taken as in-vocabulary), per lookup."""
        return max(self.runs) / 2 ** (8 * self.width)
