"""BIT PACKING of b-bit unsigned codes (b in 1..8) for ejpack v1.

Code i of a tensor occupies bits [i*b, (i+1)*b) of a little-endian bit stream (bit 0 = least significant bit of byte 0);
each tensor's stream is padded with zero bits to a whole byte. 2-bit codes take n/4 bytes, 3-bit codes 3n/8 bytes."""
import numpy as np


def nbytes(n, bits):
    """Bytes of the packed stream of n codes at `bits` bits each."""
    return (n * bits + 7) // 8


def pack(codes, bits):
    """bytes of the packed stream of codes (any integer array-like, values in [0, 2**bits)); raises on an out-of-range code."""
    q = np.ascontiguousarray(np.asarray(codes).reshape(-1)).astype(np.uint8)
    if not 1 <= bits <= 8:
        raise ValueError(f'bits must be in 1..8, got {bits}')
    if q.size and int(q.max()) >= 1 << bits:
        raise ValueError(f'code {int(q.max())} does not fit in {bits} bits')
    planes = ((q[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder='little').tobytes()


def unpack(buf, bits, n):
    """uint8 numpy array of the n codes stored in buf (inverse of pack)."""
    raw = np.frombuffer(buf, dtype=np.uint8)
    if raw.size != nbytes(n, bits):
        raise ValueError(f'{raw.size} bytes for {n} codes at {bits} bits, expected {nbytes(n, bits)}')
    planes = np.unpackbits(raw, bitorder='little')[:n * bits].reshape(n, bits)
    out = np.zeros(n, dtype=np.uint8)
    for b in range(bits):
        out |= planes[:, b] << np.uint8(b)
    return out
