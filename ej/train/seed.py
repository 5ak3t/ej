"""Fit seed: ``python -m ej.train ... --seed S`` (S >= 1) re-seeds every RNG stream of an UNMODIFIED fit, so S fits of the same
code are independent replicates (fit-to-fit noise; the keep rule of ej.eval.compare). Seed 0 (the default) patches nothing:
the fit is bit-identical to the released path.
Why a process-level patch and not a code edit: the runtime modules are sha256-pinned and hashed into the state key, so the knob
cannot live in them. What it patches (before the runtime is imported):
  torch.manual_seed(x)            -> torch.manual_seed(mix(x, s))   (the global CPU generator: init, dropout masks, randn/randperm,
                                     shuffles)
  numpy.random.default_rng(x)     -> numpy.random.default_rng(mix(x, s))   (minibatch orders, augmentation draws; int seeds only)
  numpy.random.seed, random.seed, random.Random(x)   -> mixed likewise
mix(x, s) = SplitMix64(SplitMix64(s) ^ x) mod 2^63: every call site keeps its own, distinct stream, and changing s changes all
of them. torch.Generator().manual_seed(...) is NOT patched: those fixed draws are part of the model's predictive, not the fit's
randomness. Memoised training checkpoints are keyed by code and pool, not by seed, so each seed needs its own work directory
(ej.train refuses a work directory used with another seed); share the encoder checkpoint (lowbit-*) by copying it, which
makes the replicates conditional on the encoder."""
import os
import random

M64 = (1 << 64) - 1
_ORIG = {}


def splitmix64(x):
    """One SplitMix64 output for state x (Steele, Lea & Flood 2014)."""
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def seed():
    """The fit seed from EJ_SEED (0 = default path)."""
    s = os.environ.get('EJ_SEED', '').strip()
    return int(s) if s else 0


def mix(x, s):
    """The patched seed of call-site seed x under fit seed s (s = 0 -> x itself)."""
    return x if not s else splitmix64(splitmix64(s) ^ (int(x) & M64)) & ((1 << 63) - 1)


def install(s=None):
    """Patch the RNG entry points for fit seed s (default: EJ_SEED). Returns True if patched; s = 0 patches nothing."""
    s = seed() if s is None else s
    if not s:
        return False
    if _ORIG:
        assert _ORIG['seed'] == s, 'fitseed already installed with another seed'
        return True
    import numpy as np
    import torch
    _ORIG.update(seed=s, torch=torch.manual_seed, rng=np.random.default_rng, npseed=np.random.seed, Random=random.Random,
                 rseed=random.seed)
    isint = lambda x: isinstance(x, int) and not isinstance(x, bool) or type(x).__name__.startswith(('int', 'uint'))  # noqa: E731

    def manual_seed(x):
        return _ORIG['torch'](mix(x, s))

    def default_rng(x=None, *a, **k):
        return _ORIG['rng'](mix(x, s) if isint(x) else x, *a, **k)

    def npseed(x=None):
        return _ORIG['npseed'](mix(x, s) % 2 ** 32 if isint(x) else x)

    class Random(random.Random):
        def seed(self, a=None, version=2):
            super().seed(mix(a, s) if isint(a) else a, version)

    def rseed(a=None, version=2):
        return _ORIG['rseed'](mix(a, s) if isint(a) else a, version)
    torch.manual_seed, np.random.default_rng, np.random.seed = manual_seed, default_rng, npseed
    random.Random, random.seed = Random, rseed
    return True


def claim(work, s):
    """Record seed s in <work>/seed; refuse a work directory that already holds checkpoints of another seed."""
    p = os.path.join(work, 'seed')
    if os.path.exists(p):
        with open(p) as f:
            old = int(f.read().strip() or 0)
        if old != s:
            raise RuntimeError(f'{work} holds checkpoints fitted with seed {old}; use a separate --work for seed {s}')
        return
    used = os.path.isdir(os.path.join(work, 'ckpt')) and any(not n.startswith('lowbit-') for n in os.listdir(os.path.join(work, 'ckpt')))
    if used and s:
        raise RuntimeError(f'{work} already holds seed-0 checkpoints; use a separate --work for seed {s}')
    with open(p, 'w') as f:
        f.write(f'{s}\n')
