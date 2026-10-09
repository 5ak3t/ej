"""Working environment of a training run: where the pool, the checkpoints and the caches live, set up BEFORE the runtime is
imported (several runtime modules read their paths at import time).

Layout of a work directory (default for fit: <out>.work; keep it out of any git work tree you commit):
  <work>/data/train/pool.jsonl   a copy of the training pool (the runtime reads the pool from here)
  <work>/ckpt/                   training checkpoints: low-bit encoder, decision-encoder teachers, pair teachers, memos
  <work>/cache/                  encoder embedding cache and the trimmed vocabulary
The runtime's internal names (EDGE_DATA, EDGE_CKPT, EDGE_CACHE) are set from these paths; nothing else of the process changes.
Public overrides: EJ_DEVICE (cuda / cpu, for the encoder distillation), EJ_THREADS (torch threads, default 2)."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(os.path.dirname(HERE), '_runtime')
_DONE = {}


def device():
    """'cuda' when EJ_DEVICE asks for it or a GPU is visible (and EJ_DEVICE is not 'cpu'), else 'cpu'."""
    import torch
    want = os.environ.get('EJ_DEVICE', '').lower()
    if want in ('cpu', 'cuda'):
        return want
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def setup(pool, work):
    """Prepare <work> for `pool` (copied when it differs) and point the runtime at it. Returns the work directory.
    Must run before the first runtime import; a second call with other paths in the same process is refused."""
    work = os.path.abspath(work)
    if _DONE and _DONE['work'] != work:
        raise RuntimeError(f'training environment already set up for {_DONE["work"]}; use one work directory per process')
    dst = os.path.join(work, 'data', 'train', 'pool.jsonl')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if pool and (not os.path.exists(dst) or not _same(pool, dst)):
        shutil.copyfile(pool, dst)
    if not os.path.exists(dst):
        raise FileNotFoundError(f'no training pool: pass --pool (expected {dst})')
    for sub in ('ckpt', 'cache'):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    os.environ['EDGE_DATA'] = os.path.join(work, 'data')
    os.environ['EDGE_CKPT'] = os.path.join(work, 'ckpt')
    os.environ['EDGE_CACHE'] = os.environ.get('EJ_CACHE') or os.path.join(work, 'cache')
    os.environ['EDGE_STATE'] = '0'  # always fit; the result is written as a pickle-free weights directory (ej.train.export)
    if RUNTIME not in sys.path:
        sys.path.insert(0, RUNTIME)
    register()
    _DONE['work'] = work
    return work


def register():
    """Make the training modules importable under the bare names the runtime's fit code imports (train_hcf, student_hdn)."""
    from . import hcf, hdn
    sys.modules.setdefault('train_hcf', hcf)
    sys.modules.setdefault('student_hdn', hdn)


def pool_path():
    """The pool file the runtime reads."""
    return os.path.join(os.environ['EDGE_DATA'], 'train', 'pool.jsonl')


def threads():
    """Torch threads for training (EJ_THREADS, default 2)."""
    return int(os.environ.get('EJ_THREADS', '2'))


def _same(a, b):
    from ..integrity import file_sha256
    return os.path.getsize(a) == os.path.getsize(b) and file_sha256(a) == file_sha256(b)
