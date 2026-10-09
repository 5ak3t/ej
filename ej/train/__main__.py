"""Command line of ej.train (``python -m ej.train --help``)."""
import argparse
import os
import sys


def _start(a):
    """Set up the work directory, then import torch / the runtime with the requested thread count."""
    from . import env
    work = a.work or (os.path.abspath(a.out).rstrip(os.sep) + '.work' if getattr(a, 'out', None) else None)
    if not work:
        sys.exit('ej.train: give --work DIR (fit defaults to OUT.work next to --out OUT)')
    env.setup(a.pool, work)
    from . import seed
    seed.claim(work, a.seed)
    seed.install(a.seed)  # before the runtime import; seed 0 patches nothing
    import torch
    import student  # noqa: F401  (runtime import: sets its own seed / threads; ours follow)
    torch.set_num_threads(env.threads())
    return work


def cmd_encoder(a):
    _start(a)
    from . import fit
    print(fit.encoder(a.variant))


def cmd_teachers(a):
    _start(a)
    from . import fit
    fit.teachers(a.parts)


def cmd_distil(a):
    _start(a)
    from . import distil
    distil.run()


def cmd_fit(a):
    if not a.out:
        sys.exit('ej.train fit: --out DIR is required (a new or empty directory for the weights)')
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f'ej.train fit: {out} is not empty; choose a new --out')
    _start(a)
    from . import fit
    fit.fit(out)
    print(out)


def cmd_export(a):
    from ..pack.write import from_weights_dir
    r = from_weights_dir(a.weights, a.out)
    print(f"{r['path']} ({r['bytes']:,} bytes)")


def parser():
    """The argument parser."""
    ap = argparse.ArgumentParser(prog='python -m ej.train', description='Train ej from a training pool (JSONL records with '
                                 'gold labels; format in docs/training.md). Environment: EJ_DEVICE=cuda|cpu (encoder '
                                 'distillation), EJ_THREADS (torch threads, default 2), EJ_CACHE (encoder cache).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    specs = (('encoder', cmd_encoder, 'distil the low-bit shared encoder (GPU recommended)'),
             ('teachers', cmd_teachers, 'train the decision-encoder teachers and the group-honest pair teachers (CPU)'),
             ('distil', cmd_distil, 'build the distilled NLI pair heads and the distilled decision encoder (CPU)'),
             ('fit', cmd_fit, 'fit the model (training any missing step first) and write a weights directory'))
    for name, fn, h in specs:
        p = sub.add_parser(name, help=h, description=h)
        p.add_argument('--pool', help='training pool JSONL (copied into the work directory; required on first use)')
        p.add_argument('--work', help='work directory for checkpoints and caches (default for fit: OUT.work)')
        p.add_argument('--seed', type=int, default=0, help='fit seed (0 = the released path; >= 1: a replicate, own --work)')
        if name == 'fit':
            p.add_argument('--out', help='new or empty directory that receives the weights (ej.load(OUT) reads it)')
        if name == 'encoder':
            p.add_argument('--variant', default='w23', choices=('w23', 'w2'), help='bit layout (the model reads w23)')
        if name == 'teachers':
            p.add_argument('--parts', nargs='*', help='only these parts (full, fold0.., p01..; default all)')
        p.set_defaults(fn=fn)
    p = sub.add_parser('export', help='write the packed model file (model.ejpack) of a weights directory',
                       description='Write the packed model file (ejpack v1: one file, pickle-free, no base-model download '
                       'at load) of a weights directory written by fit. Reads the base tokenizer and config once (no weights).')
    p.add_argument('--weights', required=True, help='weights directory (fit --out DIR)')
    p.add_argument('--out', required=True, help='pack file to write (e.g. model.ejpack), or a directory for model.ejpack')
    p.set_defaults(fn=cmd_export)
    return ap


def main(argv=None):
    """Entry point."""
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
