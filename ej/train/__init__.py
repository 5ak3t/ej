"""ej.train: train your own ej from a training pool (``python -m ej.train --help``; ``pip install 'ej[train]'``).

    python -m ej.train fit --pool pool.jsonl --out my-weights        # everything; resumes from my-weights.work
    python -m ej.train encoder --pool pool.jsonl --work W           # steps one by one (encoder on a GPU, the rest on CPU)
    python -m ej.train teachers --pool pool.jsonl --work W
    python -m ej.train distil --pool pool.jsonl --work W
    python -m ej.train fit --pool pool.jsonl --work W --out my-weights
    python -m ej.train export --weights my-weights --out model.ejpack   # the packed model file (ej.pack)

Modules: env (work directory and device), seed (fit-seed replicates), lowbit (low-bit encoder distillation), hcf / hdn (group-honest teachers, imported by
the runtime under the bare names train_hcf / student_hdn), distil (distilled heads), fit (the pipeline), export (pickle-free
weights directory). The pool format is described in docs/training.md. Importing this package imports nothing heavy; the
runtime is imported only after env.setup()."""
