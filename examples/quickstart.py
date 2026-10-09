"""ej quickstart: load a model and predict one record (or a JSONL file of records).

  python examples/quickstart.py --weights ./model.ejpack        # the packed model file (or a directory holding it)
  python examples/quickstart.py --weights ./model.ejpack --records my_records.jsonl
  python examples/quickstart.py --weights ./model.ejpack --low-memory
A model.ejpack comes from `python -m ej.train fit` + `python -m ej.train export`, or from the 0.0.1 release once it is
published (README: release status). A weights directory (fit output) also loads."""
import argparse
import json

import ej


def main():
    """Load, predict ej.EXAMPLE_RECORD, print each question's distribution."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', required=True, help='model.ejpack, a weights directory, or a Hugging Face repo id you can access')
    ap.add_argument('--revision', help='Hugging Face revision (tag or commit) when --weights is a repo id')
    ap.add_argument('--records', help='optional JSONL file of records; prints one JSON line per record')
    ap.add_argument('--low-memory', action='store_true', help='stream the 2/3-bit encoder weights (packs only; slower)')
    a = ap.parse_args()
    model = ej.load(a.weights, revision=a.revision, low_memory=a.low_memory)
    if a.records:
        with open(a.records) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for r, p in zip(records, model.predict(records)):
            print(json.dumps({'id': r.get('id'), 'probs': p}))
        return
    (pred,) = model.predict([ej.EXAMPLE_RECORD])
    for qid, q in ej.EXAMPLE_RECORD['questions'].items():
        top = max(range(len(pred[qid])), key=pred[qid].__getitem__)
        print(f'{qid} ({q["type"]})')
        for k, (o, p) in enumerate(zip(q['options'], pred[qid])):
            print(f'  {p:.3f}  {o["key"]}{"  <- argmax" if k == top else ""}')


if __name__ == '__main__':
    main()
