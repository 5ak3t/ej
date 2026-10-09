"""Adapting ej to one workflow: ``model.adapt(examples)`` returns an ``AdaptedModel``.

ej is trained to answer questions it has never seen, so it cannot know how often each answer occurs in your workflow. Adaptation
adds a per-(question id, option key) offset to the zero-shot log-probabilities (the OPTION TILT, ej.adapt_math), fitted on that
workflow's labelled records:

* ``examples``: records of the workflow with their correct answers, as ``'answers': {qid: option key}`` (or the benchmark format
  ``'gold': {qid: {'label': option index}}``). A few labelled records are enough.
* ``AdaptedModel.observe(batch)`` folds further labelled records in as they arrive and refits; observing in batches gives the
  same offsets as adapting on all records at once.

What the tilt learns is mostly the workflow's label prior: on unseen workflows an add-one count of the labelled answers reaches
almost the same accuracy (docs/adaptation.md). Offsets are matched by question id and option key, so use one AdaptedModel per
workflow. Without examples ``model.adapt()`` returns the model itself (identical predictions). Cost: one zero-shot pass over the
labelled records plus a fit of a few milliseconds; predicting with an adapted model costs the same as with the base model."""
import torch

from . import adapt_math as AM
from .records import RecordError, validate_records

DEFAULTS = {'sb': 0.5}
"""sb: prior scale of the option tilt (offsets ~ N(0, sb^2)); chosen on development data the model was trained on."""


def answers_of(record):
    """{qid: option index} of the answered questions of a record ('answers' by option key, or 'gold' by option index)."""
    qs, out = record['questions'], {}
    for qid, key in (record.get('answers') or {}).items():
        keys = [o['key'] for o in qs.get(qid, {}).get('options', [])]
        if key not in keys:
            raise RecordError(f'record {record.get("id")!r}: answer {key!r} of {qid!r} is not one of its option keys {keys}')
        out[qid] = keys.index(key)
    for qid, g in (record.get('gold') or {}).items():
        if qid not in out and isinstance(g, dict) and isinstance(g.get('label'), int) and qid in qs:
            if not 0 <= g['label'] < len(qs[qid]['options']):
                raise RecordError(f'record {record.get("id")!r}: gold label of {qid!r} is out of range')
            out[qid] = g['label']
    return out


def _rows(records, preds):
    """([(qid, option keys)], log-probabilities (n, K) float64 with -inf padding) in record / question order."""
    rows = [(qid, [o['key'] for o in q['options']]) for r in records for qid, q in r['questions'].items()]
    K = max((len(k) for _, k in rows), default=1)
    lp = torch.full((len(rows), K), -float('inf'), dtype=AM.DT)
    n = 0
    for r, d in zip(records, preds):
        for qid in r['questions']:
            p = torch.tensor(d[qid], dtype=AM.DT)
            lp[n, :len(p)] = p.clamp(min=1e-300).log()
            n += 1
    return rows, lp


def _blind(records):
    """The records without their answers (the base model never sees them)."""
    return [{k: v for k, v in r.items() if k not in ('answers', 'gold')} for r in records]


class AdaptedModel:
    """An ej model adapted to one workflow. Create it with ``Model.adapt``; ``predict`` as ``Model.predict``."""

    def __init__(self, base, examples=None, config=None):
        """Adapt `base` (an ej.Model, or any object with the same predict) with labelled `examples`."""
        self.base, self.config = base, {**DEFAULTS, **(config or {})}
        self.offsets, self._rows, self._y, self._lp = {}, [], [], None
        self.observe(examples or [])

    @property
    def n_labelled(self):
        """Number of labelled questions the offsets are fitted on."""
        return len(self._y)

    def observe(self, records):
        """Fold a batch of labelled records of the workflow into the adaptation and refit. Records without answers add
        nothing. Returns self."""
        records = validate_records(records)
        ans = [answers_of(r) for r in records]
        if not any(ans):
            return self
        rows, lp = _rows(records, self.base.predict(_blind(records)))
        n, keep = 0, []
        for r, a in zip(records, ans):
            for qid in r['questions']:
                if qid in a:
                    keep.append((n, a[qid]))
                n += 1
        ix = torch.tensor([i for i, _ in keep])
        ok = AM.usable(lp[ix])
        new_lp = lp[ix][ok]
        self._rows += [rows[i] for i, k in zip(ix.tolist(), ok.tolist()) if k]
        self._y += [y for (_, y), k in zip(keep, ok.tolist()) if k]
        self._lp = new_lp if self._lp is None else _cat(self._lp, new_lp)
        self.offsets = AM.fit_tilt(self._rows, self._y, self._lp, sb=self.config['sb']) if self._y else {}
        return self

    def predict(self, records):
        """[{qid: [p_1, ..., p_K]}] per record, adapted (one zero-shot pass, then the offsets)."""
        records = validate_records(records)
        if not records:
            return []
        preds = self.base.predict(records)
        return preds if not self.offsets else self._apply(records, preds)

    def _apply(self, records, preds):
        rows, lp = _rows(records, preds)
        p = torch.softmax(lp + AM.offsets(rows, self.offsets, lp.shape[1]).masked_fill(~torch.isfinite(lp), 0.0), -1)
        out, n = [], 0
        for r in records:
            d = {}
            for qid, q in r['questions'].items():
                row = p[n, :len(q['options'])].numpy().astype(float)
                d[qid] = [float(x) for x in row / row.sum()]
                n += 1
            out.append(d)
        return out

    def __repr__(self):
        return f'ej.AdaptedModel(base={self.base!r}, labelled={self.n_labelled}, offsets={len(self.offsets)})'


def _cat(a, b):
    """Stack two (n, K) log-probability blocks with -inf padding to the wider K."""
    K = max(a.shape[1], b.shape[1])
    pad = lambda t: torch.cat([t, torch.full((t.shape[0], K - t.shape[1]), -float('inf'), dtype=t.dtype)], 1)  # noqa: E731
    return torch.cat([pad(a), pad(b)], 0)
