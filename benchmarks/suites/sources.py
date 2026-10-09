"""Converters from public / in-house datasets to the ej typed-decision record format.

Record: {'id', 'source', 'state': text, 'questions': {qid: {'type': 'choice'|'noul'|'score', 'instructions': text,
         'options': [{'key': k, 'text': description}]}}, 'gold': {qid: {'label': index, 'probs': [..] or None}}}
Every converter is deterministic (seeded by record id). Licences are recorded in LICENCES (data card)."""
import hashlib
import json
import random

LICENCES = {
    'typed-decisions': 'Apache-2.0 (LocalLLaMA/typed-decisions)',
    'tickets': 'in-house corpus written with Claude Haiku for this project (not released)',
    'banking77': 'CC-BY-4.0 (PolyAI/banking77; parquet mirror mteb/banking77)',
    'clinc150': 'CC-BY-3.0 (clinc/clinc_oos, config plus)',
    'massive': 'CC-BY-4.0 (AmazonScience/massive; parquet mirror mteb/amazon_massive_intent, en)',
    'goemotions': 'Apache-2.0 (google-research-datasets/go_emotions, simplified)',
    'counterfactual': 'CC BY-NC 4.0 upstream (amazon-research; the mteb mirror says CC-BY-4.0); not used by ej',
}
NOUL_TEXT = {'false': 'No, the statement does not hold.', 'true': 'Yes, the statement holds.'}


def rng_for(rid):
    return random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:12], 16))


def human(name):
    return name.replace('_', ' ').replace('-', ' ').strip()


def typed_decisions(row):
    """One Typed Decisions case (gold = teacher distributions, label = gold label)."""
    qs, gold = json.loads(row['questions']), json.loads(row['gold'])
    out_q, out_g = {}, {}
    for qid, q in qs.items():
        crit = q.get('criteria') or dict(NOUL_TEXT)  # noul without criteria: generic yes/no texts
        if q['type'] == 'score':
            opts = [{'key': str(i), 'text': t} for i, t in enumerate(crit)]
            keys = [o['key'] for o in opts]
        else:
            opts = [{'key': k, 'text': v} for k, v in crit.items()]
            keys = list(crit)
        g = gold[qid]
        probs = g.get('probabilities')
        if isinstance(probs, dict):
            probs = [float(probs.get(k, 0.0)) for k in keys]
        elif isinstance(probs, list):
            probs = [float(p) for p in probs]
        lab = str(g['label'])
        if q['type'] == 'score' and lab not in keys:  # score gold may be the level text or index
            lab = str(crit.index(g['label'])) if g['label'] in crit else str(int(float(g['label'])))
        out_q[qid] = {'type': q['type'], 'instructions': q['instructions'], 'options': opts}
        out_g[qid] = {'label': keys.index(lab), 'probs': probs}
    return {'id': row['id'], 'source': 'typed-decisions', 'workflow': row['workflow'], 'state': row['state'],
            'questions': out_q, 'gold': out_g}


def ticket(r):
    """One rev haiku ticket (dept choice, urgent noul, mood score)."""
    q = r['request']['questions']
    dept = q['dept']
    out_q = {
        'dept': {'type': 'choice', 'instructions': dept['instructions'],
                 'options': [{'key': k, 'text': f'{k}: {v}'} for k, v in dept['criteria'].items()]},
        'urgent': {'type': 'noul', 'instructions': q['urgent']['instructions'],
                   'options': [{'key': k, 'text': NOUL_TEXT[k]} for k in ('false', 'true')]},
        'mood': {'type': 'score', 'instructions': q['mood']['instructions'],
                 'options': [{'key': str(i), 'text': t} for i, t in enumerate(q['mood']['criteria'])]},
    }
    lab = r['labels']
    return {'id': r['id'], 'source': 'tickets', 'state': r['request']['state'], 'questions': out_q,
            'gold': {k: {'label': int(lab[k]), 'probs': None} for k in ('dept', 'urgent', 'mood')}}


def sampled_choice(rid, source, text, label, names, instructions, kmin=2, kmax=8, always=()):
    """A choice question over the true label plus sampled distractors (kev-style option augmentation)."""
    rng = rng_for(rid)
    k = rng.randint(kmin, kmax)
    pool = [n for n in names if n != label and n not in always]
    opts = [label] + rng.sample(pool, min(k - 1 - len(always), len(pool))) + [a for a in always if a != label]
    rng.shuffle(opts)
    q = {'type': 'choice', 'instructions': instructions, 'options': [{'key': o, 'text': human(o)} for o in opts]}
    return {'id': rid, 'source': source, 'state': text, 'questions': {'intent': q},
            'gold': {'intent': {'label': opts.index(label), 'probs': None}}}


def noul(rid, source, text, label, instructions):
    q = {'type': 'noul', 'instructions': instructions,
         'options': [{'key': k, 'text': NOUL_TEXT[k]} for k in ('false', 'true')]}
    return {'id': rid, 'source': source, 'state': text, 'questions': {'check': q},
            'gold': {'check': {'label': int(label), 'probs': None}}}
