"""OpenThai-SystemOne (iapp/OpenThai-SystemOne v0.3, 0.75B Qwen3.5-0.8B text tower + 256-slot head), in process, float32.

The repo ships its model as transformers remote code (trust_remote_code); we do NOT execute downloaded code. Instead this
module rebuilds the documented inference with stock transformers (Qwen3_5TextModel) and the checkpoint's own tensors:
  layout (formatting.py @ dc67b02 and the openthai_systemone client, Apache-2.0, read as data):
    <|ts_state|> {state}\\n  then per question  <|ts_q|><|ts_choice|ts_score|ts_noul|> {instructions}\\n
    <|ts_opt_i|> {name}[: {description}]\\n ... <|ts_answer|>\\n      (choice: name = option key; score: name = level
    index "0".."k-1", description = level text; noul: names "no", "yes", descriptions = criteria false / true)
  head: hidden state at each <|ts_answer|> -> slot_head (1024 -> 256) / exp(log_temperature[type]) (choice 0, score 1,
    noul 2) -> mask slots >= k except 255 (abstain) -> softmax -> p[:k] renormalised (abstain mass dropped, as the
    client does); noul = p(yes) / (p(no) + p(yes)).
  order-invariant auto mode (client default): if a choice question has >= 11 options, every choice question is averaged
    over min(8, k) evenly spread cyclic option orders (one forward each); never triggered by <= 10-option suites.
  state budget: 32768 tokens (head + tail kept), as the client.
float32 compute; the 248k x 1024 embedding table is kept in its stored bf16 and upcast after lookup (bit-identical to an
fp32 table, saves 0.5 GB RSS). Same EPS smoothing / option-order mapping as the other adapters (common.finalize)."""
import functools
import json

from . import common

REPO, REV = 'iapp/OpenThai-SystemOne', 'dc67b0295ab7ac94b05d27187150d9a4b5784554'
N_SLOTS, ABSTAIN = 256, 255
TYPE_TOK = {'choice': '<|ts_choice|>', 'score': '<|ts_score|>', 'noul': '<|ts_noul|>'}
TEMP_INDEX = {'choice': 0, 'score': 1, 'noul': 2}
MAX_STATE, AUTO_MIN_OPTIONS, AUTO_PERMS = 32768, 11, 8


def sanitize(text):
    """Escape the model's control-token prefix inside user text."""
    return text.replace('<|ts_', '<​|ts_') if '<|ts_' in text else text


@functools.lru_cache(maxsize=1)
def load():
    """Load the model once per process (cached)."""
    torch = common.setup_torch()
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen2Tokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    class Bf16Embedding(torch.nn.Module):
        """Lookup in the stored bf16 table, return float32 (exactly the fp32 table's values)."""

        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, ids):
            return torch.nn.functional.embedding(ids, self.weight).float()

    tok = Qwen2Tokenizer.from_pretrained(REPO, revision=REV)  # tokenizer_config's class; config.json auto_map unused
    special = ['<|ts_state|>', '<|ts_q|>', *TYPE_TOK.values(), '<|ts_answer|>'] + [f'<|ts_opt_{i}|>' for i in range(N_SLOTS)]
    missing = [t for t in special if t not in tok.get_vocab()]
    assert not missing, f'tokenizer lacks control tokens {missing[:3]}'
    cfg = json.load(open(hf_hub_download(REPO, 'config.json', revision=REV)))
    tc = dict(cfg['text_config'])
    vocab = tc['vocab_size']
    tc['vocab_size'], tc['eos_token_id'] = 1, None  # the real table is attached below in bf16
    text_cfg = AutoConfig.for_model(tc.pop('model_type'), **tc)
    model = Qwen3_5TextModel(text_cfg).float().eval()
    head = torch.nn.Linear(text_cfg.hidden_size, N_SLOTS, bias=True)
    with safe_open(hf_hub_download(REPO, 'model.safetensors', revision=REV), framework='pt') as f:
        keys = set(f.keys())
        emb = f.get_tensor('model.embed_tokens.weight')
        assert emb.shape[0] == vocab == len(tok), (emb.shape, vocab, len(tok))
        model.embed_tokens = Bf16Embedding(emb)
        params = dict(model.named_parameters())
        for name, p in params.items():
            if name != 'embed_tokens.weight':
                p.data.copy_(f.get_tensor('model.' + name).float())
        head.weight.data.copy_(f.get_tensor('slot_head.weight').float())
        head.bias.data.copy_(f.get_tensor('slot_head.bias').float())
        temps = f.get_tensor('log_temperature').float().exp()
    loaded = {'model.' + n for n in params} | {'slot_head.weight', 'slot_head.bias', 'log_temperature'}
    assert keys == loaded, f'unmatched tensors: {sorted(keys ^ loaded)[:5]}'
    ids = {t: tok.convert_tokens_to_ids(t) for t in special}
    assert ids['<|ts_answer|>'] == cfg['answer_token_id'], ids['<|ts_answer|>']
    return torch, tok, model, head.eval(), temps, ids


def spec(q, order=None):
    """Evaluator question -> (type, [(name, desc)] in slot order, option keys in slot order)."""
    w = common.to_wire(q)
    if q['type'] == 'choice':
        names = list(w['criteria'])
        idx = order if order is not None else range(len(names))
        return 'choice', [(names[i], w['criteria'][names[i]]) for i in idx], [names[i] for i in idx]
    if q['type'] == 'score':
        return 'score', [(str(i), d) for i, d in enumerate(w['criteria'])], [str(i) for i in range(len(w['criteria']))]
    c = w.get('criteria') or {}
    return 'noul', [('no', c.get('false')), ('yes', c.get('true'))], ['false', 'true']


def question_text(t, instructions, opts):
    """One question block of the prompt (type token, instructions, numbered options, answer token)."""
    lines = [f"<|ts_q|>{TYPE_TOK[t]} {sanitize(instructions).strip()}"]
    for i, (name, desc) in enumerate(opts):
        name = sanitize(str(name)).strip()
        lines.append(f"<|ts_opt_{i}|> {name}: {sanitize(str(desc)).strip()}" if desc else f"<|ts_opt_{i}|> {name}")
    lines.append('<|ts_answer|>')
    return '\n'.join(lines) + '\n'


def encode(tok, ids, state, specs):
    """-> (token ids, answer positions) for one prompt (state + all questions), with the client's state budget."""
    enc = lambda s: tok(s, add_special_tokens=False)['input_ids']  # noqa: E731
    q_ids = [enc(question_text(t, ins, opts)) for t, ins, opts in specs]
    s_ids = enc('<|ts_state|> ' + sanitize(state).strip() + '\n')
    budget = min(MAX_STATE, 65536 - sum(map(len, q_ids)))
    if len(s_ids) > budget:
        s_ids = s_ids[:1] + s_ids[len(s_ids) - max(budget - 1, 0):]
    out, pos = list(s_ids), []
    for qi in q_ids:
        out += qi
        p = len(out) - 1
        while out[p] != ids['<|ts_answer|>']:
            p -= 1
        pos.append(p)
    return out, pos


def cyclic_orders(k, n):
    """n evenly spread cyclic orders of k options."""
    n = max(1, min(n, k))
    offsets = sorted({round(j * k / n) % k for j in range(n)})
    return [[(i + off) % k for i in range(k)] for off in offsets]


def forward(state, questions, plan):
    """One prompt per option-order plan; -> {qid: {key: prob}} averaged over the plan (client semantics)."""
    torch, tok, model, head, temps, ids = load()
    acc = {}
    for orders in plan:
        sp = {qid: spec(q, orders.get(qid)) for qid, q in questions.items()}
        seq, pos = encode(tok, ids, state, [(t, questions[qid]['instructions'], o) for qid, (t, o, _) in sp.items()])
        with torch.inference_mode():
            h = model(input_ids=torch.tensor([seq]), use_cache=False).last_hidden_state[0, pos]
            logits = head(h)
        for j, (qid, (t, opts, keys)) in enumerate(sp.items()):
            z = logits[j] / temps[TEMP_INDEX[t]]
            valid = torch.arange(N_SLOTS) < len(opts)
            valid[ABSTAIN] = True
            p = torch.softmax(z.masked_fill(~valid, float('-inf')), -1)[:len(opts)].double()
            p = p / p.sum().clamp(min=1e-12)
            d = acc.setdefault(qid, {})
            for k, v in zip(keys, p.tolist()):
                d[k] = d.get(k, 0.0) + v / len(plan)
    return acc


def make_plan(questions):
    """Option-order plan: one order, or 8 cyclic orders when a choice has >= 11 options."""
    ks = {qid: len(q['options']) for qid, q in questions.items() if q['type'] == 'choice'}
    n = AUTO_PERMS if any(k >= AUTO_MIN_OPTIONS for k in ks.values()) else 1
    n = max(1, min(n, max(ks.values(), default=1)))
    orders = {qid: cyclic_orders(k, n) for qid, k in ks.items()}
    return [{qid: o[j % len(o)] for qid, o in orders.items()} for j in range(n)]


def predict(records):
    """[{qid: probs}] per record, in each question's option order."""
    out = []
    for r in records:
        probs = forward(common.state_text(r), r['questions'], make_plan(r['questions']))
        out.append(common.from_response(r, {qid: {'probabilities': p} for qid, p in probs.items()}))
    return out
