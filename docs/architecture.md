# Architecture

How a prediction is computed, module by module. Everything named `student_*` lives in `ej/_runtime/` (sha256-pinned); the
flags are module constants at the top of `ej/_runtime/student.py` and are those of the released ej 0.0.1 model. Changing a
flag is a model change (new runtime manifest, new weights).

## Record format

`{'id', 'state': str, 'questions': {qid: {'type': 'choice' | 'noul' | 'score', 'instructions': str, 'options': [{'key',
'text'}, ...]}}}`. Training and evaluation records add `'source'` (and optionally `'workflow'`) and `'gold': {qid:
{'label': int, 'probs': list | None}}`. A `noul` question has exactly the two options false / true; `score` options are
ordered levels.

## Predict path (one record batch)

1. **Shared encoder** (`student_lb`, `student_lbq`, `student_fast`; `USE_LB = 'w23'`): a trimmed-vocabulary low-bit
   `intfloat/e5-small-v2` (2-bit embeddings and feed-forward matrices, 3-bit attention, groups of 128 columns with fp16 step
   and offset; quantisation-aware distillation by `ej.train.lowbit`), used in fit and predict; the heads are stored int8.
   ej 0.0.1 uses `lowbit-b3b010513f948ceb`. `student_enc` is the k-bit e5 wrapper and embedding cache; `student_cold`
   implements cold mode (`EJ_COLD=1`: no caches).
2. **State-once read** (`student_read`): the state is encoded once per record into `[e5(query: state); question-attended
   field units; e5(query: instructions)]`; option texts are encoded as `passage:` vectors. JSON states are read field by
   field (key-aware attention). Lexical features and the TF-IDF space come from `student_feat`.
3. **Experts** (one logit vector over the options each):
   - deep + wide (`student_deep`, `student_wide`): a convex conditional logit over state-option interactions (zero-shot
     capable) plus sparse state-token x option-slot crosses (active only when all option slots were seen in training);
   - rich (`student_rich`): a bilinear / ordinal / MLP scorer that memorises training label spaces;
   - distilled decision encoder (`student_dd`; `USE_DD = True`, `USE_DEC = False`): a rich scorer distilled from the
     out-of-fold distributions of a fine-tuned e5 top stack (`student_dec`, `student_x`, `student_ft`; teachers trained by
     `train_merge` and `ej.train.hcf`), so there is no second encoder pass at predict time;
   - centred deep logits (prior-free; the option-text label marginal removed);
   - field attention (`student_attn`, seen tasks only);
   - relational tokens of JSON states (`student_rel`, seen slots) and the slot-free relational reader (`student_rr`, fitted
     by `student_rrf`);
   - debiased main expert + a bias-only expert and its negation (`student_poe`, `USE_POE = 'poe+neg'`): the bias expert
     reads the question and options but not the state;
   - NLI expert (`USE_NLI`, `USE_DN`): a pair head with token late interaction (`student_dn`, `student_dnfit`, `student_tok`)
     distilled from the cross-encoder `cross-encoder/nli-deberta-v3-xsmall` (`student_nli`), which runs at training time
     only.
4. **Pool** (`student_pool`, `student_hbs`, `student_hbs2`): per calibration key (question type, all slots seen, JSON state)
   a log-linear pool of the experts with a lapse to uniform, `p = (1 - eps) softmax(sum_e a_e z_e) + eps / K`. Unseen-slot
   keys use Bayesian hierarchical stacking over group-honest leave-group-out rows (`student_hcf`, `ej.train.hdn`, pair
   teachers from `ej.train.hcf`); with more than 12 groups the held-out unit is a block of groups (`student_kf`).
5. **Selective head** (`student_sel`, `USE_SEL = True`): a cross-fitted correctness model on the pool's top label, adopted
   in fit only where its own cross-fitted NLL beats the pool.
6. Each question's row is renormalised to sum to 1.

## Fit path

`student.fit(train)` builds the TF-IDF space and tensors, fits the upstream experts (memoised by `student_up`), reads the
training checkpoints (decision-encoder teachers, distilled heads, pair teachers, low-bit encoder) from the work directory,
fits the pool and the selective head, compacts the heads to int8 and returns the state. `python -m ej.train` prepares
those checkpoints and writes the state as a pickle-free weights directory; `python -m ej.train export` packs it into
`model.ejpack` (`docs/training.md`).

## Group honesty

Every row used to fit the unseen-slot pool for a held-out group comes only from models and teachers that never saw that
group: leave-one-group-out experts, decision-encoder teachers trained without the group's fold, pair teachers trained
without two folds (double cross-fitting), and NLI pair heads trained without the group pair (`ej.train.hdn`). Fit prints an
audit of the fold map and refuses to continue on any violation.

## Imported but off under the released flags

`student_gli` (`USE_GLI`), `student_gcv` (`USE_GCV`), `student_selg` (`SEL_GROUP`), `student_shr` (`USE_SHR`),
`student_shallow` (`USE_SHALLOW`; its helpers are still used by `student_lb`), `student_ord` (`USE_ORD`), the decision-encoder
second pass (`USE_DEC`) and the zero-shot verifier (`USE_VER`). They stay because `student.py` imports them at module
level, which the runtime manifest pins.

## Weights format: ejpack v1

The model ships as one file, `model.ejpack` (`ej.pack`). Layout: an 8-byte magic, the header length, the header's sha256,
a JSON header (section table with offset, length and sha256 per section, plus descriptors), then 64-byte aligned sections:

| sections | contents | 0.0.1 bytes |
|---|---|---|
| `enc.q2`, `enc.q3`, `enc.s`, `enc.z` | the encoder's codes, bit-packed at 2 and 3 bits, and their fp16 steps and offsets per group of 128 | 8,173,440 |
| `heads.*`, `cross.*` | int8 head matrices (fp16 row scales) and int8 cross weights (fp16 scale per block of 64), with a 1-bit plane marking stored -0.0 values | 1,618,422 |
| `voc.*`, `rvoc.*` | the cross vocabularies as 5-byte sha256 key hashes in column order plus a slot table (token strings are not stored) | 1,269,302 |
| `rest.safetensors`, `rest.json.gz` | every other tensor (fp16 where that is exact, else fp32) and the JSON skeleton of the state | 264,005 |
| `tok.vocab.txt.gz`, `tok.config.json` | the trimmed tokenizer (11,680 pieces) and the BERT config | 40,136 |

Everything is stored exactly: an int8 or fp16 form is used only when it reproduces every stored float bit for bit, so the
pack decodes to the same fitted state as the weights directory, and predictions from both are identical (max |Δp| = 0.0 on
20 development records, `tests/test_pack_weights.py`). Hashed keys can mistake an unseen key for a known one with
probability at most about 7e-9 per lookup. Loading verifies the header digest and every section's sha256 before decoding,
never unpickles (classes from `ej.safe.ALLOWED_CLASSES` only) and downloads nothing: the encoder is built from the pack's
config on a meta-device shell, its weights dequantised once (default) or, with `low_memory=True`, kept as codes and
dequantised inside each forward call (same arithmetic, so the same values). `ej.pack.runtime` installs the result where
`ej/_runtime` looks for its encoder and checkpoint, so the runtime itself is unchanged.

The weights directory (`ej.safe`: every tensor in `.safetensors`, everything else in a JSON skeleton) is the format
`python -m ej.train fit` writes and remains loadable; it builds the encoder from the base model. `ej.scope` keeps the process
clean in both cases: thread count and RNG restored after the runtime import, threads scoped to each predict call, an
LRU-bounded encoder memo, and `torch.load` / `pickle.load` refusing inside every runtime module after `ej.load`.

## What we tried

- **A small joint-reading cross-encoder next to ej.** `cross-encoder/nli-deberta-v3-xsmall`, fine-tuned on the training pool
  and combined with ej's distribution, improved unseen-workflow accuracy (zs_wide development macro_real) at full precision,
  but not within the size budget: with int8 weights and a vocabulary trimmed to fit the budget it re-segmented most
  unseen-workflow texts and kept less than half of the gain, below the improvement set in advance as the bar for shipping;
  the untrimmed model is several times ej's size. It is not included in 0.0.1.
