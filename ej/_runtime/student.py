"""Prediction entry point of the ej model. fit(train) builds the fitted state (shared low-bit encoder features, lexical
features, the expert heads and the calibrated pool); predict(state, records) returns one probability distribution per
question, in option order; size_mb(state) gives the counted model size. The flags below select which experts are active; ej
calls only predict."""
import sys

import numpy as np
import torch

import student_attn as A
import student_cold as COLD
import student_dec as DEC
import student_deep as D
import student_enc as E
import student_feat as F
import student_gcv as GCV
import student_gli as GLI
import student_pool as P
import student_read as RD
import student_rel as REL
import student_rich as R
import student_nli as NLI
import student_dd as DD, student_hcf as HCF, student_hbs as HBS, student_rrf as RRF, student_poe as PO
import student_fast as FA, student_shallow as SH, student_lb as LB, student_kf as KF, student_ord as OR
import student_dn as DN
import student_dnfit as DNF
import student_rr as RR
import student_up as UP
import student_sel as SEL
import student_selg as SELG
import student_shr as SHR
import student_state as ST  # whole-state fit checkpoint (state-<code+pool key>.pt; EDGE_STATE=0 disables)
import student_wide as W
import student_x as XS
import train_merge as TM

MODEL, BITS = 'intfloat/e5-small-v2', 4
TYPES = ('choice', 'noul', 'score')
USE_GLI = False  # single-pass multi-option reader (student_gli) as an extra expert; off
USE_POE = 'poe+neg'  # '' off; 'poe' / 'conf' = bias-only expert (slot 10) + debiased main deep expert (slot 7; student_poe); '+neg' adds -bias (slot 11)
USE_ORD = False  # ordinal-aware score expert (student_ord, last slot): unimodal level distribution located by level-text matching
N_EXP = (10 if USE_GLI else 9) + PO.n_extra(USE_POE) + int(USE_ORD)
USE_NLI = True  # NLI reader (student_nli) over fact premises + state text, expert 9
USE_VER = False  # zero-shot verifier off -- its sign flips between questions (LOGO ~ uniform)
USE_SEL = True  # selective correctness head (student_sel); adopted inside fit only if its cross-fitted NLL beats the pool
USE_GCV = False  # unseen-slot pool by a second-level cross-fit over groups (student_gcv); off
USE_DN = True  # the NLI expert reads DISTILLED logits (student_dn pair head on the shared e5 pass); the cross-encoder is a training-time teacher only
USE_TEACHER_L = False  # large-teacher distillation path (student_tl / dn2 / dnfit2 / hdn2); off
if USE_TEACHER_L:
    import student_tl as TL; DN, DNF = TL.install()  # noqa: E702
USE_DEC = False  # False = no decision-encoder second pass (fine-tuned top stack) on the device; expert slot 3 silent
USE_DD = True  # with USE_DEC False: slot 3 = the decision encoder DISTILLED into a rich scorer (student_dd)
USE_HCF = True  # GROUP-HONEST nested cross-fitting of unseen-key LOGO rows (student_hcf/hdn, ckpt train_hcf.py)
USE_HBS = True  # Bayesian hierarchical stacking of the unseen-slot pool + new-group predictive (student_hbs)
USE_SHR = False  # head correction x hierarchical group-validated transfer slope per (seen, type, json) cell; off
USE_SHALLOW = False  # 8-layer distilled encoder, heads re-fit on it; off
USE_LB = 'w23'  # '' = 4-bit e5; otherwise the trimmed-vocabulary low-bit QAT e5 variant used as the shared encoder in fit AND predict, heads int8 (student_lb)
SEL_GROUP = False  # head per (seen, json) regime, unseen heads by leave-one-group-out (student_selg); off
torch.manual_seed(0)
torch.set_num_threads(2)
FA.install()  # EDGE_COLD=1 at predict time bypasses the embedding and token-state caches


def questions(recs):
    """Flatten to per-question items (group = source/workflow, used only in fit for leave-group-out calibration)."""
    out = []
    for r in recs:
        for qid, q in r['questions'].items():
            out.append({'state': r['state'], 'instr': q['instructions'], 'opts': [o['text'] for o in q['options']],
                        'type': TYPES.index(q['type']), 'json': F.parse_json(r['state']) is not None,
                        'group': f"{r.get('source')}/{r.get('workflow')}", 'y': r.get('gold', {}).get(qid, {}).get('label')})
    return out


def _emb(texts):
    keys = sorted(set(texts)); pos = {t: k for k, t in enumerate(keys)}
    return torch.tensor(E.embed(MODEL, BITS, keys)), pos


def tensors(items, tf):
    Ef = RD.fields(items)
    U, Hs = RD.read(items, Ef)  # state encoded once per record; each question reads it (no per-question state pass)
    Vu, pv = _emb([f'passage: {o}' for i in items for o in i['opts']])
    Hu, ph = _emb([f"passage: {i['instr']} {o}" for i in items for o in i['opts']])
    K = max(len(i['opts']) for i in items)
    iv, ih = np.full((len(items), K), -1), np.full((len(items), K), -1)
    for n, i in enumerate(items):
        iv[n, :len(i['opts'])] = [pv[f'passage: {o}'] for o in i['opts']]
        ih[n, :len(i['opts'])] = [ph[f"passage: {i['instr']} {o}"] for o in i['opts']]
    M = torch.tensor(iv >= 0)
    V, Ho = Vu[np.maximum(iv, 0)] * M[..., None], Hu[np.maximum(ih, 0)] * M[..., None]
    T = torch.tensor([i['type'] for i in items])
    u1 = torch.nn.functional.normalize(RD.fold(U, V.shape[-1]), dim=-1)
    S, ordp = D.scalars(u1, V, Hs, Ho, M, F.lexical(tf, items), RD.late(items, Ef, V, Ho), T)
    return (U, V, M, S, ordp, T), Ef, Ho


def sub(data, sel):
    return tuple(t[sel] for t in data)


def cal_key(item, seen):
    return (item['type'], bool(seen), item['json'])



def _pad(z, K):
    out = torch.full((z.shape[0], K), -1e9)
    out[:, :z.shape[1]] = z.float()
    return out


def _dec_model(ck):
    """Rebuild the fine-tuned top stack + head of the decision encoder from the checkpoint (fp16 -> fp32)."""
    _, _, top = DEC.FT.split(MODEL, BITS, DEC.K_TOP)
    top.load_state_dict({k: v.float() for k, v in ck['top'].items()}); top.eval()
    h = {k: v.float() for k, v in ck['head'].items()}
    head = XS.Head(h['w'], h['b']); head.load_state_dict(h)
    return top, head


@ST.checkpointed
@LB.compacted(USE_LB)
@FA.device_fit(lambda train: SH.fit(questions(train)) if USE_SHALLOW else LB.ck(USE_LB), lambda c: LB.encoder(c, SH))
def fit(train):
    items, real, unit = KF.relabel(questions(train))  # > 12 groups -> held-out unit = block of groups (grouped K-fold)
    hold = DEC.hold_mask([i['state'] for i in items])
    texts = sorted({i['state'] for i in items} | {i['instr'] for i in items} | {o for i in items for o in i['opts']})
    tf = F.Tfidf(texts)
    data, Ef, Ho = tensors(items, tf); y = torch.tensor([i['y'] for i in items]); K = data[2].shape[1]
    fb = A.bank(Ef, items)
    ck = TM.load(train) if USE_DEC or USE_DD else None  # decision encoder: full model + held-out logits + group-fold out-of-fold logits
    z_dec_oof = torch.full((len(items), K), -1e9) if ck else torch.zeros(len(items), K).masked_fill(~data[2], -1e9)
    z_dec_ho = _pad(ck['full']['z'], K) if USE_DEC else z_dec_oof[torch.tensor(hold)].masked_fill(~data[2][torch.tensor(hold)], 0)
    for k in range(len(DEC.FOLDS) if ck else 0):
        z_dec_oof[ck[f'fold{k}']['idx']] = _pad(ck[f'fold{k}']['z'], K)
    tr, ho = torch.tensor(~hold), torch.tensor(hold)
    gid = np.array([i['group'] for i in items])
    dd = DD.fit(data, y, z_dec_oof, hold, list(gid)) if USE_DD and not USE_DEC else None  # teacher OOF -> student
    if dd:
        z_dec_ho = dd['ho']
        for g in sorted(set(gid)):
            z_dec_oof[torch.tensor(gid == g)] = dd['logo'][g]
    up = UP.upstream(data, fb, y, hold, list(gid))  # round-2 experts, held-out + LOGO logits, refits (memoised checkpoint)
    zd_tr, zd_ho, zc_ho, zr_ho, za_ho = up['zd_tr'], up['zd_ho'], up['zc_ho'], up['zr_ho'], up['za_ho']
    ep, ep_a = up['ep'], up['ep_a']
    voc = W.vocab(items)
    it_tr, it_ho = [i for i, h in zip(items, hold) if not h], [i for i, h in zip(items, hold) if h]
    bag_tr, bag_ho = W.bags(voc, it_tr, K), W.bags(voc, it_ho, K)
    M_tr, M_ho = data[2][tr], data[2][ho]
    best = None
    for l2 in W.L2_GRID:
        w = W.fit(len(voc), bag_tr, zd_tr, M_tr, y[tr], l2)
        zw = W.logits(w, bag_ho, zd_ho.shape)
        nll = torch.nn.functional.cross_entropy((zd_ho + zw).masked_fill(~M_ho, -1e9), y[ho]).item()
        if best is None or nll < best[0]:
            best = (nll, l2, zw)
    _, l2, zw_ho = best
    seen_ho = (bag_ho[3] | ~M_ho).all(1)
    rvoc = REL.vocab(items, W.slot)  # relational expert (student_rel): binary slot x [field, relational] tokens, own L2
    jtr, jho = [torch.tensor([i['json'] for i in it]) for it in (it_tr, it_ho)]  # structured states only (else silent)
    rb_tr = REL.bags(rvoc, [i for i in it_tr if i['json']], K, W.slot)
    rb_ho = REL.bags(rvoc, [i for i in it_ho if i['json']], K, W.slot)
    rbest, sj = None, seen_ho[jho]
    for l2r in REL.REL_L2:
        wr = W.fit(len(rvoc), rb_tr, torch.zeros(M_tr[jtr].shape), M_tr[jtr], y[tr][jtr], l2r)
        zx = W.logits(wr, rb_ho, M_ho[jho].shape).masked_fill(~M_ho[jho], -1e9)
        nll = torch.nn.functional.cross_entropy(zx[sj], y[ho][jho][sj]).item()
        if rbest is None or nll < rbest[0]:
            rbest = (nll, l2r, zx)
    print('rel', [round(rbest[0], 4), rbest[1], len(rvoc)], file=sys.stderr)
    _, l2r, zx_j = rbest
    zx_ho = torch.zeros(M_ho.shape); zx_ho[jho] = zx_j.masked_fill(~M_ho[jho], 0)
    rr = RRF.rr_fit(items, data, Ho, y, hold, gid)  # slot-free relational reader (full) + zero-shot verifier
    zf_ho, zv_ho = rr['zf_ho'], rr['zv_ho'] * USE_VER
    xn = None if USE_DN else NLI.features(items, rr['mu']) if USE_NLI else torch.zeros(len(items), K, NLI.NF)  # NLI reader (LOGO-L2)
    nl = DNF.fit_expert(items, data[2], y, hold, list(gid)) if USE_DN else NLI.fit_all(xn, data[2], NLI.rows(items), y, hold, list(gid))
    print('nli', nl['rep'], file=sys.stderr)
    po = PO.fit(data, Ho, y, hold, list(gid), USE_POE) if USE_POE else None  # bias-only + PoE main, LOGO rows
    zv_ho = po['ho'][0] if po else zv_ho
    od = OR.fit(data, y, hold, list(gid)) if USE_ORD else None  # ordinal score expert, held-out + LOGO rows
    folds = tuple({g for g in set(gid) if KF.fold_of(g) == k} for k in range(len(DEC.FOLDS)))  # true fold sets (blocks nest in folds)
    hcf = HCF.honest(items, data, y, hold, list(gid), ck, train, up, dd, nl, folds, KF.fold_of) if USE_HCF else None
    for g in sorted(set(gid)) if hcf else []:  # honest slot-3 LOGO logits (teachers never saw g)
        z_dec_oof[torch.tensor(gid == g)] = dd['logo'][g]
    gl = GLI.fit(train, items, y, hold, list(gid)) if USE_GLI else None  # one joint [questions; options; state] pass per record
    groups, real_ho = {}, [r for r, h in zip(real, hold) if h]  # pool rows carry the REAL group (HBS random-effect unit)
    for n, i in enumerate(it_ho):  # seen-slot path: experts + wide on the iid held-out slice
        if seen_ho[n]:
            Z = torch.stack([zd_ho[n] + zw_ho[n], zr_ho[n] + zw_ho[n], z_dec_ho[n], zc_ho[n], za_ho[n] + zw_ho[n],
                             zx_ho[n].masked_fill(~M_ho[n], 0), zf_ho[n], zv_ho[n], nl['ho'][n]] + ([gl['ho'][n]] if USE_GLI else [])
                            + (PO.extras(po['ho'][1][n], po['full']['neg']) if po else []) + ([od['ho'][n]] if od else []))
            groups.setdefault(cal_key(i, True), []).append((Z, i['y'], real_ho[n]))
    for g in sorted(set(gid)):  # unseen-slot path: experts trained without a whole source/workflow (zero-shot cross-fitting)
        out = torch.tensor(gid == g)
        zg, zgc, rg = up['logo'][g]
        zf = z_dec_oof[out]
        print('logo', g, [round(torch.nn.functional.cross_entropy(z, y[out]).item(), 4) for z in (zg, rg, zf, zgc)], file=sys.stderr)
        zv = po['logo'][g][0] if po else rr['logo'].get(g, torch.zeros_like(zg)) * USE_VER
        zb = po['logo'][g][1] if po else zg
        zo = od['logo'][g] if od else zg
        for n, a, b, c, d, v, e, gg, bb, oo in zip(np.flatnonzero(gid == g), zg, rg, zf, zgc, zv, nl['logo'][g], gl['logo'][g] if USE_GLI else zg, zb, zo):
            z0 = torch.zeros_like(a)  # field-attention, slot relational, full relational reader: silent; verifier, NLI: LOGO logits
            Z = torch.stack([a, b, c, d, z0, z0, z0, v, e] + ([gg] if USE_GLI else []) + (PO.extras(bb, po['full']['neg']) if po else []) + ([oo] if od else []))
            groups.setdefault(cal_key(items[n], False), []).append((Z, items[n]['y'], real[n]))
    pool, cfgs, gcv_rep = GCV.fit(groups) if USE_GCV or SEL_GROUP else (None, {}, {})  # group cross-fit (rows carry groups)
    pool, gcv_rep = (pool, gcv_rep) if USE_GCV else HBS.fit(groups, unit) if USE_HBS else (P.fit(groups), gcv_rep)
    print('pool', pool, '\ngcv', gcv_rep, file=sys.stderr)
    sel, sel_rep = HBS.sel_fit(groups, pool, N_EXP) if USE_HBS else SHR.fit(groups, N_EXP) if USE_SHR else SELG.fit(groups, lambda k: k, N_EXP, cfgs) if SEL_GROUP else SEL.fit(groups, lambda k: k, N_EXP)
    print('sel', sel_rep, sel['lam'] if USE_SHR else {k: w is not None for k, w in sel.items()}, USE_SEL, file=sys.stderr)
    sel = sel if USE_SEL else None
    th_all, net_all, na_all = up['deep'], up['rich'], up['attn']
    w_all = W.fit(len(voc), W.bags(voc, items), D.logits(th_all, data), data[2], y, l2).half().float()
    ja = torch.tensor([i['json'] for i in items])
    w_rel = W.fit(len(rvoc), REL.bags(rvoc, [i for i in items if i['json']], K, W.slot), torch.zeros(data[2][ja].shape), data[2][ja],
                  y[ja], l2r).half().float()
    top, head = _dec_model(ck['full']) if USE_DEC else (None, None)
    return {'tf': tf, 'deep': th_all, 'ubar': data[0].mean(0).half().float(), 'rich': net_all, 'attn': na_all, 'voc': voc, 'w': w_all,
            'pool': pool, 'sel': sel, 'rvoc': rvoc, 'rel': w_rel, 'dec': (top, head), 'rr': rr['full'], 'rv': rr['ver'], 'mu': rr['mu'], 'nli': nl['full'], 'dn': nl.get('net'), 'dd': dd['full'] if dd else None, 'gli': gl['full'] if USE_GLI else None, 'poe': po['full'] if po else None, 'ord': od['full'] if od else None,
            'info': {'hcf': hcf, 'nli': nl['rep'], 'gli': gl['rep'] if USE_GLI else None, 'gcv': gcv_rep, 'rr': rr['rep'], 'poe': po['rep'] if po else None, 'ord': od['rep'] if od else None, 'sel': sel_rep, 'l2_rel': l2r, 'l2': l2, 'epochs': ep, 'epochs_attn': ep_a, 'pool': pool}}


@FA.device(lambda st: LB.predict_encoder(st, SH))
def predict(state, recs):
    items = questions(recs)
    data, Ef, Ho = tensors(items, state['tf'])
    M = data[2]; K = M.shape[1]
    zd, zr = D.logits(state['deep'], data), R.logits(state['rich'], data)
    zc = D.centred(state['deep'], data, state['ubar'])
    if USE_DEC:
        its = DEC.units(recs)
        tok, low = DEC.lower()
        zf = _pad(DEC.logits(*state['dec'], DEC.states(tok, low, its), its), K)
    else:
        zf = R.logits(state['dd'], data) if state.get('dd') is not None else torch.zeros(M.shape).masked_fill(~M, -1e9)
    bag = W.bags(state["voc"], items, K)
    seen = (bag[3] | ~M).all(1)
    zw = W.logits(state['w'], bag, zd.shape) * seen[:, None]
    za = torch.where(seen[:, None], A.logits(state['attn'], data, A.bank(Ef, items)) + zw, 0.0)
    par = None if USE_HBS else torch.tensor([P.params_for(state['pool'], cal_key(i, s), N_EXP) for i, s in zip(items, seen.tolist())])
    zx = W.logits(state['rel'], REL.bags(state['rvoc'], items, K, W.slot), zd.shape) * seen[:, None]
    xr = RR.inputs(items, RR.bank(items), data[0][:, -data[1].shape[-1]:], data[1], Ho, M, data[5], state['mu'])
    zrf = RR.logits(state['rr'], xr).masked_fill(~M, 0) * seen[:, None]  # full reader: seen-slot expert
    zrv = RR.logits(state['rv'], RR.verify(xr)).masked_fill(~M, 0) * USE_VER  # zero-shot verifier: every structured state
    xn = DNF.device_features(items, state['dn']) if USE_DN else NLI.features(items, state['mu']) if USE_NLI else torch.zeros(*M.shape, NLI.NF)
    zn = NLI.logits(state['nli'], xn, M, NLI.rows(items)).masked_fill(~M, 0)  # NLI reader: every question (zero-shot capable)
    zg = [GLI.device_logits(state['gli'], recs, [i['type'] for i in items]).masked_fill(~M, 0)] if USE_GLI else []
    zrv, zp = PO.device(state.get('poe'), data, Ho, zrv)  # debiased main (slot 7) + bias-only expert (slot 10)
    Z = torch.stack([zd + zw, zr + zw, zf, zc, za, zx, zrf, zrv, zn] + zg + zp + ([OR.logits(state.get('ord'), data)] if USE_ORD else []))
    p = HBS.mix_rows(state['pool'], Z, [cal_key(i, s) for i, s in zip(items, seen.tolist())], N_EXP) if USE_HBS else P.mix(Z, par[:, :N_EXP], par[:, N_EXP:])
    if state['sel'] is not None:
        js = torch.tensor([i['json'] for i in items])
        phi = SEL.features(Z, p, data[5], seen, js)
        p = (SHR.apply_rows(p, phi, state['sel'], seen, js) if USE_SHR else SELG.apply_rows(p, phi, state['sel'], seen, js) if SEL_GROUP
             else SEL.apply_rows(p, phi, state['sel'], seen))
    p = p.double().numpy()
    out, n = [], 0
    for r in recs:
        d = {}
        for qid, q in r['questions'].items():
            row = p[n, :len(q['options'])].astype(float); d[qid] = list(row / row.sum()); n += 1
        out.append(d)
    return out


def size_mb(state):
    if state.get('compact'):
        return LB.size_mb(state, D, RR) + PO.size_mb(state) + OR.size_mb(state)
    return ((SH.size_mb(state['shallow']) if state.get('shallow') else E.size_mb(MODEL, BITS)) + (D.n_params(state['deep']) + R.n_params(state['rich']) + R.n_params(state['attn'])) * 2 / 2 ** 20 + W.size_mb(state['voc']) + W.size_mb(state['rvoc'])
            + (RR.n_params(state['rr']) + RR.n_params(state['rv'])) * 2 / 2 ** 20 + state['tf'].size_mb() + (DEC.size_extra_mb(*state['dec']) if USE_DEC else R.n_params(state['dd']) * 2 / 2 ** 20 if state.get('dd') else 0)
            + (DN.n_params(state['dn']) * 2 / 2 ** 20 if USE_DN else NLI.size_mb() if USE_NLI else 0)
            + state['nli'].numel() * 2 / 2 ** 20 + state['ubar'].numel() * 2 / 2 ** 20 + PO.size_mb(state) + OR.size_mb(state) + (GLI.n_params(state['gli']) * 2 / 2 ** 20 if USE_GLI else 0))
