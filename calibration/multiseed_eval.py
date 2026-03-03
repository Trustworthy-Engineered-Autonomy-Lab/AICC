# Original: evaluation/calibration/_multiseed_eval.py
"""
Multi-seed calibration evaluation with mean +/- std and statistical tests.

Runs the full split->fit->evaluate pipeline N_SEEDS times with different
random seeds for data splitting and calibrator fitting.
Reports mean +/- std for all metrics and Wilcoxon signed-rank tests.

Usage:
    python evaluation/calibration/_multiseed_eval.py [--no-pred-error]
"""
import sys, os, pickle, warnings
sys.path.insert(0, '.'); os.chdir('.')
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors

SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
CACHE = 'evaluation/calibration/results/multi_horizon_cache.pkl'
PHI_C  = 'evaluation/calibration/results/phi_cache.pkl'
HORIZONS = list(range(5, 51, 5))
N_FOLDS = 5
DAC_K_NEIGHBORS = 10
HIST_N_BINS = 15
CNN_FEAT_DIM = 128

ABLATE_NO_PRED_ERR = '--no-pred-error' in sys.argv
if ABLATE_NO_PRED_ERR:
    SIG_IDX = [5,6,7,10,12,13,14,15,16,20,22,23]
    print('*** 12-d features (no pred error) ***')
else:
    SIG_IDX = [1,2,3,5,6,7,10,12,13,14,15,16,18,19,20,21,22,23]

SKIP_VIS_OVERLAP = {"contrast_low", "contrast_high", "bright_high",
                    "motion_blur", "blur_5", "blur_9", "dark_03", "dark_05"}
SKIP_DYN_OVERLAP = {"noisy", "delayed", "burst_hold", "sparse_drop", "frozen"}

with open(CACHE, 'rb') as f: cache = pickle.load(f)
with open(PHI_C, 'rb') as f: phi_data = pickle.load(f)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def nanom(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)

def nanom_raw(v, lo, hi):
    return (v - lo) / max(hi - lo, 1e-8)

def mdist(X, mu, ci):
    d = X - mu; return np.sqrt(np.maximum(np.sum((d @ ci) * d, 1), 0))

def ece(p, g, nb=15):
    if len(p) == 0: return float('nan')
    pred = (p > 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p)
    cor = (pred == g).astype(float); bins = np.linspace(0, 1, nb + 1); e = 0.0
    for b in range(nb):
        lo, hi = bins[b], bins[b + 1]
        m = (conf >= lo) & (conf <= hi) if b == nb - 1 else (conf >= lo) & (conf < hi)
        if m.sum() == 0: continue
        e += m.sum() / len(conf) * abs(conf[m].mean() - cor[m].mean())
    return e

def ace(p, g, nb=15):
    if len(p) == 0: return float('nan')
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    cor = (pred == g).astype(float)
    order = np.argsort(conf); conf_s, cor_s = conf[order], cor[order]
    sz = max(1, len(conf) // nb); e = 0.0; cnt = 0
    for b in range(nb):
        lo = b * sz; hi = (b + 1) * sz if b < nb - 1 else len(conf)
        if hi <= lo: continue
        e += abs(conf_s[lo:hi].mean() - cor_s[lo:hi].mean()); cnt += 1
    return e / max(cnt, 1)

def sce(p, g, nb=15):
    if len(p) == 0: return float('nan')
    bins = np.linspace(0, 1, nb + 1); e = 0.0
    for cls_p, cls_lbl in [(p, 1), (1 - p, 0)]:
        cls_gt = (g == cls_lbl).astype(float)
        for b in range(nb):
            lo, hi = bins[b], bins[b + 1]
            m = (cls_p >= lo) & (cls_p <= hi) if b == nb - 1 else (cls_p >= lo) & (cls_p < hi)
            if m.sum() == 0: continue
            e += m.sum() / len(p) * abs(cls_p[m].mean() - cls_gt[m].mean())
    return e / 2.0

def trapz_horizon_avg(metric_per_K, horizons=HORIZONS):
    ks = np.array(horizons, dtype=float)
    vals = np.array(metric_per_K, dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() < 2: return float(np.nanmean(vals))
    return float(np.trapz(vals[valid], ks[valid]) / (ks[valid][-1] - ks[valid][0]))

def gdata(pf, phi_d, fl, K):
    ps, gs, fs, phis, astds, ameans = [], [], [], [], [], []
    for fn in (fl if fl is not None else pf.keys()):
        if fn not in pf: continue
        e = pf[fn]; ck = e.get('cnn_K', {}); fk = e.get('feats_K', {})
        if K not in ck or K not in fk: continue
        ce, feat, pv = ck[K], fk[K], phi_d.get(fn, [])
        n = min(len(ce), len(feat))
        for i in range(n):
            d = ce[i]
            ps.append(1.0 - d['pred_prob_left'])
            gs.append(d['actual_label'])
            fs.append(feat[i])
            phis.append(pv[i] if i < len(pv) else 0.0)
            astds.append(d.get('aug_std', 0.0))
            ameans.append(1.0 - d.get('aug_mean', d['pred_prob_left']))
    if not ps:
        return (np.array([]), np.array([]), np.zeros((0, 24)),
                np.array([]), np.array([]), np.array([]))
    return (np.array(ps, np.float32), np.array(gs, np.float32),
            np.array(fs, np.float32), np.array(phis, np.float32),
            np.array(astds, np.float32), np.array(ameans, np.float32))

def get_cnn_feats(pf, fl, K):
    feats = []
    for fn in (fl if fl is not None else pf.keys()):
        if fn not in pf: continue
        e = pf[fn]; ck = e.get('cnn_K', {}); fk = e.get('feats_K', {})
        if K not in ck or K not in fk: continue
        ce, feat = ck[K], fk[K]; n = min(len(ce), len(feat))
        for i in range(n):
            feats.append(ce[i].get('cnn_features', np.zeros(CNN_FEAT_DIM, dtype=np.float32)))
    return np.array(feats, dtype=np.float32) if feats else np.zeros((0, CNN_FEAT_DIM), dtype=np.float32)


def run_single_seed(seed):
    rng = np.random.RandomState(seed)
    all_fn = sorted(cache['id'].keys())
    rng.shuffle(all_fn)
    n = len(all_fn); n_test = max(1, int(n * 0.24))
    test_f, dev_f = all_fn[-n_test:], all_fn[:-n_test]
    phi_id = phi_data.get('id', {})

    pk_iso, pk_ts, pk_platt, pk_hb, pk_ts_tta = {}, {}, {}, {}, {}
    for K in HORIZONS:
        p, g, _, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
        if len(p) < 10: continue
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        iso.fit(p, g); pk_iso[K] = iso
        lk, gk = logit(p), g
        def nll_T(T, lk=lk, gk=gk):
            pp = np.clip(sigmoid(lk / T), 1e-7, 1 - 1e-7)
            return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
        pk_ts[K] = minimize_scalar(nll_T, bounds=(0.1, 20), method='bounded').x
        def nll_platt(params, lk=lk, gk=gk):
            pp = np.clip(sigmoid(params[0] * lk + params[1]), 1e-7, 1 - 1e-7)
            return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
        pk_platt[K] = minimize(nll_platt, [1.0, 0.0], method='L-BFGS-B').x
        bin_edges = np.linspace(0, 1, HIST_N_BINS + 1)
        bin_means = np.zeros(HIST_N_BINS, dtype=np.float32)
        for b in range(HIST_N_BINS):
            lo_, hi_ = bin_edges[b], bin_edges[b + 1]
            mask = (p >= lo_) & (p <= hi_) if b == HIST_N_BINS - 1 else (p >= lo_) & (p < hi_)
            bin_means[b] = g[mask].mean() if mask.sum() > 0 else (lo_ + hi_) / 2.0
        pk_hb[K] = (bin_edges, bin_means)
        _, g2, _, _, _, pm = gdata(cache['id'], phi_id, dev_f, K)
        if len(pm) >= 5:
            lk2 = logit(pm)
            def nll_T2(T, lk=lk2, gk=g2):
                pp = np.clip(sigmoid(lk / T), 1e-7, 1 - 1e-7)
                return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
            pk_ts_tta[K] = minimize_scalar(nll_T2, bounds=(0.1, 20), method='bounded').x

    pk_sig, all_phi_list = {}, []
    for K in HORIZONS:
        _, _, feats, phi, _, _ = gdata(cache['id'], phi_id, dev_f, K)
        if len(feats) < 10: continue
        Xd = feats[:, SIG_IDX]; sc = StandardScaler(); Xs = sc.fit_transform(Xd)
        mu = Xs.mean(0); cov = np.cov(Xs, rowvar=False) + np.eye(len(SIG_IDX)) * 1e-6
        pk_sig[K] = (sc, mu, np.linalg.inv(cov)); all_phi_list.append(phi)
    all_phi_arr = np.concatenate(all_phi_list) if all_phi_list else np.array([0.0])
    phi_m, phi_h = all_phi_arr.mean(), all_phi_arr.mean() + 2 * all_phi_arr.std()
    pk_ss = {}
    for K in HORIZONS:
        _, _, feats, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
        if K not in pk_sig or len(feats) == 0: continue
        sc, mu, ci = pk_sig[K]; ds = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
        pk_ss[K] = (ds.mean(), ds.mean() + 2 * ds.std())

    def get_anom(phi, feats, K):
        pn = nanom(phi, phi_m, phi_h)
        if K in pk_sig:
            sc, mu, ci = pk_sig[K]; sd = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
            sm, sh = pk_ss[K]; sn = nanom(sd, sm, sh)
        else: sn = np.zeros_like(pn)
        return pn, sn

    def get_anom_raw(phi, feats, K):
        pn = nanom_raw(phi, phi_m, phi_h)
        if K in pk_sig:
            sc, mu, ci = pk_sig[K]; sd = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
            sm, sh = pk_ss[K]; sn = nanom_raw(sd, sm, sh)
        else: sn = np.zeros_like(pn)
        return pn, sn

    fold_size = len(dev_f) // N_FOLDS; folds = []
    dev_s = list(dev_f); np.random.RandomState(seed).shuffle(dev_s)
    for i in range(N_FOLDS):
        s = i * fold_size; e = s + fold_size if i < N_FOLDS - 1 else len(dev_s)
        folds.append(dev_s[s:e])
    pool_keys = ['logit_p', 'g', 'phi_n', 'sig_n', 'K_n', 'tta_p']
    pool = {k: [] for k in pool_keys}; n_cv = 0
    for fi in range(N_FOLDS):
        vf = folds[fi]; tf = [f for fj in range(N_FOLDS) if fj != fi for f in folds[fj]]
        fp, fs2 = [], {}
        for K in HORIZONS:
            _, _, feats, phi, _, _ = gdata(cache['id'], phi_id, tf, K)
            if len(feats) < 5: continue
            Xd = feats[:, SIG_IDX]; sc = StandardScaler(); Xs = sc.fit_transform(Xd)
            mu = Xs.mean(0); cov = np.cov(Xs, rowvar=False) + np.eye(len(SIG_IDX)) * 1e-6
            fs2[K] = (sc, mu, np.linalg.inv(cov)); fp.append(phi)
        fp = np.concatenate(fp) if fp else np.array([0.0])
        fpm, fph = fp.mean(), fp.mean() + 2 * fp.std(); fss = {}
        for K in HORIZONS:
            _, _, feats, _, _, _ = gdata(cache['id'], phi_id, tf, K)
            if K not in fs2 or len(feats) == 0: continue
            sc, mu, ci = fs2[K]; ds2 = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
            fss[K] = (ds2.mean(), ds2.mean() + 2 * ds2.std())
        for K in HORIZONS:
            p, g, feats, phi, _, amean = gdata(cache['id'], phi_id, vf, K)
            if len(p) == 0 or K not in fs2: continue
            nn_ = min(len(p), len(phi), len(feats), len(amean))
            p, g, feats, phi, amean = p[:nn_], g[:nn_], feats[:nn_], phi[:nn_], amean[:nn_]
            pn = nanom(phi, fpm, fph); sc, mu, ci = fs2[K]
            sd = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci); sm, sh = fss[K]; sn = nanom(sd, sm, sh)
            pool['logit_p'].append(logit(p)); pool['g'].append(g)
            pool['phi_n'].append(pn); pool['sig_n'].append(sn)
            pool['K_n'].append(np.full(nn_, K / 50.0)); pool['tta_p'].append(amean); n_cv += nn_

    phi_ood = phi_data.get('ood_v2', {}); n_ood = 0
    for src, src_data in [('ood', cache.get('ood', {})), ('ood_visual', cache.get('ood_visual', {}))]:
        for tname, tfiles in src_data.items():
            if src == 'ood_visual' and tname in SKIP_VIS_OVERLAP: continue
            if src == 'ood' and tname in SKIP_DYN_OVERLAP: continue
            for K in HORIZONS:
                p, g, feats, phi, _, amean_o = gdata(tfiles, phi_ood, None, K)
                if len(p) == 0 or K not in pk_sig: continue
                nn_ = min(len(p), len(phi), len(feats), len(amean_o))
                p, g, feats, phi, amean_o = p[:nn_], g[:nn_], feats[:nn_], phi[:nn_], amean_o[:nn_]
                if len(phi) > 0 and np.all(phi == 0): phi = feats[:, 0].copy()
                pn, sn = get_anom(phi, feats, K)
                pool['logit_p'].append(logit(p)); pool['g'].append(g)
                pool['phi_n'].append(pn); pool['sig_n'].append(sn)
                pool['K_n'].append(np.full(nn_, K / 50.0)); pool['tta_p'].append(amean_o); n_ood += nn_

    for k in pool: pool[k] = np.concatenate(pool[k])
    N = len(pool['g'])
    rng2 = np.random.RandomState(seed + 1000)
    id_idx = np.arange(n_cv); ood_idx = np.arange(n_cv, N)
    if n_ood > n_cv * 2:
        cap = n_cv * 3
        ood_sub = rng2.choice(ood_idx, size=min(cap, len(ood_idx)), replace=False)
        rep = max(1, len(ood_sub) // n_cv)
        bal_idx = np.concatenate([np.tile(id_idx, rep), ood_sub]); rng2.shuffle(bal_idx)
        for k in pool: pool[k] = pool[k][bal_idx]; N = len(pool['g'])
    rng3 = np.random.RandomState(seed + 2000)
    idx = np.arange(N); rng3.shuffle(idx)
    ntr = int(N * 0.85); tri, vai = idx[:ntr], idx[ntr:]

    K_int = (pool['K_n'] * 50).round().astype(int)
    pool_Tk = np.array([pk_ts.get(k, 1.0) for k in K_int], dtype=np.float32)
    pool_p_base = sigmoid(pool['logit_p'] / pool_Tk)
    pool_tta_logit = logit(pool['tta_p'])
    pool_tta_Tk = np.array([pk_ts_tta.get(k, 1.0) for k in K_int], dtype=np.float32)

    def fit_shrinkage():
        best_r, best_v = None, 1e9
        for init in [np.zeros(3), np.ones(3)*0.5, np.array([1.,1.,-1.])]:
            def nll(params):
                alpha = sigmoid(params[0]*pool['phi_n'][tri]+params[1]*pool['sig_n'][tri]+params[2])
                pc = np.clip((1-alpha)*pool_p_base[tri]+alpha*0.5, 1e-7, 1-1e-7)
                return -np.mean(pool['g'][tri]*np.log(pc)+(1-pool['g'][tri])*np.log(1-pc))
            r = minimize(nll, init, method='L-BFGS-B')
            def nll_v(params):
                alpha = sigmoid(params[0]*pool['phi_n'][vai]+params[1]*pool['sig_n'][vai]+params[2])
                pc = np.clip((1-alpha)*pool_p_base[vai]+alpha*0.5, 1e-7, 1-1e-7)
                return -np.mean(pool['g'][vai]*np.log(pc)+(1-pool['g'][vai])*np.log(1-pc))
            v = nll_v(r.x)
            if v < best_v: best_v = v; best_r = r
        return best_r.x

    def fit_tta_w(signal_tr, signal_vai):
        def nll(w):
            Te = pool_tta_Tk[tri]*np.exp(np.clip(w*signal_tr, -10, 10))
            pc = np.clip(sigmoid(pool_tta_logit[tri]/np.clip(Te, 0.05, 500)), 1e-7, 1-1e-7)
            return -np.mean(pool['g'][tri]*np.log(pc)+(1-pool['g'][tri])*np.log(1-pc))
        return minimize_scalar(nll, bounds=(0, 10), method='bounded').x

    def fit_tta_2w():
        best_r, best_v = None, 1e9
        for init in [np.zeros(2), np.array([0.5,0.5]), np.array([1.,0.])]:
            def nll(params):
                Te = pool_tta_Tk[tri]*np.exp(np.clip(params[0]*pool['phi_n'][tri]+params[1]*pool['sig_n'][tri], -10, 10))
                pc = np.clip(sigmoid(pool_tta_logit[tri]/np.clip(Te, 0.05, 500)), 1e-7, 1-1e-7)
                return -np.mean(pool['g'][tri]*np.log(pc)+(1-pool['g'][tri])*np.log(1-pc))
            r = minimize(nll, init, method='L-BFGS-B', bounds=[(0,None),(0,None)])
            def nll_v(params):
                Te = pool_tta_Tk[vai]*np.exp(np.clip(params[0]*pool['phi_n'][vai]+params[1]*pool['sig_n'][vai], -10, 10))
                pc = np.clip(sigmoid(pool_tta_logit[vai]/np.clip(Te, 0.05, 500)), 1e-7, 1-1e-7)
                return -np.mean(pool['g'][vai]*np.log(pc)+(1-pool['g'][vai])*np.log(1-pc))
            v = nll_v(r.x)
            if v < best_v: best_v = v; best_r = r
        return best_r.x

    def fit_raw_2w():
        best_r, best_v = None, 1e9
        for init in [np.zeros(2), np.array([0.5,0.5]), np.array([1.,0.])]:
            def nll(params):
                Te = pool_Tk[tri]*np.exp(np.clip(params[0]*pool['phi_n'][tri]+params[1]*pool['sig_n'][tri], -10, 10))
                pc = np.clip(sigmoid(pool['logit_p'][tri]/np.clip(Te, 0.05, 500)), 1e-7, 1-1e-7)
                return -np.mean(pool['g'][tri]*np.log(pc)+(1-pool['g'][tri])*np.log(1-pc))
            r = minimize(nll, init, method='L-BFGS-B', bounds=[(0,None),(0,None)])
            def nll_v(params):
                Te = pool_Tk[vai]*np.exp(np.clip(params[0]*pool['phi_n'][vai]+params[1]*pool['sig_n'][vai], -10, 10))
                pc = np.clip(sigmoid(pool['logit_p'][vai]/np.clip(Te, 0.05, 500)), 1e-7, 1-1e-7)
                return -np.mean(pool['g'][vai]*np.log(pc)+(1-pool['g'][vai])*np.log(1-pc))
            v = nll_v(r.x)
            if v < best_v: best_v = v; best_r = r
        return best_r.x

    shrink_p = fit_shrinkage()
    w_sigma = fit_tta_w(pool['sig_n'][tri], pool['sig_n'][vai])
    w_phi = fit_tta_w(pool['phi_n'][tri], pool['phi_n'][vai])
    w_ps = fit_tta_2w()
    w_raw_ps = fit_raw_2w()

    dac_knns, dac_stats_d, dac_Tk = {}, {}, {}
    for K in HORIZONS:
        cf = get_cnn_feats(cache['id'], dev_f, K)
        if len(cf) < DAC_K_NEIGHBORS+1 or cf.shape[1]==0 or np.all(cf==0): continue
        knn = NearestNeighbors(n_neighbors=DAC_K_NEIGHBORS, metric='euclidean', n_jobs=-1)
        knn.fit(cf); d_tr, _ = knn.kneighbors(cf); md_tr = d_tr.mean(axis=1)
        dac_knns[K] = knn; dac_stats_d[K] = (md_tr.mean(), md_tr.mean()+2*md_tr.std())
        dac_Tk[K] = pk_ts.get(K, 1.0)

    def ap_platt(p_raw, K):
        if K not in pk_platt: return p_raw.copy()
        a, b = pk_platt[K]; return sigmoid(a*logit(p_raw)+b)
    def ap_hb(p_raw, K):
        if K not in pk_hb: return p_raw.copy()
        edges, means = pk_hb[K]; ix = np.digitize(p_raw, edges[1:-1])
        return means[np.clip(ix, 0, len(means)-1)].astype(np.float32)
    def ap_shrink(p_raw, pn, sn, K):
        Tk = pk_ts.get(K, 1.0); pb = sigmoid(logit(p_raw)/Tk)
        alpha = sigmoid(shrink_p[0]*pn+shrink_p[1]*sn+shrink_p[2])
        return (1-alpha)*pb+alpha*0.5
    def ap_tta(p_tta, K): return sigmoid(logit(p_tta)/pk_ts_tta.get(K, 1.0))
    def ap_tta_s(p_tta, sn, K):
        Tk = pk_ts_tta.get(K, 1.0); Te = Tk*np.exp(np.clip(w_sigma*sn, -10, 10))
        return sigmoid(logit(p_tta)/np.clip(Te, 0.05, 500))
    def ap_tta_p(p_tta, pn, K):
        Tk = pk_ts_tta.get(K, 1.0); Te = Tk*np.exp(np.clip(w_phi*pn, -10, 10))
        return sigmoid(logit(p_tta)/np.clip(Te, 0.05, 500))
    def ap_tta_ps(p_tta, pn, sn, K):
        Tk = pk_ts_tta.get(K, 1.0); Te = Tk*np.exp(np.clip(w_ps[0]*pn+w_ps[1]*sn, -10, 10))
        return sigmoid(logit(p_tta)/np.clip(Te, 0.05, 500))
    def ap_raw_ps(p_raw, pn, sn, K):
        Tk = pk_ts.get(K, 1.0); Te = Tk*np.exp(np.clip(w_raw_ps[0]*pn+w_raw_ps[1]*sn, -10, 10))
        return sigmoid(logit(p_raw)/np.clip(Te, 0.05, 500))
    def ap_dac(p_raw, cf, K):
        if K not in dac_knns: return sigmoid(logit(p_raw)/pk_ts.get(K, 1.0))
        d_t, _ = dac_knns[K].kneighbors(cf); md = d_t.mean(axis=1)
        mu_d, hi_d = dac_stats_d[K]; dn = np.clip((md-mu_d)/max(hi_d-mu_d, 1e-8), 0, 1)
        Te = dac_Tk[K]*np.exp(np.clip(2.0*dn, -10, 10))
        return sigmoid(logit(p_raw)/np.clip(Te, 0.05, 500))

    datasets = {}
    for dt in ['real_dark', 'real_blur', 'real_bias', 'real_latency']:
        if dt in cache and cache[dt]:
            datasets[dt.replace('real_', '').capitalize()] = (cache[dt], None, phi_data.get(dt, {}))
    datasets['ID'] = (cache['id'], test_f, phi_id)

    mlist = ['Raw', 'TempScale', 'Platt', 'Isotonic', 'HistBin', 'DAC',
             'Shrink', 'Raw+ps', 'TTA', 'TTA+phi', 'TTA+sigma', 'TTA+ps']
    ood_ds_list = ['Dark', 'Blur', 'Bias', 'Latency']

    R_ece = {m: {dn: {} for dn in datasets} for m in mlist}
    id_phi_all, id_sig_all = [], []
    ood_phi_all, ood_sig_all = {}, {}

    for dn, (dd, df, dp) in datasets.items():
        ood_phi_dn, ood_sig_dn = [], []
        for K in HORIZONS:
            p, g, feats, phi, _, amean = gdata(dd, dp, df, K)
            if len(p) == 0 or K not in pk_iso or K not in pk_sig:
                for m in mlist: R_ece[m][dn][K] = float('nan')
                continue
            nn_ = min(len(p), len(phi), len(feats), len(amean))
            p, g, feats, phi, amean = p[:nn_], g[:nn_], feats[:nn_], phi[:nn_], amean[:nn_]
            pn, sn = get_anom(phi, feats, K)
            pn_r, sn_r = get_anom_raw(phi, feats, K)
            if dn == 'ID': id_phi_all.append(pn_r); id_sig_all.append(sn_r)
            else: ood_phi_dn.append(pn_r); ood_sig_dn.append(sn_r)

            def ev(name, p_cal): R_ece[name][dn][K] = ece(p_cal, g)
            ev('Raw', p)
            lp = logit(p)
            ev('TempScale', sigmoid(lp/pk_ts[K]) if K in pk_ts else p)
            ev('Platt', ap_platt(p, K))
            ev('Isotonic', np.clip(pk_iso[K].predict(p), 0.01, 0.99))
            ev('HistBin', np.clip(ap_hb(p, K), 0.01, 0.99))
            cf = get_cnn_feats(dd, df, K)[:nn_]
            ev('DAC', ap_dac(p, cf, K))
            ev('Shrink', ap_shrink(p, pn, sn, K))
            ev('Raw+ps', ap_raw_ps(p, pn, sn, K))
            ev('TTA', ap_tta(amean, K))
            ev('TTA+phi', ap_tta_p(amean, pn, K))
            ev('TTA+sigma', ap_tta_s(amean, sn, K))
            ev('TTA+ps', ap_tta_ps(amean, pn, sn, K))
        if dn != 'ID' and ood_phi_dn:
            ood_phi_all[dn] = np.concatenate(ood_phi_dn)
            ood_sig_all[dn] = np.concatenate(ood_sig_dn)

    id_phi_cat = np.concatenate(id_phi_all) if id_phi_all else np.array([])
    id_sig_cat = np.concatenate(id_sig_all) if id_sig_all else np.array([])

    ece_avg = {}
    for m in mlist:
        ece_avg[m] = {}
        for dn in datasets:
            ece_avg[m][dn] = trapz_horizon_avg([R_ece[m][dn].get(K, float('nan')) for K in HORIZONS])

    ood_auroc = {}
    for dn in ood_ds_list:
        if dn not in ood_phi_all or len(id_phi_cat) == 0: continue
        y_true = np.concatenate([np.zeros(len(id_phi_cat)), np.ones(len(ood_phi_all[dn]))])
        for sn_, id_s, ood_s in [
            ('phi', id_phi_cat, ood_phi_all[dn]),
            ('sigma', id_sig_cat, ood_sig_all[dn]),
            ('phi+sigma', (id_phi_cat+id_sig_cat)/2, (ood_phi_all[dn]+ood_sig_all[dn])/2)]:
            scores = np.concatenate([id_s, ood_s])
            ood_auroc[(dn, sn_)] = roc_auc_score(y_true, scores) if len(np.unique(y_true))>=2 else float('nan')

    per_k_ece = {}
    for m in mlist:
        per_k_ece[m] = []
        for dn in ood_ds_list:
            for K in HORIZONS:
                per_k_ece[m].append(R_ece[m][dn].get(K, float('nan')))

    return {'ece_avg': ece_avg, 'ood_auroc': ood_auroc, 'per_k_ece': per_k_ece,
            'w_ps': w_ps.copy(), 'w_sigma': w_sigma, 'w_phi': w_phi}


print('='*70)
print('MULTI-SEED EVALUATION (%d seeds)' % N_SEEDS)
print('SIG_IDX: %s (%d features)' % (SIG_IDX, len(SIG_IDX)))
print('='*70)

all_results = []
for si, seed in enumerate(SEEDS):
    print('\n--- Seed %d/%d (seed=%d) ---' % (si+1, N_SEEDS, seed))
    res = run_single_seed(seed)
    ours_ood = np.mean([res['ece_avg']['TTA+ps'][dn] for dn in ['Dark','Blur','Bias','Latency']])
    print('  OOD Avg ECE (Ours): %.4f' % ours_ood)
    all_results.append(res)

mlist = ['Raw', 'TempScale', 'Platt', 'Isotonic', 'HistBin', 'DAC',
         'Shrink', 'Raw+ps', 'TTA', 'TTA+phi', 'TTA+sigma', 'TTA+ps']
ood_ds = ['Dark', 'Blur', 'Bias', 'Latency']
all_ds = ['ID', 'Dark', 'Blur', 'Bias', 'Latency']

print('\n'+'='*70)
print('RESULTS: mean +/- std over %d seeds' % N_SEEDS)
print('='*70)

print('\nECE (horizon-averaged, trapz)')
hdr = '%-16s' % 'Method'
for dn in all_ds: hdr += '  %-18s' % dn
hdr += '  %-18s' % 'OOD Avg'
print(hdr); print('-'*len(hdr))
for m in mlist:
    row = '%-16s' % m; ood_seeds = []
    for dn in all_ds:
        vals = [r['ece_avg'][m][dn] for r in all_results]
        row += '  %.3f +/- %.3f    ' % (np.mean(vals), np.std(vals))
        if dn != 'ID': ood_seeds.append(vals)
    ood_m = [np.mean([ood_seeds[j][i] for j in range(4)]) for i in range(N_SEEDS)]
    row += '  %.3f +/- %.3f' % (np.mean(ood_m), np.std(ood_m))
    print(row)

print('\nOOD Detection AUROC')
hdr = '%-16s' % 'Signal'
for dn in ood_ds: hdr += '  %-18s' % dn
hdr += '  %-18s' % 'Mean'
print(hdr); print('-'*len(hdr))
for sig in ['phi', 'sigma', 'phi+sigma']:
    row = '%-16s' % sig; sm = []
    for dn in ood_ds:
        vals = [r['ood_auroc'].get((dn, sig), float('nan')) for r in all_results]
        row += '  %.3f +/- %.3f    ' % (np.mean(vals), np.std(vals))
        sm.append(vals)
    mm = [np.mean([sm[j][i] for j in range(4)]) for i in range(N_SEEDS)]
    row += '  %.3f +/- %.3f' % (np.mean(mm), np.std(mm))
    print(row)

print('\n'+'='*70)
print('WILCOXON SIGNED-RANK TESTS (Ours vs baseline)')
print('Paired on %d points (4 OOD x %d horizons x %d seeds)' % (4*len(HORIZONS)*N_SEEDS, len(HORIZONS), N_SEEDS))
print('='*70)
ours_key = 'TTA+ps'
for baseline in ['TempScale', 'DAC', 'TTA', 'TTA+sigma', 'TTA+phi']:
    all_diffs = []
    for r in all_results:
        ov = np.array(r['per_k_ece'][ours_key])
        bv = np.array(r['per_k_ece'][baseline])
        valid = ~(np.isnan(ov) | np.isnan(bv))
        all_diffs.append(bv[valid] - ov[valid])
    pooled = np.concatenate(all_diffs)
    if len(pooled) > 0 and np.any(pooled != 0):
        stat, pval = wilcoxon(pooled, alternative='greater')
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        print('  vs %-12s  diff=%.4f  W=%.0f  p=%.2e  %s' % (baseline, np.mean(pooled), stat, pval, sig))
    else:
        print('  vs %-12s  no valid differences' % baseline)

print('\nLearned params (mean +/- std):')
print('  TTA+phi+sigma: w_phi=%.3f+/-%.3f  w_sigma=%.3f+/-%.3f' % (
    np.mean([r['w_ps'][0] for r in all_results]), np.std([r['w_ps'][0] for r in all_results]),
    np.mean([r['w_ps'][1] for r in all_results]), np.std([r['w_ps'][1] for r in all_results])))

print('\nDone!')
