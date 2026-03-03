# Original: evaluation/calibration/_id_vs_ood.py
"""
Anomaly-aware Calibration Evaluation (v3 - TTA + dynamics)

Methods:
  Raw        : uncalibrated CNN predictions
  TempScale  : Per-K Temperature Scaling on raw CNN output
  Isotonic   : Per-K Isotonic Regression
  Shrink     : Per-K TS + anomaly shrinkage toward 0.5 (phi+sigma) [old approach]
  TTA        : Test-Time Augmentation mean + Per-K TS
  TTA+sigma  : TTA with dynamics-conditional temperature
               T_eff = T_K_tta * exp(w * sigma_n)
               Normal -> T_K_tta (can SHARPEN if T<1 -> confidence UP)
               Dynamics anomaly -> T increases -> confidence DOWN
  TTA+phi+sigma : same but also includes phi

Key insight: TTA naturally handles visual anomalies --
  correct predictions are augmentation-robust (stay confident),
  wrong predictions are fragile (become uncertain).
  Unlike shrinkage (which only reduces confidence), logit-space scaling
  can both increase AND decrease confidence.
"""
import sys, os, pickle, warnings, json
sys.path.insert(0, '.'); os.chdir('.')
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar, minimize
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEED = 42; np.random.seed(SEED)
CACHE = 'evaluation/calibration/results/multi_horizon_cache.pkl'
PHI_C = 'evaluation/calibration/results/phi_cache.pkl'
OUT   = 'evaluation/calibration/results'
OUT_V2 = 'evaluation/calibration/results/final_plots_v2'
HORIZONS = list(range(5, 51, 5))

ABLATE_NO_PRED_ERR = '--no-pred-error' in sys.argv
if ABLATE_NO_PRED_ERR:
    SIG_IDX = [5,6,7,10,12,13,14,15,16,20,22,23]
    print('*** ABLATION: prediction-error features REMOVED from SIG_IDX ***')
    print('    SIG_IDX =', SIG_IDX, '(%d features)' % len(SIG_IDX))
else:
    SIG_IDX = [1,2,3,5,6,7,10,12,13,14,15,16,18,19,20,21,22,23]

with open(CACHE, 'rb') as f: cache = pickle.load(f)
with open(PHI_C, 'rb') as f: phi_data = pickle.load(f)

def gdata(pf, phi_d, fl, K):
    """Extract per-horizon data: raw_p, gt, features, phi, aug_std, tta_p"""
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

CNN_FEAT_DIM = 128

def get_cnn_feats(pf, fl, K):
    """Extract CNN penultimate features (128-d) aligned with gdata() output order.
    Used by DAC (Tomani et al.) which estimates density from network hidden features."""
    feats = []
    for fn in (fl if fl is not None else pf.keys()):
        if fn not in pf: continue
        e = pf[fn]; ck = e.get('cnn_K', {}); fk = e.get('feats_K', {})
        if K not in ck or K not in fk: continue
        ce, feat = ck[K], fk[K]
        n = min(len(ce), len(feat))
        for i in range(n):
            d = ce[i]
            feats.append(d.get('cnn_features', np.zeros(CNN_FEAT_DIM, dtype=np.float32)))
    return np.array(feats, dtype=np.float32) if feats else np.zeros((0, CNN_FEAT_DIM), dtype=np.float32)

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
    """Adaptive Calibration Error -- equal-mass (adaptive) bins."""
    if len(p) == 0: return float('nan')
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    cor = (pred == g).astype(float)
    order = np.argsort(conf)
    conf_s, cor_s = conf[order], cor[order]
    sz = max(1, len(conf) // nb)
    e = 0.0; cnt = 0
    for b in range(nb):
        lo = b * sz; hi = (b + 1) * sz if b < nb - 1 else len(conf)
        if hi <= lo: continue
        e += abs(conf_s[lo:hi].mean() - cor_s[lo:hi].mean()); cnt += 1
    return e / max(cnt, 1)

def sce(p, g, nb=15):
    """Static Calibration Error -- per-class calibration for both classes."""
    if len(p) == 0: return float('nan')
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    for cls_p, cls_lbl in [(p, 1), (1 - p, 0)]:
        cls_gt = (g == cls_lbl).astype(float)
        for b in range(nb):
            lo, hi = bins[b], bins[b + 1]
            m = (cls_p >= lo) & (cls_p <= hi) if b == nb - 1 else (cls_p >= lo) & (cls_p < hi)
            if m.sum() == 0: continue
            e += m.sum() / len(p) * abs(cls_p[m].mean() - cls_gt[m].mean())
    return e / 2.0

def compute_aurc(p_cal, correct):
    """Area Under Risk-Coverage curve (Geifman & El-Yaniv 2017)."""
    if len(p_cal) == 0: return float('nan')
    conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
    n = len(conf)
    order = np.argsort(-conf)
    correct_s = correct[order]
    cum_err = np.cumsum(1 - correct_s)
    coverages = np.arange(1, n + 1) / n
    risks = cum_err / np.arange(1, n + 1)
    return float(np.trapz(risks, coverages))

def compute_eaurc(p_cal, correct):
    """Excess-AURC = AURC - optimal AURC (oracle ranking)."""
    if len(p_cal) == 0: return float('nan')
    aurc = compute_aurc(p_cal, correct)
    n = len(correct); n_err = int((1 - correct).sum())
    if n_err == 0: return aurc
    opt_risks = np.zeros(n)
    for i in range(n):
        remaining_err = max(0, n_err - max(0, n - 1 - i))
        opt_risks[i] = remaining_err / (i + 1)
    coverages = np.arange(1, n + 1) / n
    kappa_star = float(np.trapz(opt_risks, coverages))
    return aurc - kappa_star

def trapz_horizon_avg(metric_per_K, horizons=HORIZONS):
    """Trapezoidal integration of a metric over prediction horizons,
    normalized by the horizon range (k_max - k_min).
    Equivalent to the area under the metric-vs-horizon curve."""
    ks = np.array(horizons, dtype=float)
    vals = np.array(metric_per_K, dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() < 2:
        return float(np.nanmean(vals))
    return float(np.trapz(vals[valid], ks[valid]) / (ks[valid][-1] - ks[valid][0]))

def compute_augrc(p_cal, correct, n_points=200):
    """Area Under Generalized Risk-Coverage (Jaeger et al. 2023).
    Normalized so random confidence yields 0.5 * err_rate."""
    if len(p_cal) == 0: return float('nan')
    conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
    n = len(conf)
    order = np.argsort(-conf)
    correct_s = correct[order].astype(float)
    cum_err = np.cumsum(1 - correct_s)
    risks = cum_err / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / n
    return float(np.trapz(risks, coverages))

def logit(p, eps=1e-6): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
def nanom(v, lo, hi): return np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)
def nanom_raw(v, lo, hi): return (v - lo) / max(hi - lo, 1e-8)
def mdist(X, mu, ci): d = X - mu; return np.sqrt(np.maximum(np.sum((d @ ci) * d, 1), 0))

def compute_fcr(p_cal, correct, tau):
    conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
    mask = conf >= tau
    return 1.0 - correct[mask].mean() if mask.sum() > 0 else float('nan')

def compute_safe_abstention(p_cal, correct, tau):
    wrong = correct == 0
    if wrong.sum() == 0: return float('nan')
    conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
    return (conf[wrong] < tau).mean()

def compute_risk_coverage(p_cal, correct, n_points=50):
    conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
    thresholds = np.linspace(0.5, 1.0, n_points)
    covs, risks = [], []
    for t in thresholds:
        mask = conf >= t; cov = mask.mean()
        risk = (1.0 - correct[mask].mean()) if mask.sum() > 0 else 0.0
        covs.append(cov); risks.append(risk)
    return np.array(covs), np.array(risks)

def auc_rc(covs, risks):
    idx = np.argsort(covs)
    return float(np.trapz(risks[idx], covs[idx]))

all_fn = sorted(cache['id'].keys())
rng = np.random.RandomState(SEED); rng.shuffle(all_fn)
n = len(all_fn); n_test = max(1, int(n * 0.24))
test_f, dev_f = all_fn[-n_test:], all_fn[:-n_test]
phi_id = phi_data.get('id', {})
print('ID files: total=%d dev=%d test=%d' % (n, len(dev_f), len(test_f)))

pk_iso = {}
for K in HORIZONS:
    p, g, _, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
    if len(p) < 10: continue
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    iso.fit(p, g); pk_iso[K] = iso

pk_ts = {}
for K in HORIZONS:
    p, g, _, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
    if len(p) < 5: continue
    lk, gk = logit(p), g
    def nll_T(T, lk=lk, gk=gk):
        pp = np.clip(sigmoid(lk / T), 1e-7, 1 - 1e-7)
        return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
    pk_ts[K] = minimize_scalar(nll_T, bounds=(0.1, 20), method='bounded').x
print('Per-K TS (raw): %d horizons, T range [%.2f, %.2f]' % (
    len(pk_ts), min(pk_ts.values()), max(pk_ts.values())))

pk_platt = {}
for K in HORIZONS:
    p, g, _, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
    if len(p) < 5: continue
    lk, gk = logit(p), g
    def nll_platt(params, lk=lk, gk=gk):
        pp = np.clip(sigmoid(params[0] * lk + params[1]), 1e-7, 1 - 1e-7)
        return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
    res_pl = minimize(nll_platt, [1.0, 0.0], method='L-BFGS-B')
    pk_platt[K] = res_pl.x
print('Per-K Platt: %d horizons' % len(pk_platt))

def apply_platt(p_raw, K):
    if K not in pk_platt: return p_raw.copy()
    a, b = pk_platt[K]
    return sigmoid(a * logit(p_raw) + b)

HIST_N_BINS = 15
pk_hb = {}
for K in HORIZONS:
    p, g, _, _, _, _ = gdata(cache['id'], phi_id, dev_f, K)
    if len(p) < 10: continue
    bin_edges = np.linspace(0, 1, HIST_N_BINS + 1)
    bin_means = np.zeros(HIST_N_BINS, dtype=np.float32)
    bin_counts = np.zeros(HIST_N_BINS, dtype=int)
    for b in range(HIST_N_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (p >= lo) & (p <= hi) if b == HIST_N_BINS - 1 else (p >= lo) & (p < hi)
        if mask.sum() > 0:
            bin_means[b] = g[mask].mean()
            bin_counts[b] = mask.sum()
        else:
            bin_means[b] = (lo + hi) / 2.0
    pk_hb[K] = (bin_edges, bin_means)

def apply_histbin(p_raw, K):
    if K not in pk_hb: return p_raw.copy()
    edges, means = pk_hb[K]
    idx = np.digitize(p_raw, edges[1:-1])
    idx = np.clip(idx, 0, len(means) - 1)
    return means[idx].astype(np.float32)


pk_ts_tta = {}
for K in HORIZONS:
    _, g, _, _, _, pm = gdata(cache['id'], phi_id, dev_f, K)
    if len(pm) < 5: continue
    lk_tta, gk = logit(pm), g
    def nll_T_tta(T, lk=lk_tta, gk=gk):
        pp = np.clip(sigmoid(lk / T), 1e-7, 1 - 1e-7)
        return -np.mean(gk * np.log(pp) + (1 - gk) * np.log(1 - pp))
    pk_ts_tta[K] = minimize_scalar(nll_T_tta, bounds=(0.1, 20), method='bounded').x
if pk_ts_tta:
    print('Per-K TS (TTA): %d horizons, T range [%.2f, %.2f]' % (
        len(pk_ts_tta), min(pk_ts_tta.values()), max(pk_ts_tta.values())))
else:
    print('Per-K TS (TTA): no data (aug_mean not in cache?)')

pk_sig, all_phi_list = {}, []
for K in HORIZONS:
    _, _, feats, phi, _, _ = gdata(cache['id'], phi_id, dev_f, K)
    if len(feats) < 10: continue
    Xd = feats[:, SIG_IDX]; sc = StandardScaler(); Xs = sc.fit_transform(Xd)
    mu = Xs.mean(0); cov = np.cov(Xs, rowvar=False) + np.eye(len(SIG_IDX)) * 1e-6
    pk_sig[K] = (sc, mu, np.linalg.inv(cov)); all_phi_list.append(phi)
all_phi_arr = np.concatenate(all_phi_list)
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
        sc, mu, ci = pk_sig[K]
        sd = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
        sm, sh = pk_ss[K]; sn = nanom(sd, sm, sh)
    else:
        sn = np.zeros_like(pn)
    return pn, sn

def get_anom_raw(phi, feats, K):
    pn = nanom_raw(phi, phi_m, phi_h)
    if K in pk_sig:
        sc, mu, ci = pk_sig[K]
        sd = mdist(sc.transform(feats[:, SIG_IDX]), mu, ci)
        sm, sh = pk_ss[K]; sn = nanom_raw(sd, sm, sh)
    else:
        sn = np.zeros_like(pn)
    return pn, sn

N_FOLDS = 5; fold_size = len(dev_f) // N_FOLDS; folds = []
for i in range(N_FOLDS):
    s = i * fold_size; e = s + fold_size if i < N_FOLDS - 1 else len(dev_f)
    folds.append(dev_f[s:e])

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
        pool['K_n'].append(np.full(nn_, K / 50.0))
        pool['tta_p'].append(amean); n_cv += nn_

phi_ood = phi_data.get('ood_v2', {}); n_ood = 0
SKIP_VIS_OVERLAP = {"contrast_low", "contrast_high", "bright_high",
                    "motion_blur", "blur_5", "blur_9", "dark_03", "dark_05"}
SKIP_DYN_OVERLAP = {"noisy", "delayed", "burst_hold", "sparse_drop", "frozen"}
for src, src_data in [('ood', cache.get('ood', {})), ('ood_visual', cache.get('ood_visual', {}))]:
    for tname, tfiles in src_data.items():
        if src == 'ood_visual' and tname in SKIP_VIS_OVERLAP:
            continue
        if src == 'ood' and tname in SKIP_DYN_OVERLAP:
            continue
        for K in HORIZONS:
            p, g, feats, phi, _, amean_o = gdata(tfiles, phi_ood, None, K)
            if len(p) == 0 or K not in pk_sig: continue
            nn_ = min(len(p), len(phi), len(feats), len(amean_o))
            p, g, feats, phi, amean_o = p[:nn_], g[:nn_], feats[:nn_], phi[:nn_], amean_o[:nn_]
            if len(phi) > 0 and np.all(phi == 0): phi = feats[:, 0].copy()
            pn, sn = get_anom(phi, feats, K)
            pool['logit_p'].append(logit(p)); pool['g'].append(g)
            pool['phi_n'].append(pn); pool['sig_n'].append(sn)
            pool['K_n'].append(np.full(nn_, K / 50.0))
            pool['tta_p'].append(amean_o); n_ood += nn_

for k in pool: pool[k] = np.concatenate(pool[k])
N = len(pool['g']); n_id = n_cv
print('Pool: %d (ID=%d OOD=%d)' % (N, n_id, n_ood))

rng2 = np.random.RandomState(SEED); id_idx = np.arange(n_id); ood_idx = np.arange(n_id, N)
if n_ood > n_id * 2:
    cap = n_id * 3
    ood_sub = rng2.choice(ood_idx, size=min(cap, len(ood_idx)), replace=False)
    rep = max(1, len(ood_sub) // n_id)
    bal_idx = np.concatenate([np.tile(id_idx, rep), ood_sub]); rng2.shuffle(bal_idx)
    for k in pool: pool[k] = pool[k][bal_idx]; N = len(pool['g'])
print('Balanced: %d' % N)
rng3 = np.random.RandomState(SEED); idx = np.arange(N); rng3.shuffle(idx)
ntr = int(N * 0.85); tri, vai = idx[:ntr], idx[ntr:]

K_int = (pool['K_n'] * 50).round().astype(int)
pool_Tk = np.array([pk_ts.get(k, 1.0) for k in K_int], dtype=np.float32)
pool_p_base = sigmoid(pool['logit_p'] / pool_Tk)

print('\n--- Training calibrators ---')

def shrinkage_nll(params, p_base, pn, sn, g):
    alpha = sigmoid(params[0] * pn + params[1] * sn + params[2])
    pc = (1 - alpha) * p_base + alpha * 0.5
    pc = np.clip(pc, 1e-7, 1 - 1e-7)
    return -np.mean(g * np.log(pc) + (1 - g) * np.log(1 - pc))

best_res, best_val = None, 1e9
for init in [np.zeros(3), np.ones(3) * 0.5, np.array([1.0, 1.0, -1.0])]:
    res = minimize(shrinkage_nll, init,
        args=(pool_p_base[tri], pool['phi_n'][tri], pool['sig_n'][tri], pool['g'][tri]),
        method='L-BFGS-B')
    vnll = shrinkage_nll(res.x, pool_p_base[vai], pool['phi_n'][vai], pool['sig_n'][vai], pool['g'][vai])
    if vnll < best_val: best_val = vnll; best_res = res
shrink_params = best_res.x
print('Shrink(phi+sigma): w_phi=%.4f w_sigma=%.4f b=%.4f  val_nll=%.4f' % (*shrink_params, best_val))

def apply_shrinkage(p_raw, phi_n, sig_n, K):
    Tk = pk_ts.get(K, 1.0)
    p_base = sigmoid(logit(p_raw) / Tk)
    alpha = sigmoid(shrink_params[0] * phi_n + shrink_params[1] * sig_n + shrink_params[2])
    return (1 - alpha) * p_base + alpha * 0.5, alpha

pool_tta_logit = logit(pool['tta_p'])
pool_tta_Tk = np.array([pk_ts_tta.get(k, 1.0) for k in K_int], dtype=np.float32)

def tta_sigma_nll(w_sigma, tta_logit, Tk_tta, sn, g):
    Te = Tk_tta * np.exp(np.clip(w_sigma * sn, -10, 10))
    pc = sigmoid(tta_logit / np.clip(Te, 0.05, 500))
    pc = np.clip(pc, 1e-7, 1 - 1e-7)
    return -np.mean(g * np.log(pc) + (1 - g) * np.log(1 - pc))

res_tta_s = minimize_scalar(tta_sigma_nll, bounds=(0, 10), method='bounded',
    args=(pool_tta_logit[tri], pool_tta_Tk[tri], pool['sig_n'][tri], pool['g'][tri]))
w_tta_sigma = res_tta_s.x
val_tta_s = tta_sigma_nll(w_tta_sigma, pool_tta_logit[vai], pool_tta_Tk[vai],
                          pool['sig_n'][vai], pool['g'][vai])
print('TTA+sigma: w_sigma=%.4f  val_nll=%.4f' % (w_tta_sigma, val_tta_s))

def tta_phi_nll(w_phi, tta_logit, Tk_tta, pn, g):
    Te = Tk_tta * np.exp(np.clip(w_phi * pn, -10, 10))
    pc = sigmoid(tta_logit / np.clip(Te, 0.05, 500))
    pc = np.clip(pc, 1e-7, 1 - 1e-7)
    return -np.mean(g * np.log(pc) + (1 - g) * np.log(1 - pc))

res_tta_p = minimize_scalar(tta_phi_nll, bounds=(0, 10), method='bounded',
    args=(pool_tta_logit[tri], pool_tta_Tk[tri], pool['phi_n'][tri], pool['g'][tri]))
w_tta_phi = res_tta_p.x
val_tta_p = tta_phi_nll(w_tta_phi, pool_tta_logit[vai], pool_tta_Tk[vai],
                        pool['phi_n'][vai], pool['g'][vai])
print('TTA+phi: w_phi=%.4f  val_nll=%.4f' % (w_tta_phi, val_tta_p))

def tta_phisigma_nll(params, tta_logit, Tk_tta, pn, sn, g):
    Te = Tk_tta * np.exp(np.clip(params[0] * pn + params[1] * sn, -10, 10))
    pc = sigmoid(tta_logit / np.clip(Te, 0.05, 500))
    pc = np.clip(pc, 1e-7, 1 - 1e-7)
    return -np.mean(g * np.log(pc) + (1 - g) * np.log(1 - pc))

best_res2, best_val2 = None, 1e9
for init in [np.zeros(2), np.array([0.5, 0.5]), np.array([1.0, 0.0])]:
    res2 = minimize(tta_phisigma_nll, init,
        args=(pool_tta_logit[tri], pool_tta_Tk[tri],
              pool['phi_n'][tri], pool['sig_n'][tri], pool['g'][tri]),
        method='L-BFGS-B', bounds=[(0, None), (0, None)])
    vnll2 = tta_phisigma_nll(res2.x, pool_tta_logit[vai], pool_tta_Tk[vai],
                             pool['phi_n'][vai], pool['sig_n'][vai], pool['g'][vai])
    if vnll2 < best_val2: best_val2 = vnll2; best_res2 = res2
w_tta_ps = best_res2.x
print('TTA+phi+sigma: w_phi=%.4f w_sigma=%.4f  val_nll=%.4f' % (*w_tta_ps, best_val2))

def apply_tta(p_tta_raw, K):
    Tk = pk_ts_tta.get(K, 1.0)
    return sigmoid(logit(p_tta_raw) / Tk)

def apply_tta_sigma(p_tta_raw, sig_n, K):
    Tk = pk_ts_tta.get(K, 1.0)
    Te = Tk * np.exp(np.clip(w_tta_sigma * sig_n, -10, 10))
    return sigmoid(logit(p_tta_raw) / np.clip(Te, 0.05, 500))

def apply_tta_phi(p_tta_raw, phi_n, K):
    Tk = pk_ts_tta.get(K, 1.0)
    Te = Tk * np.exp(np.clip(w_tta_phi * phi_n, -10, 10))
    return sigmoid(logit(p_tta_raw) / np.clip(Te, 0.05, 500))

def apply_tta_phisigma(p_tta_raw, phi_n, sig_n, K):
    Tk = pk_ts_tta.get(K, 1.0)
    Te = Tk * np.exp(np.clip(w_tta_ps[0] * phi_n + w_tta_ps[1] * sig_n, -10, 10))
    return sigmoid(logit(p_tta_raw) / np.clip(Te, 0.05, 500))

def raw_phisigma_nll(params, logit_p, Tk_raw, pn, sn, g):
    Te = Tk_raw * np.exp(np.clip(params[0] * pn + params[1] * sn, -10, 10))
    pc = sigmoid(logit_p / np.clip(Te, 0.05, 500))
    pc = np.clip(pc, 1e-7, 1 - 1e-7)
    return -np.mean(g * np.log(pc) + (1 - g) * np.log(1 - pc))

best_raw_ps, best_val_raw = None, 1e9
for init in [np.zeros(2), np.array([0.5, 0.5]), np.array([1.0, 0.0])]:
    res_r = minimize(raw_phisigma_nll, init,
        args=(pool['logit_p'][tri], pool_Tk[tri],
              pool['phi_n'][tri], pool['sig_n'][tri], pool['g'][tri]),
        method='L-BFGS-B', bounds=[(0, None), (0, None)])
    vnll_r = raw_phisigma_nll(res_r.x, pool['logit_p'][vai], pool_Tk[vai],
                              pool['phi_n'][vai], pool['sig_n'][vai], pool['g'][vai])
    if vnll_r < best_val_raw: best_val_raw = vnll_r; best_raw_ps = res_r
w_raw_ps = best_raw_ps.x
print('Raw+phi+sigma (w/o TTA): w_phi=%.4f w_sigma=%.4f  val_nll=%.4f' % (*w_raw_ps, best_val_raw))

def apply_raw_phisigma(p_raw, phi_n, sig_n, K):
    Tk = pk_ts.get(K, 1.0)
    Te = Tk * np.exp(np.clip(w_raw_ps[0] * phi_n + w_raw_ps[1] * sig_n, -10, 10))
    return sigmoid(logit(p_raw) / np.clip(Te, 0.05, 500))

print('\nPer-K Temperature (TTA vs Raw):')
for K in sorted(pk_ts_tta.keys()):
    sign = 'SHARPEN' if pk_ts_tta[K] < 1.0 else 'flatten'
    print('  K=%2d: T_tta=%.3f (%s)  T_raw=%.3f' % (K, pk_ts_tta[K], sign, pk_ts.get(K, 1.0)))

DAC_K_NEIGHBORS = 10
dac_knns, dac_stats, dac_Tk = {}, {}, {}
for K in HORIZONS:
    cnn_feats_d = get_cnn_feats(cache['id'], dev_f, K)
    if len(cnn_feats_d) < DAC_K_NEIGHBORS + 1: continue
    if cnn_feats_d.shape[1] == 0 or np.all(cnn_feats_d == 0):
        print('  K=%d: no CNN features in cache, skipping DAC' % K)
        continue
    knn = NearestNeighbors(n_neighbors=DAC_K_NEIGHBORS, metric='euclidean', n_jobs=-1)
    knn.fit(cnn_feats_d)
    d_tr, _ = knn.kneighbors(cnn_feats_d)
    md_tr = d_tr.mean(axis=1)
    dac_knns[K] = knn
    dac_stats[K] = (md_tr.mean(), md_tr.mean() + 2 * md_tr.std())
    dac_Tk[K] = pk_ts.get(K, 1.0)
print('DAC: trained KNN on CNN hidden features for %d horizons (k=%d, dim=%d)' % (
    len(dac_knns), DAC_K_NEIGHBORS, CNN_FEAT_DIM))

def apply_dac(p_raw, cnn_feats_test, K):
    if K not in dac_knns: return sigmoid(logit(p_raw) / pk_ts.get(K, 1.0))
    knn = dac_knns[K]
    d_test, _ = knn.kneighbors(cnn_feats_test)
    md = d_test.mean(axis=1)
    mu_d, hi_d = dac_stats[K]
    dn = np.clip((md - mu_d) / max(hi_d - mu_d, 1e-8), 0, 1)
    Tk = dac_Tk[K]
    Te = Tk * np.exp(np.clip(2.0 * dn, -10, 10))
    return sigmoid(logit(p_raw) / np.clip(Te, 0.05, 500))

print('\n' + '=' * 70)
print('EVALUATION')
print('=' * 70)

datasets = {}
for dt in ['real_dark', 'real_blur', 'real_bias', 'real_latency']:
    if dt in cache and cache[dt]:
        datasets[dt.replace('real_', '').capitalize()] = (cache[dt], None, phi_data.get(dt, {}))
datasets['ID'] = (cache['id'], test_f, phi_id)

PS, SS = chr(966), chr(963)
mlist = ['Raw', 'TempScale', 'Platt', 'Isotonic', 'HistBin', 'DAC',
         'Shrink('+PS+'+'+SS+')', 'Raw+'+PS+'+'+SS, 'TTA', 'TTA+'+PS, 'TTA+'+SS, 'TTA+'+PS+'+'+SS]
ds_order = ['ID', 'Dark', 'Blur', 'Bias', 'Latency']
ood_ds = ['Dark', 'Blur', 'Bias', 'Latency']
ds_lbl = {'ID': 'In-Distribution', 'Dark': 'Dark (Visual)', 'Blur': 'Blur (Visual)',
           'Bias': 'Bias (Dynamics)', 'Latency': 'Latency (Dynamics)'}

R = {m: {ds: {} for ds in datasets} for m in mlist}
R_ACE = {m: {ds: {} for ds in datasets} for m in mlist}
R_SCE = {m: {ds: {} for ds in datasets} for m in mlist}
ACC = {ds: {} for ds in datasets}
AUC = {m: {ds: {} for ds in datasets} for m in mlist}
CAL = {m: {ds: {'p': [], 'c': []} for ds in datasets} for m in mlist}
ALPHA = {ds: {'phi_n': [], 'sig_n': [], 'alpha': []} for ds in datasets}
ALPHA_RAW = {ds: {'phi_r': [], 'sig_r': []} for ds in datasets}

for dn, (dd, df, dp) in datasets.items():
    for K in HORIZONS:
        p, g, feats, phi, _, amean = gdata(dd, dp, df, K)
        if len(p) == 0 or K not in pk_iso or K not in pk_sig:
            for m in mlist: R[m][dn][K] = float('nan'); AUC[m][dn][K] = float('nan')
            ACC[dn][K] = float('nan'); continue
        nn_ = min(len(p), len(phi), len(feats), len(amean))
        p, g, feats, phi, amean = p[:nn_], g[:nn_], feats[:nn_], phi[:nn_], amean[:nn_]
        pn, sn = get_anom(phi, feats, K)
        pred_raw = (p > 0.5).astype(int); correct_raw = (pred_raw == g).astype(int)
        ACC[dn][K] = correct_raw.mean()

        def eval_method(name, p_cal):
            R[name][dn][K] = ece(p_cal, g)
            R_ACE[name][dn][K] = ace(p_cal, g)
            R_SCE[name][dn][K] = sce(p_cal, g)
            pred_m = (p_cal > 0.5).astype(int)
            correct_m = (pred_m == g).astype(int)
            conf = np.where(p_cal > 0.5, p_cal, 1 - p_cal)
            AUC[name][dn][K] = roc_auc_score(correct_m, conf) if len(np.unique(correct_m)) >= 2 else float('nan')
            CAL[name][dn]['p'].append(p_cal); CAL[name][dn]['c'].append(correct_m)

        eval_method('Raw', p)
        lp = logit(p)
        if K in pk_ts:
            eval_method('TempScale', sigmoid(lp / pk_ts[K]))
        else:
            R['TempScale'][dn][K] = float('nan'); AUC['TempScale'][dn][K] = float('nan')
        eval_method('Platt', apply_platt(p, K))
        eval_method('Isotonic', np.clip(pk_iso[K].predict(p), 0.01, 0.99))
        eval_method('HistBin', np.clip(apply_histbin(p, K), 0.01, 0.99))
        cnn_feats_dac = get_cnn_feats(dd, df, K)[:nn_]
        eval_method('DAC', apply_dac(p, cnn_feats_dac, K))

        p_shrink, alpha_s = apply_shrinkage(p, pn, sn, K)
        eval_method('Shrink('+PS+'+'+SS+')', p_shrink)
        ALPHA[dn]['phi_n'].append(pn); ALPHA[dn]['sig_n'].append(sn)
        ALPHA[dn]['alpha'].append(alpha_s)
        pn_r, sn_r = get_anom_raw(phi, feats, K)
        ALPHA_RAW[dn]['phi_r'].append(pn_r); ALPHA_RAW[dn]['sig_r'].append(sn_r)

        eval_method('Raw+'+PS+'+'+SS, apply_raw_phisigma(p, pn, sn, K))
        eval_method('TTA', apply_tta(amean, K))
        eval_method('TTA+'+PS, apply_tta_phi(amean, pn, K))
        eval_method('TTA+'+SS, apply_tta_sigma(amean, sn, K))
        eval_method('TTA+'+PS+'+'+SS, apply_tta_phisigma(amean, pn, sn, K))

for m in mlist:
    for dn in datasets:
        d = CAL[m][dn]
        d['p'] = np.concatenate(d['p']) if d['p'] else np.array([])
        d['c'] = np.concatenate(d['c']) if d['c'] else np.array([])
for dn in datasets:
    for k in ALPHA[dn]:
        ALPHA[dn][k] = np.concatenate(ALPHA[dn][k]) if ALPHA[dn][k] else np.array([])

for dn in datasets:
    for k in ALPHA_RAW[dn]:
        ALPHA_RAW[dn][k] = np.concatenate(ALPHA_RAW[dn][k]) if ALPHA_RAW[dn][k] else np.array([])
from sklearn.metrics import roc_auc_score as sk_auroc, roc_curve as sk_roc
print('\n' + '=' * 70)
print('ANOMALY DETECTION ACCURACY (ID test vs OOD)')
print('=' * 70)

id_phi = ALPHA_RAW['ID']['phi_r']; id_sig = ALPHA_RAW['ID']['sig_r']
if len(id_phi) > 0 and len(id_sig) > 0:
    ad_header = '%-12s | %-8s | %7s | %7s | %7s' % ('OOD', 'Signal', 'AUROC', 'FPR@95', 'Acc')
    print(ad_header); print('-' * len(ad_header))
    ad_results = {}
    for dn in ood_ds:
        ood_phi = ALPHA_RAW[dn]['phi_r']; ood_sig = ALPHA_RAW[dn]['sig_r']
        if len(ood_phi) == 0: continue
        ad_results[dn] = {}
        y_true = np.concatenate([np.zeros(len(id_phi)), np.ones(len(ood_phi))])
        for sig_name, id_s, ood_s in [
            (chr(966), id_phi, ood_phi),
            (chr(963), id_sig, ood_sig),
            (chr(966)+'+'+chr(963), (id_phi + id_sig) / 2, (ood_phi + ood_sig) / 2),
        ]:
            scores = np.concatenate([id_s, ood_s])
            auroc = sk_auroc(y_true, scores) if len(np.unique(y_true)) >= 2 else float('nan')
            fpr, tpr, thrs = sk_roc(y_true, scores)
            fpr95 = fpr[tpr >= 0.95][0] if np.any(tpr >= 0.95) else 1.0
            best_thr = thrs[np.argmax(tpr - fpr)] if len(thrs) > 0 else 0.5
            preds = (scores >= best_thr).astype(int)
            acc = (preds == y_true).mean()
            ad_results[dn][sig_name] = {'auroc': auroc, 'fpr95': fpr95, 'acc': acc}
            print('%-12s | %-8s | %7.4f | %7.4f | %7.4f' % (dn, sig_name, auroc, fpr95, acc))
        print('-' * len(ad_header))

    fig_ad, (ax_aur, ax_fpr) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    fig_ad.subplots_adjust(wspace=0.32, bottom=0.20, top=0.88, left=0.08, right=0.97)
    sig_names = [chr(966), chr(963), chr(966)+'+'+chr(963)]
    sig_labels = ['$\\varphi$ (perception)', '$\\sigma$ (dynamics)', '$\\varphi + \\sigma$ (combined)']
    sig_colors = ['#7B1FA2', '#1976D2', '#2E7D32']
    x_ad = np.arange(len(ood_ds)); w_ad = 0.22
    for si, sn in enumerate(sig_names):
        aurocs = [ad_results.get(dn, {}).get(sn, {}).get('auroc', 0) for dn in ood_ds]
        fpr95s = [ad_results.get(dn, {}).get(sn, {}).get('fpr95', 1) for dn in ood_ds]
        off = (si - 1) * w_ad
        ax_aur.bar(x_ad + off, aurocs, w_ad, label=sig_labels[si], color=sig_colors[si], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax_fpr.bar(x_ad + off, fpr95s, w_ad, color=sig_colors[si], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_aur.set_xticks(x_ad); ax_aur.set_xticklabels([ds_lbl.get(d, d).split(' (')[0] for d in ood_ds], fontsize=8)
    ax_aur.set_ylabel('AUROC '+chr(8593), fontsize=9); ax_aur.set_title('(a) OOD Detection AUROC', fontweight='bold', fontsize=10)
    ax_aur.set_ylim(0.4, 1.0); ax_aur.axhline(0.5, color='#aaa', ls='--', lw=0.6)
    ax_aur.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax_aur.grid(axis='y', alpha=0.15); ax_aur.spines['top'].set_visible(False); ax_aur.spines['right'].set_visible(False)
    ax_fpr.set_xticks(x_ad); ax_fpr.set_xticklabels([ds_lbl.get(d, d).split(' (')[0] for d in ood_ds], fontsize=8)
    ax_fpr.set_ylabel('FPR@95 '+chr(8595), fontsize=9); ax_fpr.set_title('(b) FPR at 95% TPR', fontweight='bold', fontsize=10)
    ax_fpr.set_ylim(0, 1.0); ax_fpr.grid(axis='y', alpha=0.15)
    ax_fpr.spines['top'].set_visible(False); ax_fpr.spines['right'].set_visible(False)
    for ext in ['.png', '.pdf']:
        fig_ad.savefig(os.path.join(OUT_V2, 'ood_detection_auroc_fpr95' + ext), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_ad)
    print('\nSaved:', os.path.join(OUT_V2, 'ood_detection_auroc_fpr95.png'))
else:
    print('  No ID anomaly signals available, skipping.')

print('\nACCURACY (CNN on reconstructed images)')
for ds in ds_order:
    vals = [ACC[ds].get(K, float('nan')) for K in HORIZONS]
    print('  %-12s mean=%.3f' % (ds, np.nanmean(vals)))

print('\nECE (lower = better calibrated)')
hdr = '%-25s' + '  '.join(['%-12s'] * len(ds_order))
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = ['%.4f' % np.nanmean([R[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ds_order]
    print(hdr % (m, *parts))

print('\nACE - Adaptive Calibration Error (lower = better)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = ['%.4f' % np.nanmean([R_ACE[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ds_order]
    print(hdr % (m, *parts))

print('\nSCE - Static Calibration Error (lower = better)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = ['%.4f' % np.nanmean([R_SCE[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ds_order]
    print(hdr % (m, *parts))

print('\nAUGRC - Area Under Generalized Risk-Coverage (lower = better)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = []
    for ds in ds_order:
        d = CAL[m][ds]
        parts.append('%.4f' % compute_augrc(d['p'], d['c']) if len(d['p']) > 0 else 'nan')
    print(hdr % (m, *parts))

print('\nE-AURC - Excess AURC (lower = better)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = []
    for ds in ds_order:
        d = CAL[m][ds]
        parts.append('%.4f' % compute_eaurc(d['p'], d['c']) if len(d['p']) > 0 else 'nan')
    print(hdr % (m, *parts))

print('\nFailure Prediction AUROC (higher = better at ranking correct vs incorrect)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = ['%.4f' % np.nanmean([AUC[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ds_order]
    print(hdr % (m, *parts))

for tau in [0.6, 0.7, 0.8]:
    print('\nSelective Risk @ tau=%.1f (lower = better)' % tau)
    print(hdr % ('Method', *ds_order))
    for m in mlist:
        parts = []
        for ds in ds_order:
            d = CAL[m][ds]
            parts.append('%.4f' % compute_fcr(d['p'], d['c'], tau) if len(d['p']) > 0 else 'nan')
        print(hdr % (m, *parts))

print('\nAURC - Area Under Risk-Coverage (lower = better, Corbiere et al. 2019)')
print(hdr % ('Method', *ds_order))
for m in mlist:
    parts = []
    for ds in ds_order:
        d = CAL[m][ds]
        if len(d['p']) > 0:
            co, ri = compute_risk_coverage(d['p'], d['c'])
            parts.append('%.4f' % auc_rc(co, ri))
        else:
            parts.append('nan')
    print(hdr % (m, *parts))

fk = 'TTA+'+SS
print('\nOOD Average ECE')
for m in mlist:
    vals = [np.nanmean([R[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ood_ds]
    print('  %-25s %.4f' % (m, np.mean(vals)))

print('\nOurs (%s) vs Per-K TS' % fk)
for ds in ds_order:
    eo = np.nanmean([R[fk][ds].get(K, float('nan')) for K in HORIZONS])
    et = np.nanmean([R['TempScale'][ds].get(K, float('nan')) for K in HORIZONS])
    d = eo - et; pct = d / et * 100 if et > 0 else 0
    print('  %-12s TS=%.4f  Ours=%.4f  %+.4f (%+.1f%%)' % (ds, et, eo, d, pct))

print('\nLearned parameters')
print('  Shrink(phi+sigma): w_phi=%.4f w_sigma=%.4f b=%.4f' % tuple(shrink_params))
print('  TTA+sigma: w_sigma=%.4f' % w_tta_sigma)
print('  TTA+phi+sigma: w_phi=%.4f w_sigma=%.4f' % tuple(w_tta_ps))


from sklearn.metrics import roc_auc_score as _auroc, roc_curve as _roc_curve
from scipy.ndimage import laplace as _laplace

def fpr_at_tpr(labels, scores, tpr_target=0.95):
    fpr, tpr, _ = _roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_target)
    return fpr[min(idx, len(fpr) - 1)]

_VIS_NPZ_DIRS = {
    'ID': ['data_renewed/processed_64x64'],
    'Dark': ['data_renewed/processed_64x64_dark'],
    'Blur': ['data_renewed/processed_64x64_blur'],
    'Bias': ['data_renewed/processed_64x64_bias'],
    'Latency': ['data_renewed/processed_64x64_latency'],
}
_VIS_CACHE_MAP = {'ID': 'id', 'Dark': 'real_dark', 'Blur': 'real_blur',
                  'Bias': 'real_bias', 'Latency': 'real_latency'}
_CTX = 100; _MK = 50; _STR = 32; _MW = 30

def _compute_vis_file(npz_path):
    """Compute per-window brightness and Laplacian variance for one NPZ."""
    data = np.load(npz_path)
    frames = data['frame'].astype(np.float32)
    n = len(frames)
    total = _CTX + _MK
    starts = list(range(0, max(1, n - total + 1), _STR))[:_MW]
    bright, lapvar = [], []
    for s in starts:
        ctx = frames[s:s + _CTX]
        if len(ctx) < _CTX:
            break
        bright.append(float(ctx.mean()))
        lap_scores = [float(np.var(_laplace(f.mean(axis=-1) if f.ndim == 3 else f))) for f in ctx[::10]]
        lapvar.append(float(np.mean(lap_scores)))
    return bright, lapvar

vis_cache = {}
print('\nPre-computing visual statistics (brightness, Laplacian var)...')
for dn, dirs in _VIS_NPZ_DIRS.items():
    ck = _VIS_CACHE_MAP[dn]
    file_list = test_f if dn == 'ID' else None
    fns = file_list if file_list is not None else list(cache[ck].keys())
    vis_cache[dn] = {}
    for fn in fns:
        npz_path = None
        for d in dirs:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                npz_path = p; break
        if npz_path is None:
            continue
        br, lv = _compute_vis_file(npz_path)
        vis_cache[dn][fn] = {'brightness': br, 'laplacian': lv}
    print('  %s: %d files' % (dn, len(vis_cache[dn])))

ood_sigs = {dn: {'phi': [], 'sigma': [], 'tta_std': [], 'msp': [], 'entropy': [], 'feat_norm': [],
                 'brightness': [], 'lap_var': [], 'mc_dropout': []} for dn in ['ID','Dark','Blur','Bias','Latency']}

for dn, (dd, df, dp) in datasets.items():
    if dn not in ood_sigs: continue
    for K in HORIZONS:
        p, g, feats, phi, astd, _ = gdata(dd, dp, df, K)
        if len(p) == 0 or K not in pk_sig: continue
        nn_ = min(len(p), len(phi), len(feats), len(astd))
        feats, phi, astd = feats[:nn_], phi[:nn_], astd[:nn_]
        pn, sn = get_anom(phi, feats, K)
        ood_sigs[dn]['phi'].append(pn)
        ood_sigs[dn]['sigma'].append(sn)
        ood_sigs[dn]['tta_std'].append(astd)
        conf_ood = np.where(p > 0.5, p, 1 - p)
        ood_sigs[dn]['msp'].append(1.0 - conf_ood)
        ent_ood = -(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1)))
        ood_sigs[dn]['entropy'].append(ent_ood)
        ood_sigs[dn]['feat_norm'].append(np.linalg.norm(feats, axis=1))
        ood_sigs[dn]['mc_dropout'].append(feats[:, 5])
        _v_br, _v_lp = [], []
        _v_fl = df if df is not None else list(dd.keys())
        for _vfn in _v_fl:
            if _vfn not in dd: continue
            _ve = dd[_vfn]; _vck2 = _ve.get('cnn_K', {}); _vfk2 = _ve.get('feats_K', {})
            if K not in _vck2 or K not in _vfk2: continue
            _vn = min(len(_vck2[K]), len(_vfk2[K]))
            _vvs = vis_cache.get(dn, {}).get(_vfn, {})
            _vbr2 = _vvs.get('brightness', [])
            _vlp2 = _vvs.get('laplacian', [])
            for _vi in range(_vn):
                _v_br.append(_vbr2[_vi] if _vi < len(_vbr2) else 128.0)
                _v_lp.append(_vlp2[_vi] if _vi < len(_vlp2) else 1.0)
        ood_sigs[dn]['brightness'].append(np.array(_v_br, dtype=np.float32))
        ood_sigs[dn]['lap_var'].append(np.array(_v_lp, dtype=np.float32))

for dn in ood_sigs:
    for k in ood_sigs[dn]:
        ood_sigs[dn][k] = np.concatenate(ood_sigs[dn][k]) if ood_sigs[dn][k] else np.array([])

_tta_id = ood_sigs['ID']['tta_std']
_tta_m, _tta_h = _tta_id.mean(), _tta_id.mean() + 2 * _tta_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['tta_std_n'] = nanom(ood_sigs[dn]['tta_std'], _tta_m, _tta_h)

_msp_id = ood_sigs['ID']['msp']
_msp_m, _msp_h = _msp_id.mean(), _msp_id.mean() + 2 * _msp_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['msp_n'] = nanom(ood_sigs[dn]['msp'], _msp_m, _msp_h)
_ent_id = ood_sigs['ID']['entropy']
_ent_m, _ent_h = _ent_id.mean(), _ent_id.mean() + 2 * _ent_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['entropy_n'] = nanom(ood_sigs[dn]['entropy'], _ent_m, _ent_h)
_fn_id = ood_sigs['ID']['feat_norm']
_fn_m, _fn_h = _fn_id.mean(), _fn_id.mean() + 2 * _fn_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['feat_norm_n'] = nanom(ood_sigs[dn]['feat_norm'], _fn_m, _fn_h)
_mcd_id = ood_sigs['ID']['mc_dropout']
_mcd_m, _mcd_h = _mcd_id.mean(), _mcd_id.mean() + 2 * _mcd_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['mc_dropout_n'] = nanom(ood_sigs[dn]['mc_dropout'], _mcd_m, _mcd_h)

_br_id = ood_sigs['ID']['brightness']
_br_mu = _br_id.mean()
for dn in ood_sigs:
    ood_sigs[dn]['bright_dev'] = np.abs(ood_sigs[dn]['brightness'] - _br_mu)
_bd_id = ood_sigs['ID']['bright_dev']
_bd_m, _bd_h = _bd_id.mean(), _bd_id.mean() + 2 * _bd_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['bright_dev_n'] = nanom(ood_sigs[dn]['bright_dev'], _bd_m, _bd_h)

_lp_id = ood_sigs['ID']['lap_var']
for dn in ood_sigs:
    ood_sigs[dn]['lap_inv'] = -ood_sigs[dn]['lap_var']
_li_id = ood_sigs['ID']['lap_inv']
_li_m, _li_h = _li_id.mean(), _li_id.mean() + 2 * _li_id.std()
for dn in ood_sigs:
    ood_sigs[dn]['lap_inv_n'] = nanom(ood_sigs[dn]['lap_inv'], _li_m, _li_h)

print('\n' + '=' * 70)
print('OOD DETECTION (ID vs each OOD type)')
print('=' * 70)

_ood_list = ['Dark', 'Blur', 'Bias', 'Latency']

def _build_scores(sig_keys, dn):
    parts = [ood_sigs[dn][k] for k in sig_keys]
    return sum(parts) / len(parts)

signal_defs = [
    ('MSP',                 ['msp_n']),
    ('Entropy',             ['entropy_n']),
    ('Feature Norm',        ['feat_norm_n']),
    ('MC Dropout',          ['mc_dropout_n']),
    ('TTA std',             ['tta_std_n']),
    ('Brightness',          ['bright_dev_n']),
    ('Laplacian Var',       ['lap_inv_n']),
    ('φ (perception)',   ['phi']),
    ('σ (dynamics)',     ['sigma']),
    ('φ+σ',           ['phi', 'sigma']),
]

ood_results = {}
for sig_name, sig_keys in signal_defs:
    auroc_v, fpr95_v = [], []
    for od in _ood_list:
        if len(ood_sigs['ID']['phi']) == 0 or len(ood_sigs[od]['phi']) == 0:
            auroc_v.append(float('nan')); fpr95_v.append(float('nan')); continue
        s_id = _build_scores(sig_keys, 'ID')
        s_ood = _build_scores(sig_keys, od)
        labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
        scores = np.concatenate([s_id, s_ood])
        auroc_v.append(_auroc(labels, scores) if len(np.unique(labels)) >= 2 else float('nan'))
        fpr95_v.append(fpr_at_tpr(labels, scores, 0.95) if len(np.unique(labels)) >= 2 else float('nan'))
    ood_results[sig_name] = {'auroc': auroc_v, 'fpr95': fpr95_v}

hdr_o = '%-22s  %-10s %-10s %-10s %-10s  %-10s' % ('Signal', 'Dark', 'Blur', 'Bias', 'Latency', 'Mean')
print('\n  AUROC ↑ (higher = better OOD detection)')
print('  ' + hdr_o)
for sn, _ in signal_defs:
    v = ood_results[sn]['auroc']
    mv = np.nanmean(v)
    print('  %-22s  %-10s %-10s %-10s %-10s  %-10s' % (sn, *['%.4f'%x for x in v], '%.4f'%mv))

print('\n  FPR@95 ↓ (lower = better)')
print('  ' + hdr_o)
for sn, _ in signal_defs:
    v = ood_results[sn]['fpr95']
    mv = np.nanmean(v)
    print('  %-22s  %-10s %-10s %-10s %-10s  %-10s' % (sn, *['%.4f'%x for x in v], '%.4f'%mv))

print('\n  Samples: ID=%d, Dark=%d, Blur=%d, Bias=%d, Latency=%d' % (
    len(ood_sigs['ID']['phi']), len(ood_sigs['Dark']['phi']),
    len(ood_sigs['Blur']['phi']), len(ood_sigs['Bias']['phi']),
    len(ood_sigs['Latency']['phi'])))

_n_sig = len(signal_defs)
_sig_colors = ['#aaaaaa', '#aaaaaa', '#aaaaaa', '#aaaaaa', '#aaaaaa', '#FF8F00', '#FF8F00', '#7B1FA2', '#1976D2', '#2E7D32']
_sig_short = ['MSP', 'Entropy', 'FeatNorm', 'MCDrop', 'TTA std', 'Bright', 'LapVar', chr(966), chr(963), chr(966)+'+'+chr(963)]

fig_ood, (ax_auc, ax_fpr) = plt.subplots(1, 2, figsize=(10, 2.8))
fig_ood.subplots_adjust(wspace=0.30, bottom=0.20, left=0.07, right=0.97)
bw_o = 0.07; x_o = np.arange(len(_ood_list))
for i, (sn, _) in enumerate(signal_defs):
    off = (i - (_n_sig-1)/2) * bw_o
    ax_auc.bar(x_o + off, ood_results[sn]['auroc'], bw_o,
               label=_sig_short[i], color=_sig_colors[i], edgecolor='white', linewidth=0.3)
    ax_fpr.bar(x_o + off, ood_results[sn]['fpr95'], bw_o,
               label=_sig_short[i], color=_sig_colors[i], edgecolor='white', linewidth=0.3)

ax_auc.set_xticks(x_o); ax_auc.set_xticklabels(_ood_list, fontsize=9)
ax_auc.set_ylabel('AUROC ↑', fontsize=10)
ax_auc.set_title('(a) OOD Detection AUROC', fontweight='bold', fontsize=10)
ax_auc.set_ylim(0, 1.05)
ax_auc.legend(fontsize=6.5, loc='upper right', frameon=False, ncol=2)
ax_auc.axhline(0.5, color='#ccc', ls='--', lw=0.6, zorder=0)

ax_fpr.set_xticks(x_o); ax_fpr.set_xticklabels(_ood_list, fontsize=9)
ax_fpr.set_ylabel('FPR@95 ↓', fontsize=10)
ax_fpr.set_title('(b) FPR at 95% TPR', fontweight='bold', fontsize=10)
ax_fpr.set_ylim(0, 1.15)
ax_fpr.legend(fontsize=6.5, loc='upper right', frameon=False, ncol=2)

fig_ood.savefig(os.path.join(OUT, 'ood_detection_auroc_fpr95.png'), dpi=300, bbox_inches='tight')
fig_ood.savefig(os.path.join(OUT, 'ood_detection_auroc_fpr95.pdf'), bbox_inches='tight')
plt.close(fig_ood)
print('\nSaved: ood_detection_auroc_fpr95.png')


import matplotlib
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
from matplotlib.lines import Line2D

PS = chr(966); SS = chr(963); TAU = chr(964); DARR = chr(8595)

_cb = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
colors = {'Raw': _cb[7], 'TempScale': _cb[1], 'Platt': '#009688', 'Isotonic': _cb[4],
          'HistBin': _cb[5], 'DAC': '#FF5722',
          'Shrink('+PS+'+'+SS+')': _cb[6],
          'Raw+'+PS+'+'+SS: _cb[8],
          'TTA': _cb[2], 'TTA+'+PS: _cb[9], 'TTA+'+SS: _cb[3], 'TTA+'+PS+'+'+SS: _cb[0]}
mkrs = {'Raw': 's', 'TempScale': 'D', 'Platt': 'p', 'Isotonic': '^',
        'HistBin': 'v', 'DAC': '*',
        'Shrink('+PS+'+'+SS+')': 'X',
        'Raw+'+PS+'+'+SS: 'h',
        'TTA': 'o', 'TTA+'+PS: 'H', 'TTA+'+SS: 'P', 'TTA+'+PS+'+'+SS: 'd'}
lsts = {'Raw': (0,(4,2)), 'TempScale': (0,(5,1,1,1)), 'Platt': (0,(2,1,2,1)), 'Isotonic': (0,(1,1.5)),
        'HistBin': (0,(3,1,1,1,1,1)), 'DAC': (0,(3,1)),
        'Shrink('+PS+'+'+SS+')': (0,(6,2)),
        'Raw+'+PS+'+'+SS: '-.',
        'TTA': '-', 'TTA+'+PS: '--', 'TTA+'+SS: '-', 'TTA+'+PS+'+'+SS: '-'}

disp = {'Raw': 'Uncalibrated',
        'TempScale': 'Temp. Scaling',
        'Platt': 'Platt Scaling',
        'Isotonic': 'Isotonic Reg.',
        'HistBin': 'Hist. Binning',
        'DAC': 'DAC (KNN)',
        'Shrink('+PS+'+'+SS+')': 'Shrinkage ('+PS+'+'+SS+')',
        'Raw+'+PS+'+'+SS: 'Raw+'+PS+'+'+SS+' (w/o TTA)',
        'TTA': 'TTA',
        'TTA+'+PS: 'TTA+'+PS,
        'TTA+'+SS: 'TTA+'+SS,
        'TTA+'+PS+'+'+SS: 'TTA+'+PS+'+'+SS+' (Ours)'}

bl_keys = ['Raw', 'TempScale', 'Platt', 'Isotonic', 'HistBin', 'DAC', 'TTA+φ+σ']
ab_keys = ['Raw', 'Shrink('+PS+'+'+SS+')', 'Raw+'+PS+'+'+SS, 'TTA', 'TTA+'+PS, 'TTA+'+SS, 'TTA+'+PS+'+'+SS]
nd = len(ds_order)
taus = np.linspace(0.5, 0.95, 30)

ds_short = {'ID': 'In-Distribution', 'Dark': 'Dark', 'Blur': 'Blur',
            'Bias': 'Bias', 'Latency': 'Latency'}

def _make_ece(methods, hkeys, fname):
    ood_ds = ['Dark', 'Blur', 'Bias', 'Latency']
    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.30, bottom=0.14, top=0.84, left=0.12, right=0.97)
    for ci, ds in enumerate(ood_ds):
        r, c = divmod(ci, 2)
        ax = axes[r, c]
        for m in methods:
            rv = [float(R[m][ds].get(K, float('nan'))) for K in HORIZONS]
            ax.plot(HORIZONS, rv, ls=lsts[m], color=colors[m],
                    marker=mkrs[m], markersize=4, markeredgecolor='white',
                    markeredgewidth=0.5, linewidth=1.5,
                    zorder=3 if m == hkeys[-1] else 2)
        ax.set_title(ds_short[ds], fontweight='bold', fontsize=10, pad=3)
        ax.set_xlim(3, 52); ax.set_ylim(-0.005, 0.50)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.tick_params(length=2, width=0.5)
    axes[0, 0].set_ylabel('ECE ↓', fontsize=10)
    axes[1, 0].set_ylabel('ECE ↓', fontsize=10)
    fig.text(0.545, 0.03, 'Prediction Horizon K', ha='center', fontsize=10)
    h = [Line2D([0],[0], color=colors[m], ls=lsts[m], lw=1.5, marker=mkrs[m],
         ms=4, mec='white', mew=0.5) for m in hkeys]
    fig.legend(h, [disp[m] for m in hkeys], loc='upper center',
               ncol=min(len(hkeys), 5), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.545, 1.0), columnspacing=0.8, handlelength=1.8,
               handletextpad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, fname.replace('.png','.pdf')), bbox_inches='tight')
    plt.close(fig)
    print('Saved:', fname)

def _make_sr(methods, hkeys, fname):
    ood_ds = ['Dark', 'Blur', 'Bias', 'Latency']
    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.30, bottom=0.14, top=0.84, left=0.12, right=0.97)
    for ci, ds in enumerate(ood_ds):
        r, c = divmod(ci, 2)
        ax = axes[r, c]
        for m in methods:
            d = CAL[m][ds]
            if len(d['p']) == 0: continue
            fcr_curve = [compute_fcr(d['p'], d['c'], t) for t in taus]
            ax.plot(taus, fcr_curve, ls=lsts[m], color=colors[m], linewidth=1.5,
                    zorder=3 if m == hkeys[-1] else 2)
        ax.set_title(ds_short[ds], fontweight='bold', fontsize=10, pad=3)
        ax.set_ylim(-0.005, 0.50)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.tick_params(length=2, width=0.5)
    axes[0, 0].set_ylabel('Selective Risk ↓', fontsize=10)
    axes[1, 0].set_ylabel('Selective Risk ↓', fontsize=10)
    fig.text(0.545, 0.03, 'Confidence Threshold τ', ha='center', fontsize=10)
    h = [Line2D([0],[0], color=colors[m], ls=lsts[m], lw=1.5) for m in hkeys]
    fig.legend(h, [disp[m] for m in hkeys], loc='upper center',
               ncol=min(len(hkeys), 5), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.545, 1.0), columnspacing=0.8, handlelength=1.8,
               handletextpad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, fname.replace('.png','.pdf')), bbox_inches='tight')
    plt.close(fig)
    print('Saved:', fname)

def _make_metric(methods, hkeys, fname, result_dict, ylabel):
    ood_ds_m = ['Dark', 'Blur', 'Bias', 'Latency']
    fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.30, bottom=0.14, top=0.84, left=0.12, right=0.97)
    for ci, ds in enumerate(ood_ds_m):
        r, c = divmod(ci, 2)
        ax = axes[r, c]
        for m in methods:
            rv = [float(result_dict[m][ds].get(K, float('nan'))) for K in HORIZONS]
            ax.plot(HORIZONS, rv, ls=lsts[m], color=colors[m],
                    marker=mkrs[m], markersize=4, markeredgecolor='white',
                    markeredgewidth=0.5, linewidth=1.5,
                    zorder=3 if m == hkeys[-1] else 2)
        ax.set_title(ds_short[ds], fontweight='bold', fontsize=10, pad=3)
        ax.set_xlim(3, 52); ax.set_ylim(-0.005, 0.50)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.tick_params(length=2, width=0.5)
    axes[0, 0].set_ylabel(ylabel, fontsize=10)
    axes[1, 0].set_ylabel(ylabel, fontsize=10)
    fig.text(0.545, 0.03, 'Prediction Horizon K', ha='center', fontsize=10)
    h = [Line2D([0],[0], color=colors[m], ls=lsts[m], lw=1.5, marker=mkrs[m],
         ms=4, mec='white', mew=0.5) for m in hkeys]
    fig.legend(h, [disp[m] for m in hkeys], loc='upper center',
               ncol=min(len(hkeys), 5), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.545, 1.0), columnspacing=0.8, handlelength=1.8,
               handletextpad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, fname.replace('.png','.pdf')), bbox_inches='tight')
    plt.close(fig)
    print('Saved:', fname)

_make_ece(bl_keys, bl_keys, 'paper_baseline_ece.png')
_make_sr(bl_keys, bl_keys, 'paper_baseline_sr.png')
_make_ece(ab_keys, ab_keys, 'paper_ablation_ece.png')
_make_sr(ab_keys, ab_keys, 'paper_ablation_sr.png')

_make_metric(bl_keys, bl_keys, 'paper_baseline_ace.png', R_ACE, 'ACE \u2193')
_make_metric(bl_keys, bl_keys, 'paper_baseline_sce.png', R_SCE, 'SCE \u2193')

fig_rc, axes_rc = plt.subplots(1, nd, figsize=(10, 2.5), sharey=True, sharex=True)
fig_rc.subplots_adjust(wspace=0.08, bottom=0.25, top=0.82, left=0.06, right=0.97)
for ci, ds in enumerate(ds_order):
    ax = axes_rc[ci]
    for m in mlist:
        d = CAL[m][ds]
        if len(d['p']) == 0: continue
        co, ri = compute_risk_coverage(d['p'], d['c'])
        ax.plot(co, ri, ls=lsts[m], color=colors[m], linewidth=1.5, label=disp[m])
    ax.set_title(ds_short[ds], fontweight='bold', fontsize=10, pad=3)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(length=2, width=0.5)
    if ci > 0: ax.set_yticklabels([])
axes_rc[0].set_ylabel('Selective Risk '+DARR, fontsize=10)
fig_rc.text(0.515, 0.07, 'Coverage', ha='center', fontsize=10)
axes_rc[0].legend(fontsize=6, loc='upper right', frameon=False)
fig_rc.savefig(os.path.join(OUT, 'safety_risk_coverage.png'), dpi=300, bbox_inches='tight')
plt.close(fig_rc); print('Saved: safety_risk_coverage.png')

tau_bar = 0.7
fig3, (ax_fcr, ax_auc) = plt.subplots(1, 2, figsize=(10, 2.8))
fig3.subplots_adjust(wspace=0.30, bottom=0.22, left=0.07, right=0.97)
bw = 0.08; x = np.arange(nd); nm = len(mlist)
for i, m in enumerate(mlist):
    fcr_v = [compute_fcr(CAL[m][ds]['p'], CAL[m][ds]['c'], tau_bar) if len(CAL[m][ds]['p'])>0 else float('nan') for ds in ds_order]
    auc_v = [np.nanmean([AUC[m][ds].get(K, float('nan')) for K in HORIZONS]) for ds in ds_order]
    off = (i-(nm-1)/2)*bw
    ax_fcr.bar(x+off, fcr_v, bw, label=disp[m], color=colors[m], edgecolor='white', linewidth=0.3)
    ax_auc.bar(x+off, auc_v, bw, label=disp[m], color=colors[m], edgecolor='white', linewidth=0.3)
ax_fcr.set_xticks(x); ax_fcr.set_xticklabels([ds_short[d] for d in ds_order], fontsize=8, rotation=15, ha='right')
ax_fcr.set_ylabel('Selective Risk '+DARR, fontsize=9); ax_fcr.set_title('(a) Sel. Risk at '+TAU+'=0.7', fontweight='bold', fontsize=10)
ax_fcr.legend(fontsize=5.5, ncol=2, loc='upper right', frameon=False)
ax_auc.set_xticks(x); ax_auc.set_xticklabels([ds_short[d] for d in ds_order], fontsize=8, rotation=15, ha='right')
ax_auc.set_ylabel('AUROC', fontsize=9); ax_auc.set_title('(b) Failure Prediction AUROC', fontweight='bold', fontsize=10)
ax_auc.legend(fontsize=5.5, ncol=2, loc='lower right', frameon=False)
ax_auc.set_ylim(0.3, 0.85)
fig3.savefig(os.path.join(OUT, 'selective_risk_auroc.png'), dpi=300, bbox_inches='tight')
plt.close(fig3); print('Saved: selective_risk_auroc.png')

fk = 'TTA+'+SS
fig4, axes4 = plt.subplots(1, nd, figsize=(10, 2.5), sharey=True)
fig4.subplots_adjust(wspace=0.08, bottom=0.25, top=0.82, left=0.06, right=0.97)
for ci, ds in enumerate(ds_order):
    ax = axes4[ci]
    d_raw = CAL['Raw'][ds]; d_ours = CAL[fk][ds]
    if len(d_raw['p'])==0: continue
    conf_raw = np.where(d_raw['p']>0.5, d_raw['p'], 1-d_raw['p'])
    conf_ours = np.where(d_ours['p']>0.5, d_ours['p'], 1-d_ours['p'])
    delta = conf_ours - conf_raw; correct = d_ours['c']
    dc = delta[correct==1]; dw = delta[correct==0]
    bins_d = np.linspace(-0.4, 0.4, 40)
    if len(dc)>0: ax.hist(dc, bins=bins_d, density=True, alpha=0.5, color=_cb[2], edgecolor='none', label='Correct')
    if len(dw)>0: ax.hist(dw, bins=bins_d, density=True, alpha=0.5, color=_cb[3], edgecolor='none', label='Wrong')
    ax.axvline(0, color='black', ls='-', lw=0.5, alpha=0.4)
    if len(dc)>0: ax.axvline(dc.mean(), color=_cb[2], ls='--', lw=1.2)
    if len(dw)>0: ax.axvline(dw.mean(), color=_cb[3], ls='--', lw=1.2)
    ax.set_title(ds_short[ds], fontweight='bold', fontsize=10, pad=3)
    ax.tick_params(length=2, width=0.5)
    if ci == 0: ax.legend(fontsize=7, frameon=False, loc='upper left')
    if ci > 0: ax.set_yticklabels([])
axes4[0].set_ylabel('Density', fontsize=10)
fig4.text(0.515, 0.07, 'Confidence Change (ours − raw)', ha='center', fontsize=10)
fig4.savefig(os.path.join(OUT, 'confidence_direction.png'), dpi=300, bbox_inches='tight')
plt.close(fig4); print('Saved: confidence_direction.png')

def _oracle_risk_coverage(correct, n_points=200):
    n = len(correct); n_err = int((1 - correct).sum())
    order_oracle = np.argsort(correct)
    correct_oracle = correct[order_oracle]
    cum_err = np.cumsum(1 - correct_oracle)
    coverages = np.arange(1, n + 1) / n
    risks = cum_err / np.arange(1, n + 1)
    return coverages, risks

gap_methods = ['Raw', 'TempScale', 'DAC', 'TTA+'+PS+'+'+SS]
fig_gap, axes_gap = plt.subplots(1, len(ood_ds), figsize=(10, 2.5), sharey=True)
fig_gap.subplots_adjust(wspace=0.08, bottom=0.25, top=0.82, left=0.08, right=0.97)
for ci, ds in enumerate(ood_ds):
    ax = axes_gap[ci]
    d_ref = CAL[gap_methods[0]][ds]
    if len(d_ref['p']) == 0: continue
    cov_o, risk_o = _oracle_risk_coverage(d_ref['c'])
    for m in gap_methods:
        d = CAL[m][ds]
        if len(d['p']) == 0: continue
        conf = np.where(d['p'] > 0.5, d['p'], 1 - d['p'])
        order = np.argsort(-conf)
        correct_s = d['c'][order].astype(float)
        n = len(correct_s)
        cum_err = np.cumsum(1 - correct_s)
        coverages = np.arange(1, n + 1) / n
        risks = cum_err / np.arange(1, n + 1)
        n_o = len(risk_o)
        n_m = len(risks)
        n_pts = min(n_o, n_m)
        delta = risks[:n_pts] - risk_o[:n_pts]
        ax.plot(coverages[:n_pts], delta, ls=lsts.get(m, '-'), color=colors.get(m, 'gray'),
                linewidth=1.5, label=disp.get(m, m), zorder=3 if 'Ours' in disp.get(m,'') else 2)
    ax.axhline(0, color='black', ls='-', lw=0.4, alpha=0.3)
    ax.set_title(ds_short.get(ds, ds), fontweight='bold', fontsize=10, pad=3)
    ax.tick_params(length=2, width=0.5)
    ax.set_xlim(0, 1)
    if ci > 0: ax.set_yticklabels([])
axes_gap[0].set_ylabel('Excess Selective Risk $\\Delta(c)$', fontsize=10)
fig_gap.text(0.525, 0.07, 'Coverage $c$', ha='center', fontsize=10)
axes_gap[0].legend(fontsize=6.5, loc='upper left', frameon=False)
for ext in ['.png', '.pdf']:
    fig_gap.savefig(os.path.join(OUT, 'paper_selection_gap' + ext), dpi=300, bbox_inches='tight')
plt.close(fig_gap)
print('Saved: paper_selection_gap.png')

with open(os.path.join(OUT, 'ece_id_vs_ood_results.json'), 'w') as f:
    json.dump({
        'horizons': HORIZONS, 'methods': mlist,
        'shrink_params': shrink_params.tolist(),
        'w_tta_sigma': float(w_tta_sigma),
        'w_tta_phisigma': w_tta_ps.tolist(),
        'pk_ts_tta': {str(k): v for k, v in pk_ts_tta.items()},
        'results': {m: {ds: {str(K): R[m][ds].get(K) for K in HORIZONS} for ds in ds_order} for m in mlist},
        'accuracy': {ds: {str(K): ACC[ds].get(K) for K in HORIZONS} for ds in ds_order},
    }, f, indent=2)
fig_id, ax_id = plt.subplots(1, 1, figsize=(3.5, 2.8))
fig_id.subplots_adjust(bottom=0.18, top=0.88, left=0.16, right=0.95)
for m in bl_keys:
    rv = [float(R[m]['ID'].get(K, float('nan'))) for K in HORIZONS]
    ax_id.plot(HORIZONS, rv, ls=lsts[m], color=colors[m],
               marker=mkrs[m], markersize=5, markeredgecolor='white',
               markeredgewidth=0.5, linewidth=1.5,
               zorder=3 if m == bl_keys[-1] else 2,
               label=disp[m])
ax_id.set_xlabel('Prediction Horizon K', fontsize=10)
ax_id.set_ylabel('ECE \u2193', fontsize=10)
ax_id.set_title('In-Distribution', fontweight='bold', fontsize=11)
ax_id.set_xlim(3, 52); ax_id.set_xticks([10, 20, 30, 40, 50])
ax_id.set_ylim(-0.005, 0.20)
ax_id.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2)
ax_id.tick_params(length=2, width=0.5)
for ext in ['.png', '.pdf']:
    fig_id.savefig(os.path.join(OUT, 'paper_id_ece' + ext), dpi=300, bbox_inches='tight')
plt.close(fig_id)
print('Saved: paper_id_ece.png')

_REL_METHODS = ['Raw', 'TempScale', 'DAC', 'TTA+\u03c6+\u03c3']
_REL_TITLES = ['Uncalibrated', 'Temp. Scaling', 'DAC (KNN)',
               'TTA+\u03c6+\u03c3 (Ours)']
_REL_DS_LIST = ['Dark', 'Blur', 'Bias', 'Latency']
_N_BINS_REL = 10
_COLOR_OUT = '#3B82F6'
_COLOR_GAP = '#F87171'

fig_rel, axes_rel = plt.subplots(2, 2, figsize=(5.5, 5), sharex=True, sharey=True)
fig_rel.subplots_adjust(wspace=0.15, hspace=0.35, bottom=0.10, top=0.92,
                        left=0.12, right=0.95)

for idx, (m, title) in enumerate(zip(_REL_METHODS, _REL_TITLES)):
    r, c = divmod(idx, 2)
    ax = axes_rel[r, c]
    _p_parts, _c_parts = [], []
    for _ds in _REL_DS_LIST:
        _d = CAL[m][_ds]
        if len(_d['p']) > 0:
            _p_parts.append(_d['p'])
            _c_parts.append(_d['c'])
    if not _p_parts:
        continue
    p_cal = np.concatenate(_p_parts)
    correct = np.concatenate(_c_parts)

    pred = (p_cal > 0.5).astype(int)
    g = np.where(correct == 1, pred, 1 - pred)

    bins_r = np.linspace(0, 1, _N_BINS_REL + 1)
    bin_centers = (bins_r[:-1] + bins_r[1:]) / 2
    bin_width = bins_r[1] - bins_r[0]

    bin_acc = np.zeros(_N_BINS_REL)
    bin_conf = np.zeros(_N_BINS_REL)
    bin_count = np.zeros(_N_BINS_REL)
    for b in range(_N_BINS_REL):
        lo, hi = bins_r[b], bins_r[b + 1]
        mask = (p_cal >= lo) & (p_cal <= hi) if b == _N_BINS_REL - 1 else (p_cal >= lo) & (p_cal < hi)
        bin_count[b] = mask.sum()
        if mask.sum() > 0:
            bin_acc[b] = g[mask].mean()
            bin_conf[b] = p_cal[mask].mean()

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, zorder=1)

    for b in range(_N_BINS_REL):
        if bin_count[b] == 0:
            continue
        acc_b = bin_acc[b]
        conf_b = bin_conf[b]
        gap_b = abs(conf_b - acc_b)
        ax.bar(bin_centers[b], acc_b, width=bin_width * 0.85,
               color=_COLOR_OUT, edgecolor='white', linewidth=0.3, zorder=2)
        if acc_b < conf_b:
            ax.bar(bin_centers[b], gap_b, width=bin_width * 0.85,
                   bottom=acc_b, color=_COLOR_GAP, alpha=0.5,
                   edgecolor='white', linewidth=0.3, zorder=2)
        else:
            ax.bar(bin_centers[b], gap_b, width=bin_width * 0.85,
                   bottom=conf_b, color=_COLOR_GAP, alpha=0.5,
                   edgecolor='white', linewidth=0.3, zorder=2)

    ece_val = np.mean([np.nanmean([R[m][_ds].get(K, float('nan')) for K in HORIZONS]) for _ds in _REL_DS_LIST])
    ax.text(0.05, 0.08, 'Error=%.1f' % (ece_val * 100), transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#B2EBF2',
                      edgecolor='#4DD0E1', alpha=0.9, linewidth=0.8))

    ax.set_title(title, fontweight='bold', fontsize=10, pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(length=2, width=0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=_COLOR_OUT, edgecolor='white', label='Outputs'),
                   Patch(facecolor=_COLOR_GAP, alpha=0.5, edgecolor='white', label='Gap')]
axes_rel[0, 0].legend(handles=legend_elements, fontsize=7, loc='upper left',
                       frameon=True, facecolor='white', edgecolor='#cccccc',
                       framealpha=0.9)
axes_rel[1, 0].set_xlabel('Confidence', fontsize=10)
axes_rel[1, 1].set_xlabel('Confidence', fontsize=10)
axes_rel[0, 0].set_ylabel('Accuracy', fontsize=10)
axes_rel[1, 0].set_ylabel('Accuracy', fontsize=10)

for ext in ['.png', '.pdf']:
    fig_rel.savefig(os.path.join(OUT, 'paper_reliability_diagram' + ext),
                    dpi=300, bbox_inches='tight')
plt.close(fig_rel)
print('Saved: paper_reliability_diagram.png')

_REL2_METHODS = ['Raw', 'TempScale', 'DAC', 'TTA+\u03c6+\u03c3']
_REL2_TITLES = ['Uncalibrated', 'Temp. Scaling', 'DAC (KNN)',
                'TTA+\u03c6+\u03c3 (Ours)']
_REL2_DS_LIST = ['Dark', 'Blur', 'Bias', 'Latency']
_N_BINS2 = 10
_N_BOOT2 = 200
_COLOR2 = '#5B69A7'

fig_r2, axes_r2 = plt.subplots(2, 2, figsize=(5.5, 5), sharex=True, sharey=True)
fig_r2.subplots_adjust(wspace=0.15, hspace=0.35, bottom=0.10, top=0.92,
                       left=0.12, right=0.95)

for idx2, (m2, title2) in enumerate(zip(_REL2_METHODS, _REL2_TITLES)):
    r2, c2 = divmod(idx2, 2)
    ax2 = axes_r2[r2, c2]
    _p2, _c2 = [], []
    for _ds2 in _REL2_DS_LIST:
        _d2 = CAL[m2][_ds2]
        if len(_d2['p']) > 0:
            _p2.append(_d2['p'])
            _c2.append(_d2['c'])
    if not _p2:
        continue
    p_cal2 = np.concatenate(_p2)
    correct2 = np.concatenate(_c2)

    conf2 = np.where(p_cal2 > 0.5, p_cal2, 1 - p_cal2)
    bins2 = np.linspace(0.5, 1.0, _N_BINS2 + 1)
    bctrs2 = (bins2[:-1] + bins2[1:]) / 2
    bw2 = bins2[1] - bins2[0]

    accs2 = np.zeros(_N_BINS2)
    props2 = np.zeros(_N_BINS2)
    for b2 in range(_N_BINS2):
        lo2, hi2 = bins2[b2], bins2[b2 + 1]
        mask2 = (conf2 >= lo2) & (conf2 <= hi2) if b2 == _N_BINS2 - 1 else (conf2 >= lo2) & (conf2 < hi2)
        props2[b2] = mask2.sum() / max(len(conf2), 1)
        if mask2.sum() > 0:
            accs2[b2] = correct2[mask2].mean()

    boot2 = np.zeros((_N_BOOT2, _N_BINS2))
    rng2 = np.random.RandomState(42)
    for bi2 in range(_N_BOOT2):
        ix2 = rng2.choice(len(conf2), size=len(conf2), replace=True)
        cb2, crb2 = conf2[ix2], correct2[ix2]
        for b2 in range(_N_BINS2):
            lo2, hi2 = bins2[b2], bins2[b2 + 1]
            mask2 = (cb2 >= lo2) & (cb2 <= hi2) if b2 == _N_BINS2 - 1 else (cb2 >= lo2) & (cb2 < hi2)
            if mask2.sum() > 0:
                boot2[bi2, b2] = crb2[mask2].mean()

    lo_ci2 = np.percentile(boot2, 5, axis=0)
    hi_ci2 = np.percentile(boot2, 95, axis=0)

    mx2 = props2.max() if props2.max() > 0 else 1
    bh2 = props2 / mx2 * 0.35
    ax2.bar(bctrs2, bh2, width=bw2 * 0.85, bottom=0, color='#D3D3D3',
            edgecolor='white', linewidth=0.5, zorder=1, label='Probability Proportion')

    ax2.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, alpha=0.5,
             zorder=2, label='Perfect Calibration')

    hd2 = props2 > 0
    ax2.fill_between(bctrs2[hd2], lo_ci2[hd2], hi_ci2[hd2],
                     color=_COLOR2, alpha=0.2, zorder=3)
    ax2.plot(bctrs2[hd2], accs2[hd2], '-o', color=_COLOR2,
             linewidth=1.8, markersize=5, markeredgecolor='white',
             markeredgewidth=0.5, zorder=4, label='Calibration Plot')

    ece2 = np.mean([np.nanmean([R[m2][_ds2].get(K, float('nan')) for K in HORIZONS]) for _ds2 in _REL2_DS_LIST])
    ax2.text(0.05, 0.95, 'ECE = %.3f' % ece2, transform=ax2.transAxes,
             fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#cccccc', alpha=0.9))

    ax2.set_title(title2, fontweight='bold', fontsize=10, pad=4)
    ax2.set_xlim(0.48, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.tick_params(length=2, width=0.5)

axes_r2[0, 0].legend(fontsize=6.5, loc='upper left', frameon=True,
                      facecolor='white', edgecolor='#cccccc', framealpha=0.9)
axes_r2[1, 0].set_xlabel('Confidence', fontsize=10)
axes_r2[1, 1].set_xlabel('Confidence', fontsize=10)
axes_r2[0, 0].set_ylabel('Accuracy', fontsize=10)
axes_r2[1, 0].set_ylabel('Accuracy', fontsize=10)

for ext in ['.png', '.pdf']:
    fig_r2.savefig(os.path.join(OUT, 'paper_reliability_line' + ext),
                   dpi=300, bbox_inches='tight')
plt.close(fig_r2)
print('Saved: paper_reliability_line.png')


print('\nDone!')
