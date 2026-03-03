# Original: evaluation/calibration/_extract_safety_auroc.py
"""
Extract Safety Classifier AUROC: how well the CNN output probability
separates safe (g=1) from unsafe (g=0) across all conditions.
"""
import sys, os, pickle, warnings
sys.path.insert(0, '.'); os.chdir('.')
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

CACHE = 'evaluation/calibration/results/multi_horizon_cache.pkl'
PHI_C = 'evaluation/calibration/results/phi_cache.pkl'
HORIZONS = list(range(5, 51, 5))

with open(CACHE, 'rb') as f:
    cache = pickle.load(f)
with open(PHI_C, 'rb') as f:
    phi_data = pickle.load(f)

test_f = None
if 'id' in cache:
    all_files = sorted(cache['id'].keys())
    n = len(all_files)
    test_f = all_files[int(n * 0.7):]

datasets = {}
for dt in ['real_dark', 'real_blur', 'real_bias', 'real_latency']:
    if dt in cache and cache[dt]:
        datasets[dt.replace('real_', '').capitalize()] = (cache[dt], None)
datasets['ID'] = (cache['id'], test_f)

def gdata(pf, fl, K):
    ps, gs = [], []
    for fn in (fl if fl is not None else pf.keys()):
        if fn not in pf:
            continue
        e = pf[fn]
        ck = e.get('cnn_K', {})
        if K not in ck:
            continue
        ce = ck[K]
        for i in range(len(ce)):
            d = ce[i]
            ps.append(1.0 - d['pred_prob_left'])
            gs.append(d['actual_label'])
    return np.array(ps, np.float32), np.array(gs, np.float32)

print("=" * 80)
print("SAFETY CLASSIFIER AUROC (Binary: safe=1 vs unsafe=0)")
print("  Metric: roc_auc_score(ground_truth, CNN_probability)")
print("=" * 80)

ds_order = ['ID', 'Dark', 'Blur', 'Bias', 'Latency']

print(f"\n{'Dataset':<12}", end="")
for K in HORIZONS:
    print(f" {'K='+str(K):>7}", end="")
print(f" {'Mean':>7}")
print("-" * (12 + 8 * len(HORIZONS) + 8))

summary = {}
for dn in ds_order:
    if dn not in datasets:
        continue
    dd, fl = datasets[dn]
    aurocs_K = []
    print(f"{dn:<12}", end="")
    for K in HORIZONS:
        p, g = gdata(dd, fl, K)
        if len(p) < 10 or len(np.unique(g)) < 2:
            aurocs_K.append(float('nan'))
            print(f" {'N/A':>7}", end="")
            continue
        auc = roc_auc_score(g, p)
        aurocs_K.append(auc)
        print(f" {auc:>7.4f}", end="")
    mean_auc = np.nanmean(aurocs_K)
    summary[dn] = {'per_K': dict(zip(HORIZONS, aurocs_K)), 'mean': mean_auc}
    print(f" {mean_auc:>7.4f}")

print(f"\n{'Dataset':<12} {'Pooled AUROC':>12} {'AUPRC':>8} {'N_samples':>10} {'Pos_rate':>10}")
print("-" * 56)
for dn in ds_order:
    if dn not in datasets:
        continue
    dd, fl = datasets[dn]
    all_p, all_g = [], []
    for K in HORIZONS:
        p, g = gdata(dd, fl, K)
        if len(p) > 0:
            all_p.append(p)
            all_g.append(g)
    if not all_p:
        continue
    all_p = np.concatenate(all_p)
    all_g = np.concatenate(all_g)
    if len(np.unique(all_g)) < 2:
        print(f"{dn:<12} {'N/A':>12} {'N/A':>8} {len(all_g):>10} {all_g.mean():>10.3f}")
        continue
    auc = roc_auc_score(all_g, all_p)
    ap = average_precision_score(all_g, all_p)
    summary[dn]['pooled_auroc'] = auc
    summary[dn]['pooled_auprc'] = ap
    summary[dn]['n_samples'] = len(all_g)
    summary[dn]['pos_rate'] = float(all_g.mean())
    print(f"{dn:<12} {auc:>12.4f} {ap:>8.4f} {len(all_g):>10} {all_g.mean():>10.3f}")

print("\n\n" + "=" * 80)
print("FAILURE PREDICTION AUROC (confidence ranking: correct vs incorrect)")
print("  Metric: roc_auc_score(correct_prediction, confidence)")
print("=" * 80)

print(f"\n{'Dataset':<12}", end="")
for K in HORIZONS:
    print(f" {'K='+str(K):>7}", end="")
print(f" {'Mean':>7}")
print("-" * (12 + 8 * len(HORIZONS) + 8))

for dn in ds_order:
    if dn not in datasets:
        continue
    dd, fl = datasets[dn]
    aurocs_K = []
    print(f"{dn:<12}", end="")
    for K in HORIZONS:
        p, g = gdata(dd, fl, K)
        if len(p) < 10 or len(np.unique(g)) < 2:
            aurocs_K.append(float('nan'))
            print(f" {'N/A':>7}", end="")
            continue
        pred = (p > 0.5).astype(int)
        correct = (pred == g).astype(int)
        conf = np.where(p > 0.5, p, 1 - p)
        if len(np.unique(correct)) < 2:
            aurocs_K.append(float('nan'))
            print(f" {'N/A':>7}", end="")
            continue
        auc = roc_auc_score(correct, conf)
        aurocs_K.append(auc)
        print(f" {auc:>7.4f}", end="")
    mean_auc = np.nanmean(aurocs_K)
    print(f" {mean_auc:>7.4f}")

print("\n\nDone.")
