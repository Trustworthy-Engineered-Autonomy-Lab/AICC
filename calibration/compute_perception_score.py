# Original: evaluation/calibration/_update_phi_cache.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pickle, numpy as np, torch

CACHE_PATH = 'evaluation/calibration/results/multi_horizon_cache.pkl'
PHI_PATH   = 'evaluation/calibration/results/phi_cache.pkl'
VAE_PATH   = 'vae_recon/checkpoints_v2_rgb/best_model.pt'
CONTEXT_LENGTH = 100; MAX_K = 50; STRIDE = 32; MAX_WINDOWS = 30

NPZ_DIRS = {
    'real_bias': ['data_renewed/processed_64x64_bias'],
    'real_dark': ['data_renewed/processed_64x64_dark'],
    'real_blur': ['data_renewed/processed_64x64_blur'],
    'real_latency': ['data_renewed/processed_64x64_latency'],
    'id': ['data_renewed/processed_64x64'],
    'ood_v2': ['data_renewed/ood_posthoc'],
}

def find_npz(fname, dirs):
    for d in dirs:
        fp = os.path.join(d, fname)
        if os.path.isfile(fp): return fp
    return None

def compute_phi(vae, npz_path, device):
    data = np.load(npz_path)
    frames = data['frame']; n = len(frames)
    total = CONTEXT_LENGTH + MAX_K
    starts = list(range(0, max(1, n - total + 1), STRIDE))[:MAX_WINDOWS]
    phis = []
    for s in starts:
        w = frames[s:s + total]
        if len(w) < total: break
        ctx = torch.from_numpy(w[:CONTEXT_LENGTH]).float().to(device) / 255.0
        with torch.no_grad():
            mses = []
            for bi in range(0, len(ctx), 20):
                b = ctx[bi:bi+20]
                mu, lv, skip = vae.encode(b)
                z = vae.reparameterize(mu, lv)
                r = vae.decode(z, skip).clamp(0, 1)
                mses.append(((r - b)**2).mean(dim=(1,2,3)))
            phis.append(float(torch.cat(mses).mean().item()))
    return phis

with open(CACHE_PATH, 'rb') as f: cache = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
from vae_recon.vae_model_64x64_v2 import load_model_v2
vae = load_model_v2(VAE_PATH, device); vae.eval()
print('VAE loaded')

phi_data = {}
for ds in ['id', 'real_dark', 'real_blur', 'real_bias', 'real_latency', 'ood_v2']:
    if ds not in cache: continue
    dirs = NPZ_DIRS.get(ds, [])
    phi_data[ds] = {}
    files = sorted(cache[ds].keys())
    print('--- %s: %d files ---' % (ds, len(files)))
    for i, fn in enumerate(files):
        path = find_npz(fn, dirs)
        if path is None:
            print('  [%d/%d] %s NOT FOUND' % (i+1, len(files), fn)); continue
        pv = compute_phi(vae, path, device)
        phi_data[ds][fn] = pv
        print('  [%d/%d] %s -> %d w, mean=%.6f' % (i+1, len(files), fn, len(pv), np.mean(pv) if pv else 0))

with open(PHI_PATH, 'wb') as f: pickle.dump(phi_data, f)
print('Saved to', PHI_PATH)
for ds in phi_data:
    d = phi_data[ds]; tw = sum(len(v) for v in d.values())
    print('  %s: %d files, %d windows' % (ds, len(d), tw))
