# Original: evaluation/calibration/_patch_cnn_features.py
#!/usr/bin/env python3
"""
Patch CNN penultimate features (128-d) into an existing multi_horizon_cache.pkl
without re-running the full extraction pipeline.

For each cached window, re-runs:
    VAE encode -> LSTM predict (deterministic, no MC) -> VAE decode -> CNN forward_with_features
and stores the 128-d penultimate layer as cnn_features in each cnn_K entry.

Usage:
    py -3.11 evaluation/calibration/_patch_cnn_features.py
"""

import sys
import os

sys.path.insert(0, '.')

import pickle
import shutil
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from glob import glob


VAE_PATH = "vae_recon/checkpoints_v2_rgb/best_model.pt"
LSTM_PATH = "predictor/checkpoints_v2_rgb_recdrop/best_model.pt"
CNN_PATH = "lane_classifier/checkpoints/best_model_finetuned.pt"

ID_DIR = "data_renewed/processed_64x64"
OOD_DIR = "data_renewed/ood_dynamics"
OOD_VISUAL_DIR = "data_renewed/ood_posthoc"
REAL_LATENCY_DIR = "data_renewed/processed_64x64_latency"
REAL_BIAS_DIR = "data_renewed/processed_64x64_bias"
REAL_DARK_DIR = "data_renewed/processed_64x64_dark"
REAL_BLUR_DIR = "data_renewed/processed_64x64_blur"

CACHE_PATH = "evaluation/calibration/results/multi_horizon_cache.pkl"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

CONTEXT_LENGTH = 100
MAX_K = 50
HORIZONS = list(range(5, 51, 5))
STRIDE = 32
MAX_WINDOWS_PER_FILE = 30

ACT_MEAN = np.array([-0.15291041, 0.5], dtype=np.float32)
ACT_STD = np.array([0.40647972, 1e-6], dtype=np.float32)


def load_models(device):
    from vae_recon.vae_model_64x64_v2 import load_model_v2 as load_vae_v2
    from predictor.core.vae_predictor_v2 import VAEPredictorV2
    from lane_classifier.models.cnn_model import LaneCNN

    print("[Model] Loading VAE V2...")
    vae = load_vae_v2(VAE_PATH, device)
    vae.eval()

    print("[Model] Loading LSTM V2...")
    ckpt = torch.load(LSTM_PATH, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    args = ckpt.get('args', {})

    predictor = VAEPredictorV2(
        latent_dim=config.get('latent_dim', 64),
        spatial_size=config.get('spatial_size', 4),
        action_dim=config.get('action_dim', 2),
        hidden_channels=config.get('hidden_channels', 128),
        num_layers=config.get('num_layers', 2),
        dropout=args.get('dropout', 0.3),
        recurrent_dropout=config.get('recurrent_dropout', 0.0),
        residual=config.get('residual', True),
        vae_model_path=None,
        freeze_vae=True,
    ).to(device)

    state_dict = ckpt['model_state_dict']
    pred_state = {k: v for k, v in state_dict.items() if not k.startswith('vae_')}
    predictor.load_state_dict(pred_state, strict=False)
    predictor.eval()
    print(f"  LSTM loaded: hidden={config.get('hidden_channels', 128)}, "
          f"dropout={args.get('dropout', 0.3)}")

    print("[Model] Loading CNN Lane Classifier...")
    cnn_ckpt = torch.load(CNN_PATH, map_location=device, weights_only=False)
    cnn_args = cnn_ckpt.get('args', {})
    cnn = LaneCNN(dropout_rate=cnn_args.get('dropout', 0.5)).to(device)
    cnn.load_state_dict(cnn_ckpt['model_state_dict'])
    cnn.eval()
    print(f"  CNN loaded: val_acc={cnn_ckpt.get('val_acc', 'N/A'):.2f}%")

    return vae, predictor, cnn


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    if 'frame' in data:
        frames = data['frame']
    elif 'images' in data:
        frames = data['images']
    else:
        frames = data[list(data.keys())[0]]

    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = np.transpose(frames, (0, 3, 1, 2))
    frames = frames.astype(np.float32) / 255.0

    T = len(frames)

    if 'action' in data:
        actions = data['action'].astype(np.float32)
    elif 'actual_actions' in data:
        steering = data['actual_actions'].astype(np.float32)
        actions = np.stack([steering, np.full(T, 0.5, dtype=np.float32)], axis=1)
    else:
        actions = np.zeros((T, 2), dtype=np.float32)

    if actions.ndim == 1:
        actions = np.stack([actions, np.full(T, 0.5, dtype=np.float32)], axis=1)

    actions_norm = (actions - ACT_MEAN) / ACT_STD
    actions_norm = np.clip(actions_norm, -3.0, 3.0) / 3.0

    return frames, actions_norm


@torch.no_grad()
def encode_frames(vae, frames_tensor):
    latents = []
    for i in range(0, len(frames_tensor), 32):
        mu, _, _ = vae.encode(frames_tensor[i:i+32])
        latents.append(mu)
    return torch.cat(latents, dim=0)


@torch.no_grad()
def patch_file_cnn_features(vae, predictor, cnn, npz_path, file_entry, device):
    """Re-derive CNN 128-d features for every window stored in file_entry['cnn_K']."""
    cnn_K = file_entry.get('cnn_K', {})
    if not cnn_K:
        return 0

    frames, actions = load_npz(npz_path)
    T = len(frames)

    usable_horizons = [K for K in HORIZONS if CONTEXT_LENGTH + K <= T]
    if not usable_horizons:
        return 0

    run_k = max(usable_horizons)
    total_needed = CONTEXT_LENGTH + run_k

    starts = list(range(0, T - total_needed + 1, STRIDE))
    np.random.shuffle(starts)
    starts = starts[:MAX_WINDOWS_PER_FILE]

    n_windows = len(starts)
    first_K = next((K for K in HORIZONS if K in cnn_K and cnn_K[K]), None)
    if first_K is None:
        return 0
    n_entries = len(cnn_K[first_K])
    if n_entries != n_windows:
        print(f"\n    WARN: window count mismatch (cache={n_entries}, recomputed={n_windows}), "
              f"processing min({n_entries}, {n_windows})")
        n_windows = min(n_entries, n_windows)

    frames_t = torch.from_numpy(frames).to(device)
    actions_t = torch.from_numpy(actions).to(device)

    mean_t = torch.tensor(IMAGENET_MEAN, device=device)
    std_t = torch.tensor(IMAGENET_STD, device=device)

    patched = 0
    for wi, s in enumerate(starts[:n_windows]):
        window_frames = frames_t[s:s + total_needed]
        window_actions = actions_t[s:s + total_needed]

        latents = encode_frames(vae, window_frames)
        context_latents = latents[:CONTEXT_LENGTH].unsqueeze(0)
        actions_full = window_actions.unsqueeze(0)

        predictor.eval()
        z_pred_det = predictor.predict_sequence(
            z_context=context_latents,
            actions=actions_full,
            num_steps=run_k,
        )
        z_pred_det = z_pred_det.squeeze(0)

        for K in HORIZONS:
            if K > run_k or K not in cnn_K:
                continue
            entries = cnn_K[K]
            if wi >= len(entries):
                continue

            z_k = z_pred_det[K - 1].unsqueeze(0)
            pred_img = vae.decode(z_k).clamp(0, 1)
            pred_norm = (pred_img - mean_t) / std_t
            _, cnn_hidden = cnn.forward_with_features(pred_norm)
            entries[wi]['cnn_features'] = cnn_hidden[0].cpu().numpy()
            patched += 1

    return patched


def _has_cnn_features(cache):
    """Check if any cnn_K entry already has a non-zero cnn_features array."""
    for fname, fdata in cache.get('id', {}).items():
        for K, entries in fdata.get('cnn_K', {}).items():
            for e in entries:
                feat = e.get('cnn_features')
                if feat is not None and np.any(feat != 0):
                    return True
        break
    return False


def main():
    print("=" * 70)
    print("Patch CNN Penultimate Features (128-d) into Cache")
    print("=" * 70)

    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        print("Run the full extraction first:")
        print("  py -3.11 evaluation/calibration/extract_and_cache.py")
        sys.exit(1)

    print(f"\nLoading cache: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print("  Loaded.")

    if _has_cnn_features(cache):
        print("\nCache already contains non-zero cnn_features. Skipping patch.")
        print("Delete cnn_features entries or remove the cache backup to force re-patch.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vae, predictor, cnn = load_models(device)

    bak_path = CACHE_PATH + ".bak"
    print(f"\nBacking up cache to {bak_path}")
    shutil.copy2(CACHE_PATH, bak_path)

    np.random.seed(42)
    total_patched = 0

    print("\n--- Patching ID data ---")
    id_data = cache.get('id', {})
    id_files_on_disk = {os.path.basename(f): f for f in sorted(glob(os.path.join(ID_DIR, "*.npz")))}
    for i, (fname, fdata) in enumerate(sorted(id_data.items())):
        npz_path = id_files_on_disk.get(fname)
        if npz_path is None:
            print(f"  [{i+1}/{len(id_data)}] {fname}: FILE NOT FOUND, skip")
            continue
        print(f"  [{i+1}/{len(id_data)}] {fname}...", end="", flush=True)
        n = patch_file_cnn_features(vae, predictor, cnn, npz_path, fdata, device)
        total_patched += n
        print(f" {n} entries patched")

    print("\n--- Patching OOD data ---")
    ood_data = cache.get('ood', {})
    for type_name, type_files in sorted(ood_data.items()):
        type_dir = os.path.join(OOD_DIR, type_name)
        files_on_disk = {os.path.basename(f): f for f in sorted(glob(os.path.join(type_dir, "*.npz")))}
        print(f"  Type [{type_name}]: {len(type_files)} files")
        for fname, fdata in sorted(type_files.items()):
            npz_path = files_on_disk.get(fname)
            if npz_path is None:
                print(f"    {fname}: FILE NOT FOUND, skip")
                continue
            print(f"    {fname}...", end="", flush=True)
            n = patch_file_cnn_features(vae, predictor, cnn, npz_path, fdata, device)
            total_patched += n
            print(f" {n}")

    print("\n--- Patching OOD Visual data ---")
    ood_visual_data = cache.get('ood_visual', {})
    vis_files_on_disk = {os.path.basename(f): f for f in sorted(glob(os.path.join(OOD_VISUAL_DIR, "*.npz")))}
    for type_name, type_files in sorted(ood_visual_data.items()):
        print(f"  Type [{type_name}]: {len(type_files)} files")
        for fname, fdata in sorted(type_files.items()):
            npz_path = vis_files_on_disk.get(fname)
            if npz_path is None:
                print(f"    {fname}: FILE NOT FOUND, skip")
                continue
            print(f"    {fname}...", end="", flush=True)
            n = patch_file_cnn_features(vae, predictor, cnn, npz_path, fdata, device)
            total_patched += n
            print(f" {n}")

    real_sources = [
        ("real_latency", REAL_LATENCY_DIR),
        ("real_bias", REAL_BIAS_DIR),
        ("real_dark", REAL_DARK_DIR),
        ("real_blur", REAL_BLUR_DIR),
    ]
    for key, real_dir in real_sources:
        if key not in cache:
            continue
        print(f"\n--- Patching {key} ---")
        real_data = cache[key]
        files_on_disk = {os.path.basename(f): f for f in sorted(glob(os.path.join(real_dir, "*.npz")))}
        for i, (fname, fdata) in enumerate(sorted(real_data.items())):
            npz_path = files_on_disk.get(fname)
            if npz_path is None:
                print(f"  [{i+1}/{len(real_data)}] {fname}: FILE NOT FOUND, skip")
                continue
            print(f"  [{i+1}/{len(real_data)}] {fname}...", end="", flush=True)
            n = patch_file_cnn_features(vae, predictor, cnn, npz_path, fdata, device)
            total_patched += n
            print(f" {n}")

    print(f"\n{'='*70}")
    print(f"Total entries patched: {total_patched}")
    print(f"Saving patched cache to {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    with open(CACHE_PATH, "rb") as f:
        verify = pickle.load(f)
    ok = _has_cnn_features(verify)
    print(f"Verification: cnn_features present = {ok}")
    if ok:
        print(f"\nDone! Backup at {bak_path}")
        print("Next step:  py -3.11 evaluation/calibration/_id_vs_ood.py")
    else:
        print("WARNING: verification failed — cnn_features not found after patching.")
        print(f"Original cache backed up at {bak_path}")


if __name__ == "__main__":
    main()
