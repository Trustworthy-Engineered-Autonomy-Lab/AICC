# Original: evaluation/calibration/extract_and_cache.py
#!/usr/bin/env python3
"""
Multi-Signal Fusion Anomaly Detection — Horizon Comparison Evaluation

Computes 3 anomaly signals at 10 horizons (K=20,40,...,200):
  1. Dynamics Mahalanobis (18-dim features → Mahalanobis distance)
  2. MC Dropout (sigma_K_mean from LSTM MC sampling)
  3. Feature Density (GMM density on K-dependent dynamics features)

Combines with 9 methods (3 individual + 3 model-based + 3 learning-based).
Evaluates on real bias/latency data with 9 metrics.
Outputs horizon comparison plots.

Usage:
    py -3.11 evaluation/calibration/eval_fusion_calibration.py
    py -3.11 evaluation/calibration/eval_fusion_calibration.py --skip-extract
"""

import sys
import os
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


VAE_PATH = "vae_recon/checkpoints_v2_rgb/best_model.pt"
LSTM_PATH = "predictor/checkpoints_v2_rgb_recdrop/best_model.pt"
CNN_PATH = "lane_classifier/checkpoints/best_model_finetuned.pt"

ID_DIR = "data_renewed/processed_64x64"
OOD_DIR = "data_renewed/ood_dynamics"
REAL_LATENCY_DIR = "data_renewed/processed_64x64_latency"
REAL_BIAS_DIR = "data_renewed/processed_64x64_bias"
REAL_DARK_DIR = "data_renewed/processed_64x64_dark"
REAL_BLUR_DIR = "data_renewed/processed_64x64_blur"
OOD_VISUAL_DIR = "data_renewed/ood_posthoc"
OUTPUT_DIR = "evaluation/calibration/results"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

CONTEXT_LENGTH = 100
MAX_K = 50
HORIZONS = list(range(5, 51, 5))
MC_SAMPLES = 20
STRIDE = 32
MAX_WINDOWS_PER_FILE = 30

ACT_MEAN = np.array([-0.15291041, 0.5], dtype=np.float32)
ACT_STD = np.array([0.40647972, 1e-6], dtype=np.float32)

FREEZE_EPS = 0.02
ACTIVE_STEER_THR = 0.15

N_FEATURES = 24
ALL_FEATURE_NAMES = [
    "e1", "eK_mean", "eK_max", "e_hf_energy",
    "sigma_1", "sigma_K_mean", "sigma_K_max", "sigma_vol",
    "da_max", "a_p95", "freeze_max_run", "freeze_count",
    "cond_freeze_run", "cond_freeze_frac",
    "steer_reversal", "steer_jerk",
    "steer_mean", "e1_curve",
    "autocorr_lag1", "autocorr_lag3",
    "action_latent_lag", "error_slope",
    "steer_spectral_energy", "reaction_delay",
]

K_DEP_INDICES = [1, 2, 3, 5, 6, 7, 21]

ID_TRAIN_RATIO = 0.68
ID_CAL_RATIO = 0.16

OOD_TRAIN_RATIO = 0.7

METHOD_NAMES = [
    "OE", "MC", "FD",
    "OE+MC (avg)", "OE+FD (avg)", "OE+MC+FD (avg)",
    "OE+MC (LR)", "OE+FD (LR)", "OE+MC+FD (LR)",
]

FUSION_CONFIGS = [
    ([0],       "individual"),
    ([1],       "individual"),
    ([2],       "individual"),
    ([0, 1],    "model"),
    ([0, 2],    "model"),
    ([0, 1, 2], "model"),
    ([0, 1],    "learning"),
    ([0, 2],    "learning"),
    ([0, 1, 2], "learning"),
]

GMM_N_COMPONENTS = 5


def _apply_test_aug(img_chw):
    """Apply random augmentation to a (3,64,64) float tensor in [0,1].
    Uses brightness, contrast, saturation, noise — no horizontal flip
    (would swap left/right label)."""
    import torchvision.transforms.functional as TF
    import random
    img = img_chw.clone()
    img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.25, 0.25))
    img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.2, 0.2))
    img = TF.adjust_saturation(img, 1.0 + random.uniform(-0.15, 0.15))
    if random.random() < 0.5:
        noise = torch.randn_like(img) * random.uniform(0.01, 0.05)
        img = (img + noise).clamp(0, 1)
    return img


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
    print(f"  LSTM loaded: hidden={config.get('hidden_channels', 128)}, "
          f"dropout={args.get('dropout', 0.3)}")

    cnn = None
    cnn_temperature = 1.0
    if os.path.exists(CNN_PATH):
        print("[Model] Loading CNN Lane Classifier...")
        cnn_ckpt = torch.load(CNN_PATH, map_location=device, weights_only=False)
        cnn_args = cnn_ckpt.get('args', {})
        cnn = LaneCNN(dropout_rate=cnn_args.get('dropout', 0.5)).to(device)
        cnn.load_state_dict(cnn_ckpt['model_state_dict'])
        cnn.eval()
        print(f"  CNN loaded: val_acc={cnn_ckpt.get('val_acc', 'N/A'):.2f}%")

        cnn_temperature = 1.0
        print(f"  CNN Temperature: T={cnn_temperature:.4f} (raw softmax, calibration applied downstream)")
    else:
        print(f"[Model] CNN not found at {CNN_PATH} — calibration metrics will be skipped")

    return vae, predictor, cnn, cnn_temperature


def _learn_cnn_temperature(cnn, device):
    """Learn optimal temperature T for CNN calibration on ID val data."""
    import cv2
    from scipy.optimize import minimize_scalar
    id_files = sorted(glob(os.path.join(ID_DIR, "*.npz")))
    n_val = max(1, int(len(id_files) * 0.2))
    val_files = id_files[-n_val:]

    all_logits = []
    all_labels = []
    for f in val_files:
        data = np.load(f)
        frames = data['frame']
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2))
        if frames.dtype == np.uint8:
            frames_f = frames.astype(np.float32) / 255.0
        else:
            frames_f = frames.astype(np.float32)
        indices = np.random.choice(len(frames_f), min(200, len(frames_f)), replace=False)
        for idx in indices:
            frame = frames_f[idx]
            img_hwc = np.transpose(frame, (1, 2, 0))
            img_uint8 = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            mask = (cv2.inRange(hsv, np.array([0,50,50]), np.array([10,255,255]))
                    | cv2.inRange(hsv, np.array([170,50,50]), np.array([180,255,255])))
            _, x_coords = np.where(mask > 0)
            red_x = np.mean(x_coords) / mask.shape[1] if len(x_coords) > 0 else 0.5
            label = 0 if red_x > 0.5 else 1

            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            frame_norm = (frame_t - torch.tensor(IMAGENET_MEAN, device=device)) / torch.tensor(IMAGENET_STD, device=device)
            with torch.no_grad():
                logits = cnn(frame_norm)
            all_logits.append(logits.cpu())
            all_labels.append(label)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    nll_fn = torch.nn.CrossEntropyLoss()

    def nll_at_T(T):
        return nll_fn(all_logits / T, all_labels).item()

    result = minimize_scalar(nll_at_T, bounds=(0.1, 10.0), method='bounded')
    T_opt = result.x
    print(f"  CNN Temperature: T={T_opt:.4f} (NLL: {nll_at_T(1.0):.4f} → {nll_at_T(T_opt):.4f})")
    return T_opt


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


def longest_run_below(arr, eps):
    max_run = cur_run = 0
    for v in arr:
        if v < eps:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def conditional_freeze_features(steer, deltas, active_thr, freeze_eps):
    active_mask = np.abs(steer[:-1]) > active_thr
    active_deltas = deltas[active_mask]
    if len(active_deltas) < 3:
        return 0.0, 0.0
    max_run = cur_run = 0
    for i, is_active in enumerate(active_mask):
        if is_active and deltas[i] < freeze_eps:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    n_frozen_active = np.sum(active_deltas < freeze_eps)
    frac = float(n_frozen_active) / len(active_deltas)
    return float(max_run), frac


def steering_pattern_features(steer):
    deltas = np.diff(steer)
    if len(deltas) < 2:
        return 0.0, 0.0
    signs = np.sign(deltas)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        reversal_rate = 0.0
    else:
        reversals = np.sum(np.diff(nonzero) != 0)
        reversal_rate = float(reversals) / (len(nonzero) - 1)
    second_deriv = np.diff(deltas)
    jerk = float(np.mean(np.abs(second_deriv))) if len(second_deriv) > 0 else 0.0
    return reversal_rate, jerk


def _autocorr(x, lag):
    """Normalized autocorrelation at given lag."""
    if len(x) <= lag:
        return 0.0
    xm = x - x.mean()
    var = np.dot(xm, xm)
    if var < 1e-12:
        return 0.0
    return float(np.dot(xm[:len(xm)-lag], xm[lag:]) / var)


def temporal_features(steer, latent_norms):
    """Compute K-independent temporal features from context window.

    Args:
        steer: (CONTEXT_LENGTH,) steering values
        latent_norms: (CONTEXT_LENGTH,) L2 norm of each latent frame

    Returns:
        autocorr_lag1, autocorr_lag3, action_latent_lag,
        steer_spectral_energy, reaction_delay
    """
    autocorr_lag1 = _autocorr(steer, 1)
    autocorr_lag3 = _autocorr(steer, 3)

    steer_diff = np.diff(steer)
    lat_diff = np.diff(latent_norms)
    action_latent_lag = 0.0
    reaction_delay = 0.0
    if len(steer_diff) > 4 and len(lat_diff) > 4:
        sd_m = steer_diff - steer_diff.mean()
        ld_m = lat_diff - lat_diff.mean()
        sd_norm = np.sqrt(np.dot(sd_m, sd_m))
        ld_norm = np.sqrt(np.dot(ld_m, ld_m))
        if sd_norm > 1e-12 and ld_norm > 1e-12:
            max_lag = min(10, len(sd_m) - 1)
            xcorr = np.array([np.dot(sd_m[:len(sd_m)-l], ld_m[l:]) / (sd_norm * ld_norm)
                              if l > 0 else np.dot(sd_m, ld_m) / (sd_norm * ld_norm)
                              for l in range(max_lag + 1)])
            peak_lag = int(np.argmax(np.abs(xcorr)))
            action_latent_lag = float(xcorr[peak_lag])
            reaction_delay = float(peak_lag)

    steer_spectral_energy = 0.0
    if len(steer) >= 8:
        fft_vals = np.abs(np.fft.rfft(steer - steer.mean()))
        n_freq = len(fft_vals)
        mid = n_freq // 2
        total_energy = np.sum(fft_vals ** 2)
        if total_energy > 1e-12:
            high_energy = np.sum(fft_vals[mid:] ** 2)
            steer_spectral_energy = float(high_energy / total_energy)

    return autocorr_lag1, autocorr_lag3, action_latent_lag, steer_spectral_energy, reaction_delay


def error_slope_feature(step_errors, K):
    """Linear regression slope of prediction errors over K steps."""
    if K < 2:
        return 0.0
    errs = step_errors[:K]
    x = np.arange(K, dtype=np.float32)
    x_m = x - x.mean()
    denom = np.dot(x_m, x_m)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_m, errs - errs.mean()) / denom)


def extract_multi_horizon_window(vae, predictor, frames_np, actions_np, start, device,
                                  run_k=None, cnn=None, cnn_temperature=1.0):
    """
    Extract features at multiple horizons for one window.
    LSTM runs once for run_k steps, then slices at each horizon <= run_k.

    Args:
        run_k: how many steps to run LSTM. Defaults to MAX_K.
              Horizons > run_k will be absent from the output.
        cnn: CNN lane classifier (optional, for calibration metrics)
        cnn_temperature: temperature scaling factor for CNN

    Returns:
        features_per_K: dict {K: np.array(24,)} for each K in HORIZONS where K <= run_k
        hidden_repr: np.array (2048,) LSTM hidden state for dynamics density
        cnn_data_per_K: dict {K: {actual_conf, pred_conf, actual_label, ...}} or None
    """
    if run_k is None:
        run_k = MAX_K
    total_needed = CONTEXT_LENGTH + run_k
    window_frames = frames_np[start:start + total_needed]
    window_actions = actions_np[start:start + total_needed]

    frames_t = torch.from_numpy(window_frames).to(device)
    actions_t = torch.from_numpy(window_actions).to(device)

    with torch.no_grad():
        latents = encode_frames(vae, frames_t)

    context_latents = latents[:CONTEXT_LENGTH].unsqueeze(0)
    target_latents = latents[CONTEXT_LENGTH:]
    actions_full = actions_t.unsqueeze(0)

    with torch.no_grad():
        predictor.eval()
        z_pred_det, primed_hidden = predictor.predict_sequence(
            z_context=context_latents,
            actions=actions_full,
            num_steps=run_k,
            return_primed_hidden=True,
        )
    z_pred_det = z_pred_det.squeeze(0)

    (_, _), (h2, _) = primed_hidden
    hidden_repr = h2.squeeze(0).cpu().numpy().flatten()

    n_target = min(run_k, len(target_latents))
    all_step_errors = np.zeros(run_k, dtype=np.float32)
    for k in range(n_target):
        all_step_errors[k] = F.mse_loss(z_pred_det[k], target_latents[k]).item()
    if n_target < run_k:
        all_step_errors[n_target:] = all_step_errors[n_target - 1]

    mc_preds = []
    for _ in range(MC_SAMPLES):
        predictor.train()
        with torch.no_grad():
            z_mc = predictor.predict_sequence(
                z_context=context_latents,
                actions=actions_full,
                num_steps=run_k,
            )
        mc_preds.append(z_mc.squeeze(0))
    predictor.eval()

    mc_stack = torch.stack(mc_preds, dim=0)
    mc_std = mc_stack.std(dim=0)
    all_per_step_sigma = mc_std.mean(dim=(1, 2, 3)).detach().cpu().numpy()

    steer = window_actions[:CONTEXT_LENGTH, 0]
    deltas = np.abs(np.diff(steer))

    e1 = float(all_step_errors[0])
    sigma_1 = float(all_per_step_sigma[0])
    da_max = float(np.max(deltas)) if len(deltas) > 0 else 0.0
    a_p95 = float(np.percentile(np.abs(steer), 95))
    freeze_max_run = float(longest_run_below(deltas, FREEZE_EPS))
    freeze_count = float(np.sum(deltas < FREEZE_EPS)) / max(len(deltas), 1)
    cond_freeze_run, cond_freeze_frac = conditional_freeze_features(
        steer, deltas, ACTIVE_STEER_THR, FREEZE_EPS)
    steer_reversal, steer_jerk = steering_pattern_features(steer)
    steer_mean = float(np.mean(steer))

    active_mask = np.abs(steer[:-1]) > ACTIVE_STEER_THR
    if np.sum(active_mask) >= 3:
        active_errors = all_step_errors[0:1]
        e1_curve = float(e1)
    else:
        e1_curve = 0.0

    ctx_latents_np = latents[:CONTEXT_LENGTH].detach().cpu().numpy()
    latent_norms = np.sqrt((ctx_latents_np.reshape(len(ctx_latents_np), -1) ** 2).sum(axis=1))
    autocorr_lag1, autocorr_lag3, action_latent_lag, steer_spectral_energy, reaction_delay = \
        temporal_features(steer, latent_norms)

    features_per_K = {}
    for K in HORIZONS:
        if K > run_k:
            continue
        step_errors_K = all_step_errors[:K]
        sigma_K = all_per_step_sigma[:K]

        eK_mean = float(np.mean(step_errors_K))
        eK_max = float(np.max(step_errors_K))
        e_hf_energy = float(np.sum(np.diff(step_errors_K) ** 2)) if K > 1 else 0.0

        sigma_K_mean = float(np.mean(sigma_K))
        sigma_K_max = float(np.max(sigma_K))
        sigma_vol = float(np.std(sigma_K))

        err_slope = error_slope_feature(all_step_errors, K)

        feat = np.array([
            e1, eK_mean, eK_max, e_hf_energy,
            sigma_1, sigma_K_mean, sigma_K_max, sigma_vol,
            da_max, a_p95, freeze_max_run, freeze_count,
            cond_freeze_run, cond_freeze_frac,
            steer_reversal, steer_jerk,
            steer_mean, e1_curve,
            autocorr_lag1, autocorr_lag3,
            action_latent_lag, err_slope,
            steer_spectral_energy, reaction_delay,
        ], dtype=np.float32)
        features_per_K[K] = feat

    cnn_data_per_K = None
    if cnn is not None:
        import cv2
        cnn_data_per_K = {}
        mean_t = torch.tensor(IMAGENET_MEAN, device=device)
        std_t = torch.tensor(IMAGENET_STD, device=device)
        for K in HORIZONS:
            if K > run_k:
                continue
            actual_idx = CONTEXT_LENGTH + K - 1
            actual_frame = window_frames[actual_idx]

            img_hwc = np.transpose(actual_frame, (1, 2, 0))
            img_uint8 = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            mask = (cv2.inRange(hsv, np.array([0,50,50]), np.array([10,255,255]))
                    | cv2.inRange(hsv, np.array([170,50,50]), np.array([180,255,255])))
            _, x_coords = np.where(mask > 0)
            red_x = float(np.mean(x_coords) / mask.shape[1]) if len(x_coords) > 0 else 0.5
            actual_label = 0 if red_x > 0.5 else 1

            frame_t = torch.from_numpy(actual_frame).unsqueeze(0).to(device)
            frame_norm = (frame_t - mean_t) / std_t
            with torch.no_grad():
                logits_actual = cnn(frame_norm)
                probs_actual = F.softmax(logits_actual / cnn_temperature, dim=1)
            actual_conf = float(probs_actual.max().item())
            actual_prob_left = float(probs_actual[0, 0].item())
            actual_pred_class = int(probs_actual.argmax(dim=1).item())

            z_k = z_pred_det[K - 1].unsqueeze(0)
            with torch.no_grad():
                pred_img = vae.decode(z_k).clamp(0, 1)
            pred_norm = (pred_img - mean_t) / std_t
            with torch.no_grad():
                logits_pred, cnn_hidden = cnn.forward_with_features(pred_norm)
                probs_pred = F.softmax(logits_pred / cnn_temperature, dim=1)
            pred_conf = float(probs_pred.max().item())
            pred_prob_left = float(probs_pred[0, 0].item())
            cnn_features_np = cnn_hidden[0].cpu().numpy()

            N_AUG = 8
            aug_probs = [pred_prob_left]
            pred_img_np = pred_img[0].cpu()
            for _ in range(N_AUG):
                aug_img = _apply_test_aug(pred_img_np)
                aug_t = aug_img.unsqueeze(0).to(device)
                aug_norm = (aug_t - mean_t) / std_t
                with torch.no_grad():
                    aug_logits = cnn(aug_norm)
                    aug_p = F.softmax(aug_logits / cnn_temperature, dim=1)
                aug_probs.append(float(aug_p[0, 0].item()))
            aug_probs_arr = np.array(aug_probs, dtype=np.float32)
            aug_std = float(np.std(aug_probs_arr))
            aug_mean = float(np.mean(aug_probs_arr))
            aug_agreement = float(np.mean(
                (aug_probs_arr > 0.5) == (aug_probs_arr[0] > 0.5)
            ))

            cnn_data_per_K[K] = {
                'actual_label': actual_label,
                'actual_conf': actual_conf,
                'actual_pred_class': actual_pred_class,
                'actual_prob_left': actual_prob_left,
                'pred_conf': pred_conf,
                'pred_prob_left': pred_prob_left,
                'anomaly_score_cnn': abs(pred_prob_left - actual_prob_left),
                'aug_std': aug_std,
                'aug_mean': aug_mean,
                'aug_agreement': aug_agreement,
                'cnn_features': cnn_features_np,
            }

    return features_per_K, hidden_repr, cnn_data_per_K


def extract_multi_horizon_file(vae, predictor, npz_path, device, max_windows=MAX_WINDOWS_PER_FILE,
                                cnn=None, cnn_temperature=1.0):
    """Extract multi-horizon features from one NPZ file.
    Adaptively uses the largest K that fits the file length.

    Returns:
        features_per_K: dict {K: np.array(N, 18)} for each K in HORIZONS
        hidden_reprs: np.array(N, 2048) LSTM hidden states for dynamics density
        cnn_data_per_K: dict {K: list of dicts} CNN confidence data per horizon
    """
    frames, actions = load_npz(npz_path)
    T = len(frames)

    usable_horizons = [K for K in HORIZONS if CONTEXT_LENGTH + K <= T]
    if not usable_horizons:
        empty_feats = {K: np.zeros((0, N_FEATURES), dtype=np.float32) for K in HORIZONS}
        return empty_feats, np.zeros((0,), dtype=np.float32), {K: [] for K in HORIZONS}

    run_k = max(usable_horizons)
    total_needed = CONTEXT_LENGTH + run_k

    starts = list(range(0, T - total_needed + 1, STRIDE))
    np.random.shuffle(starts)
    starts = starts[:max_windows]

    all_feats_per_K = {K: [] for K in HORIZONS}
    all_hidden_reprs = []
    all_cnn_per_K = {K: [] for K in HORIZONS}

    for s in starts:
        feats_K, hidden_repr, cnn_data = extract_multi_horizon_window(
            vae, predictor, frames, actions, s, device, run_k=run_k,
            cnn=cnn, cnn_temperature=cnn_temperature)
        for K in HORIZONS:
            if K in feats_K:
                all_feats_per_K[K].append(feats_K[K])
            if cnn_data is not None and K in cnn_data:
                all_cnn_per_K[K].append(cnn_data[K])
        all_hidden_reprs.append(hidden_repr)

    result_feats = {}
    for K in HORIZONS:
        if all_feats_per_K[K]:
            result_feats[K] = np.stack(all_feats_per_K[K], axis=0)
        else:
            result_feats[K] = np.zeros((0, N_FEATURES), dtype=np.float32)

    if all_hidden_reprs:
        hidden_reprs = np.stack(all_hidden_reprs, axis=0)
    else:
        hidden_reprs = np.zeros((0, 2048), dtype=np.float32)

    return result_feats, hidden_reprs, all_cnn_per_K


def phase1_extract(device, cache_path):
    """Extract multi-horizon features from all data sources."""
    print("\n" + "=" * 70)
    print("PHASE 1: Multi-Horizon Feature Extraction")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Window: {CONTEXT_LENGTH} ctx + {MAX_K} pred = {CONTEXT_LENGTH + MAX_K} frames")
    print("=" * 70)

    vae, predictor, cnn, cnn_temperature = load_models(device)

    all_data = {}

    print("\n--- ID Data ---")
    id_files = sorted(glob(os.path.join(ID_DIR, "*.npz")))
    print(f"  Files: {len(id_files)}")
    id_per_file = {}
    for i, f in enumerate(id_files):
        fname = os.path.basename(f)
        print(f"  [{i+1}/{len(id_files)}] {fname}...", end="", flush=True)
        feats_K, lats, cnn_K = extract_multi_horizon_file(
            vae, predictor, f, device, cnn=cnn, cnn_temperature=cnn_temperature)
        n = len(feats_K[HORIZONS[0]])
        id_per_file[fname] = {"feats_K": feats_K, "latents": lats, "cnn_K": cnn_K}
        print(f" {n} windows")
    all_data["id"] = id_per_file

    print("\n--- OOD Data ---")
    SKIP_DYN_OVERLAP = {"noisy", "delayed", "burst_hold", "sparse_drop", "frozen"}
    ood_data = {}
    for type_dir in sorted(glob(os.path.join(OOD_DIR, "*"))):
        if not os.path.isdir(type_dir):
            continue
        type_name = os.path.basename(type_dir)
        if type_name in SKIP_DYN_OVERLAP:
            print(f"  Type [{type_name}]: SKIP (overlaps with test OOD)")
            continue
        ood_files = sorted(glob(os.path.join(type_dir, "*.npz")))
        if not ood_files:
            continue
        print(f"  Type [{type_name}]: {len(ood_files)} files")
        per_file = {}
        for f in ood_files:
            fname = os.path.basename(f)
            print(f"    {fname}...", end="", flush=True)
            feats_K, lats, cnn_K = extract_multi_horizon_file(
                vae, predictor, f, device, cnn=cnn, cnn_temperature=cnn_temperature)
            n = len(feats_K[HORIZONS[0]])
            per_file[fname] = {"feats_K": feats_K, "latents": lats, "cnn_K": cnn_K}
            print(f" {n} windows")
        ood_data[type_name] = per_file
    all_data["ood"] = ood_data

    print("\n--- Visual OOD Data (ood_posthoc) ---")
    global MC_SAMPLES
    _orig_mc = MC_SAMPLES
    MC_SAMPLES = 5
    ood_visual_data = {}
    vis_files = sorted(glob(os.path.join(OOD_VISUAL_DIR, "*.npz")))
    if vis_files:
        import re
        type_groups = {}
        for f in vis_files:
            fn = os.path.basename(f)
            m = re.match(r"ood_([a-z0-9_]+?)_data_", fn)
            if m:
                tname = m.group(1)
            else:
                tname = "unknown"
            type_groups.setdefault(tname, []).append(f)

        SKIP_DYNAMICS = {"bias_large", "bias_neg", "bias_pos",
                         "latency_5", "latency_10", "noise_act"}
        SKIP_OVERLAP = {"contrast_low", "contrast_high", "bright_high",
                        "motion_blur", "blur_5", "blur_9", "dark_03", "dark_05"}
        VIS_MAX_FILES = 8
        for tname, tfiles_list in sorted(type_groups.items()):
            if tname in SKIP_DYNAMICS:
                print(f"  Type [{tname}]: SKIP (dynamics, not visual)")
                continue
            if tname in SKIP_OVERLAP:
                print(f"  Type [{tname}]: SKIP (overlaps with test OOD)")
                continue
            tfiles_list = tfiles_list[:VIS_MAX_FILES]
            print(f"  Type [{tname}]: {len(tfiles_list)} files")
            per_file = {}
            for f in tfiles_list:
                fname = os.path.basename(f)
                print(f"    {fname}...", end="", flush=True)
                feats_K, lats, cnn_K = extract_multi_horizon_file(
                    vae, predictor, f, device, max_windows=5,
                    cnn=cnn, cnn_temperature=cnn_temperature)
                n = len(feats_K[HORIZONS[0]])
                per_file[fname] = {"feats_K": feats_K, "latents": lats, "cnn_K": cnn_K}
                print(f" {n} windows")
            ood_visual_data[tname] = per_file
    else:
        print(f"  No files found in {OOD_VISUAL_DIR}")
    MC_SAMPLES = _orig_mc
    all_data["ood_visual"] = ood_visual_data

    for real_name, real_dir in [("latency", REAL_LATENCY_DIR), ("bias", REAL_BIAS_DIR), ("dark", REAL_DARK_DIR), ("blur", REAL_BLUR_DIR)]:
        print(f"\n--- Real [{real_name}] ---")
        real_files = sorted(glob(os.path.join(real_dir, "*.npz")))
        if not real_files:
            print(f"  No files found in {real_dir}")
            continue
        print(f"  Files: {len(real_files)}")
        per_file = {}
        for f in real_files:
            fname = os.path.basename(f)
            print(f"  {fname}...", end="", flush=True)
            feats_K, lats, cnn_K = extract_multi_horizon_file(
                vae, predictor, f, device, cnn=cnn, cnn_temperature=cnn_temperature)
            n = len(feats_K[HORIZONS[0]])
            per_file[fname] = {"feats_K": feats_K, "latents": lats, "cnn_K": cnn_K}
            print(f" {n} windows")
        all_data[f"real_{real_name}"] = per_file

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as fout:
        pickle.dump(all_data, fout)
    print(f"\nFeatures cached to: {cache_path}")

    print("\n--- Extraction Summary ---")
    n_id = sum(len(v["feats_K"][HORIZONS[0]]) for v in all_data["id"].values())
    print(f"  ID: {len(all_data['id'])} files, {n_id} windows")
    for tname, tfiles in all_data["ood"].items():
        n = sum(len(v["feats_K"][HORIZONS[0]]) for v in tfiles.values())
        print(f"  OOD [{tname}]: {len(tfiles)} files, {n} windows")
    for key in ["real_latency", "real_bias", "real_dark", "real_blur"]:
        if key in all_data:
            n = sum(len(v["feats_K"][HORIZONS[0]]) for v in all_data[key].values())
            print(f"  {key}: {len(all_data[key])} files, {n} windows")

    return all_data


def concat_feats_at_K(per_file_dict, K):
    """Concatenate features at horizon K across all files."""
    arrays = []
    for v in per_file_dict.values():
        f = v["feats_K"].get(K)
        if f is not None and len(f) > 0:
            arrays.append(f)
    if arrays:
        return np.concatenate(arrays, axis=0)
    return np.zeros((0, N_FEATURES), dtype=np.float32)


def concat_latents(per_file_dict):
    """Concatenate latent means across all files."""
    arrays = []
    for v in per_file_dict.values():
        lat = v["latents"]
        if lat.ndim == 2 and len(lat) > 0:
            arrays.append(lat)
    if arrays:
        return np.concatenate(arrays, axis=0)
    return np.zeros((0, 1), dtype=np.float32)


def concat_feats_and_latents_at_K(per_file_dict, K):
    """Concatenate features AND latents only for files that have features at K.

    This ensures feats and latents are perfectly aligned (same files, same order).
    Critical for FD hybrid scoring which needs both.
    """
    feat_arrays = []
    lat_arrays = []
    for v in per_file_dict.values():
        fk = v["feats_K"].get(K)
        if fk is not None and len(fk) > 0:
            feat_arrays.append(fk)
            lat = v["latents"]
            if lat.ndim == 2 and len(lat) > 0:
                lat_arrays.append(lat)
            else:
                lat_arrays.append(np.zeros((len(fk), 2048), dtype=np.float32))
    if feat_arrays:
        return (np.concatenate(feat_arrays, axis=0),
                np.concatenate(lat_arrays, axis=0))
    return (np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros((0, 2048), dtype=np.float32))


def concat_cnn_at_K(per_file_dict, K):
    """Concatenate CNN data at horizon K across all files.
    Returns list of dicts, each with keys: actual_label, actual_conf, actual_pred_class, etc.
    """
    result = []
    for v in per_file_dict.values():
        cnn_K = v.get("cnn_K", {})
        if K in cnn_K:
            result.extend(cnn_K[K])
    return result


def file_level_split_mh(per_file_dict, train_ratio, cal_ratio=0.0):
    """Split at file level, return dict with feats at each K + latents."""
    filenames = sorted(per_file_dict.keys())
    np.random.shuffle(filenames)
    n = len(filenames)
    n_train = max(1, int(n * train_ratio))
    n_cal = max(0, int(n * cal_ratio))

    train_files = filenames[:n_train]
    cal_files = filenames[n_train:n_train + n_cal]
    test_files = filenames[n_train + n_cal:]

    def concat_files_K_and_lat(file_list, K):
        """Return matched (feats, latents) for files that have features at K."""
        feat_arrays, lat_arrays = [], []
        for f in file_list:
            fk = per_file_dict[f]["feats_K"].get(K)
            if fk is not None and len(fk) > 0:
                feat_arrays.append(fk)
                lat = per_file_dict[f]["latents"]
                if lat.ndim == 2 and len(lat) > 0:
                    lat_arrays.append(lat)
                else:
                    lat_arrays.append(np.zeros((len(fk), 2048), dtype=np.float32))
        feats = np.concatenate(feat_arrays, axis=0) if feat_arrays else np.zeros((0, N_FEATURES), dtype=np.float32)
        lats = np.concatenate(lat_arrays, axis=0) if lat_arrays else np.zeros((0, 2048), dtype=np.float32)
        return feats, lats

    train_pairs = {K: concat_files_K_and_lat(train_files, K) for K in HORIZONS}
    cal_pairs = {K: concat_files_K_and_lat(cal_files, K) for K in HORIZONS}
    test_pairs = {K: concat_files_K_and_lat(test_files, K) for K in HORIZONS}

    return {
        "train_K": {K: train_pairs[K][0] for K in HORIZONS},
        "cal_K": {K: cal_pairs[K][0] for K in HORIZONS},
        "test_K": {K: test_pairs[K][0] for K in HORIZONS},
        "train_lat": {K: train_pairs[K][1] for K in HORIZONS},
        "cal_lat": {K: cal_pairs[K][1] for K in HORIZONS},
        "test_lat": {K: test_pairs[K][1] for K in HORIZONS},
        "train_files": train_files,
        "cal_files": cal_files,
        "test_files": test_files,
    }


def mahalanobis_distance(X, mean, cov_inv):
    diff = X - mean
    left = diff @ cov_inv
    return np.sqrt(np.maximum(np.sum(left * diff, axis=1), 0.0))


def phase2_train(all_data, output_dir):
    """Train all models at K=20 for later multi-horizon evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 2: Train Models at K=20")
    print("=" * 70)

    K0 = HORIZONS[0]
    np.random.seed(42)

    id_split = file_level_split_mh(all_data["id"], ID_TRAIN_RATIO, ID_CAL_RATIO)
    X_id_train = id_split["train_K"][K0]
    X_id_cal = id_split["cal_K"][K0]
    X_id_test = id_split["test_K"][K0]
    lat_id_train = id_split["train_lat"][K0]
    lat_id_cal = id_split["cal_lat"][K0]
    lat_id_test = id_split["test_lat"][K0]

    print(f"  ID split: train={len(X_id_train)}, cal={len(X_id_cal)}, test={len(X_id_test)}")

    ood_feats_per_K = {K: [] for K in HORIZONS}
    ood_lats_per_K = {K: [] for K in HORIZONS}
    SKIP_DYN_OVERLAP_T = {"noisy", "delayed", "burst_hold", "sparse_drop", "frozen"}
    for type_name, type_files in all_data["ood"].items():
        if type_name in SKIP_DYN_OVERLAP_T:
            continue
        ood_split = file_level_split_mh(type_files, OOD_TRAIN_RATIO)
        for K in HORIZONS:
            ood_feats_per_K[K].append(ood_split["train_K"][K])
            ood_lat_K = ood_split["train_lat"][K]
            if ood_lat_K.ndim == 2:
                ood_lats_per_K[K].append(ood_lat_K)

    SKIP_VIS_OVERLAP = {"contrast_low", "contrast_high", "bright_high",
                        "motion_blur", "blur_5", "blur_9", "dark_03", "dark_05"}
    for type_name, type_files in all_data.get("ood_visual", {}).items():
        if not type_files or type_name in SKIP_VIS_OVERLAP:
            continue
        ood_split = file_level_split_mh(type_files, OOD_TRAIN_RATIO)
        for K in HORIZONS:
            ood_feats_per_K[K].append(ood_split["train_K"][K])
            ood_lat_K = ood_split["train_lat"][K]
            if ood_lat_K.ndim == 2:
                ood_lats_per_K[K].append(ood_lat_K)

    X_ood_per_K = {}
    lat_ood_per_K = {}
    for K in HORIZONS:
        X_ood_per_K[K] = np.concatenate(ood_feats_per_K[K], axis=0) if ood_feats_per_K[K] else np.zeros((0, N_FEATURES))
        lat_ood_per_K[K] = np.concatenate(ood_lats_per_K[K], axis=0) if ood_lats_per_K[K] else np.zeros((0, 1))

    X_ood_train = X_ood_per_K[K0]
    lat_ood_train = lat_ood_per_K[K0]
    print(f"  OOD train: {len(X_ood_train)} windows")

    models = {}

    print("\n  [A] Fitting Mahalanobis on ID-train...")
    scaler_mahal = StandardScaler()
    X_train_s = scaler_mahal.fit_transform(X_id_train)
    mu_mahal = np.mean(X_train_s, axis=0)
    cov = np.cov(X_train_s, rowvar=False) + np.eye(N_FEATURES) * 1e-6
    cov_inv_mahal = np.linalg.inv(cov)
    models["mahal"] = {"scaler": scaler_mahal, "mu": mu_mahal, "cov_inv": cov_inv_mahal}
    print(f"    Fitted on {len(X_train_s)} windows")

    print("\n  [B] Fitting PCA + GMM for LSTM Hidden State Density...")
    n_pca = min(50, lat_id_train.shape[0] - 1) if lat_id_train.ndim == 2 else 0
    if lat_id_train.ndim == 2 and lat_id_train.shape[0] > GMM_N_COMPONENTS and n_pca > 0:
        pca = PCA(n_components=n_pca, random_state=42)
        lat_train_pca = pca.fit_transform(lat_id_train)
        models["pca"] = pca
        explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {lat_id_train.shape[1]}d → {n_pca}d, explained variance: {explained:.2%}")

        gmm = GaussianMixture(n_components=GMM_N_COMPONENTS, covariance_type='full',
                              random_state=42, max_iter=200)
        gmm.fit(lat_train_pca)
        models["gmm"] = gmm
        id_log_probs = gmm.score_samples(lat_train_pca)
        print(f"    GMM: fitted on {len(lat_train_pca)} windows, log-prob range: "
              f"[{id_log_probs.min():.1f}, {id_log_probs.max():.1f}]")
    else:
        print(f"    WARNING: Not enough data ({lat_id_train.shape}), skipping PCA+GMM")
        models["pca"] = None
        models["gmm"] = None

    print("\n  [B2] Fitting K-dependent dynamics feature GMM for FD hybrid...")
    fd_train = X_id_train[:, K_DEP_INDICES]
    scaler_fd = StandardScaler()
    fd_train_s = scaler_fd.fit_transform(fd_train)
    n_gmm_fd = min(GMM_N_COMPONENTS, max(2, len(fd_train_s) - 1))
    if len(fd_train_s) > n_gmm_fd:
        gmm_fd = GaussianMixture(n_components=n_gmm_fd, covariance_type='full',
                                  random_state=42, max_iter=200)
        gmm_fd.fit(fd_train_s)
        models["scaler_fd"] = scaler_fd
        models["gmm_fd"] = gmm_fd
        fd_log_probs = gmm_fd.score_samples(fd_train_s)
        print(f"    GMM-FD(K-dep): fitted on {len(fd_train_s)} windows, {len(K_DEP_INDICES)} features")
        print(f"    Features: {[ALL_FEATURE_NAMES[i] for i in K_DEP_INDICES]}")
        print(f"    Log-prob range: [{fd_log_probs.min():.1f}, {fd_log_probs.max():.1f}]")
    else:
        print(f"    WARNING: Not enough data, skipping K-dep GMM")
        models["scaler_fd"] = None
        models["gmm_fd"] = None

    if (models.get("pca") is not None and models["gmm"] is not None
            and lat_id_cal.ndim == 2 and len(lat_id_cal) > 0
            and models.get("gmm_fd") is not None):
        base_fd_cal = -models["gmm"].score_samples(models["pca"].transform(lat_id_cal))
        kdep_fd_cal = -models["gmm_fd"].score_samples(
            models["scaler_fd"].transform(X_id_cal[:, K_DEP_INDICES]))
        models["fd_base_norm"] = (float(np.percentile(base_fd_cal, 5)),
                                   float(np.percentile(base_fd_cal, 95)))
        models["fd_kdep_norm"] = (float(np.percentile(kdep_fd_cal, 5)),
                                   float(np.percentile(kdep_fd_cal, 95)))
        print(f"    FD hybrid norm — base: [{models['fd_base_norm'][0]:.2f}, {models['fd_base_norm'][1]:.2f}], "
              f"K-dep: [{models['fd_kdep_norm'][0]:.2f}, {models['fd_kdep_norm'][1]:.2f}]")
    else:
        models["fd_base_norm"] = None
        models["fd_kdep_norm"] = None

    def compute_3_scores_local(feats_K20, latents, models_dict):
        """Compute the 3 anomaly scores. Returns (n, 3) or (0, 3) if empty."""
        n = len(feats_K20)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        scores = np.zeros((n, 3), dtype=np.float32)

        X_s = models_dict["mahal"]["scaler"].transform(feats_K20)
        scores[:, 0] = mahalanobis_distance(X_s, models_dict["mahal"]["mu"],
                                            models_dict["mahal"]["cov_inv"])

        scores[:, 1] = (feats_K20[:, 6] - feats_K20[:, 4]) + feats_K20[:, 7]

        scores[:, 2] = _compute_fd_hybrid(feats_K20, latents, n, models_dict)

        return scores

    scores_id_train = compute_3_scores_local(X_id_train, lat_id_train, models)
    scores_id_cal = compute_3_scores_local(X_id_cal, lat_id_cal, models)
    scores_ood_train = compute_3_scores_local(X_ood_train, lat_ood_train, models)

    print("\n  [D] Fitting percentile normalization on ID-cal...")
    norm_params = {}
    for j in range(3):
        p5 = np.percentile(scores_id_cal[:, j], 5) if len(scores_id_cal) > 0 else 0.0
        p95 = np.percentile(scores_id_cal[:, j], 95) if len(scores_id_cal) > 0 else 1.0
        if p95 - p5 < 1e-12:
            p95 = p5 + 1.0
        norm_params[j] = (p5, p95)
    models["norm_params"] = norm_params

    def normalize_scores(scores_raw):
        scores_norm = np.zeros_like(scores_raw)
        for j in range(3):
            p5, p95 = norm_params[j]
            scores_norm[:, j] = (scores_raw[:, j] - p5) / (p95 - p5)
        return scores_norm

    has_ood = len(scores_ood_train) > 0
    if not has_ood:
        print("\n  [E*] No synthetic OOD data — generating pseudo-OOD from ID high-distance samples...")
        id_dists = scores_id_train[:, 0]
        threshold_pseudo = np.percentile(id_dists, 90)
        pseudo_ood_mask = id_dists >= threshold_pseudo
        pseudo_id_mask = id_dists < np.percentile(id_dists, 50)
        scores_ood_for_train = scores_id_train[pseudo_ood_mask]
        scores_id_for_train = scores_id_train[pseudo_id_mask]
        print(f"    Pseudo-OOD: {len(scores_ood_for_train)} windows (top 10% Mahalanobis)")
        print(f"    Clean ID: {len(scores_id_for_train)} windows (bottom 50% Mahalanobis)")
    else:
        scores_ood_for_train = scores_ood_train
        scores_id_for_train = scores_id_train

    print("\n  [F] Fitting Platt scaling for methods 0-5...")
    n_platt_ood = min(len(scores_ood_for_train), len(scores_id_cal) * 2)
    platt_ood = scores_ood_for_train[:max(n_platt_ood, 1)]
    platt_id = scores_id_cal

    platt_models = {}
    if len(platt_id) > 0 and len(platt_ood) > 0:
        y_platt = np.concatenate([np.zeros(len(platt_id)), np.ones(len(platt_ood))])
        platt_all_raw = np.concatenate([platt_id, platt_ood], axis=0)
        platt_all_norm = normalize_scores(platt_all_raw)

        for m_idx in range(6):
            sig_indices = FUSION_CONFIGS[m_idx][0]
            ftype = FUSION_CONFIGS[m_idx][1]

            if ftype == "individual":
                s = platt_all_raw[:, sig_indices[0]].reshape(-1, 1)
            else:
                s = platt_all_norm[:, sig_indices].mean(axis=1, keepdims=True)

            platt_lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
            platt_lr.fit(s, y_platt)
            platt_models[m_idx] = platt_lr
            print(f"    Method {m_idx} ({METHOD_NAMES[m_idx]}): fitted")
    else:
        print("    WARNING: Not enough data for Platt scaling, using identity mapping")
        for m_idx in range(6):
            platt_models[m_idx] = None

    models["platt"] = platt_models

    print("\n  [G] Training LR fusion models for pairwise/triple combos...")
    X_fuse_all = np.concatenate([scores_id_for_train, scores_ood_for_train], axis=0)
    y_fuse_all = np.concatenate([np.zeros(len(scores_id_for_train)),
                                 np.ones(len(scores_ood_for_train))])

    lr_models = {}
    if len(X_fuse_all) >= 5 and len(np.unique(y_fuse_all)) >= 2:
        perm = np.random.permutation(len(X_fuse_all))
        X_fuse_all = X_fuse_all[perm]
        y_fuse_all = y_fuse_all[perm]

        for m_idx in range(6, 9):
            sig_indices = FUSION_CONFIGS[m_idx][0]
            X_sel = X_fuse_all[:, sig_indices]
            lr = LogisticRegressionCV(Cs=10, cv=min(5, max(2, int(min(np.sum(y_fuse_all==0), np.sum(y_fuse_all==1))))),
                                      max_iter=1000, solver='lbfgs', class_weight='balanced')
            lr.fit(X_sel, y_fuse_all)
            lr_models[m_idx] = lr
            print(f"    Method {m_idx} ({METHOD_NAMES[m_idx]}): "
                  f"trained on {len(X_fuse_all)} samples, {len(sig_indices)} features, C={lr.C_[0]:.4f}")
    else:
        print("    WARNING: Not enough training data, using dummy classifiers")
        for m_idx in range(6, 9):
            lr_models[m_idx] = None

    models["lr_models"] = lr_models

    print("\n  [H] Fitting per-K Platt scaling and LR fusion...")
    per_K_platt = {}
    per_K_lr = {}
    per_K_norm = {}

    for K in HORIZONS:
        X_cal_K = id_split["cal_K"][K]
        lat_cal_K = id_split["cal_lat"][K]
        X_ood_K = X_ood_per_K[K]
        lat_ood_K = lat_ood_per_K[K]
        X_train_K = id_split["train_K"][K]
        lat_train_K = id_split["train_lat"][K]

        if len(X_cal_K) == 0:
            continue

        s_cal = compute_3_scores_local(X_cal_K, lat_cal_K, models)
        s_ood = compute_3_scores_local(X_ood_K, lat_ood_K, models)
        s_train = compute_3_scores_local(X_train_K, lat_train_K, models)

        norm_K = {}
        for j in range(3):
            p5 = np.percentile(s_cal[:, j], 5) if len(s_cal) > 0 else 0.0
            p95 = np.percentile(s_cal[:, j], 95) if len(s_cal) > 0 else 1.0
            if p95 - p5 < 1e-12:
                p95 = p5 + 1.0
            norm_K[j] = (p5, p95)
        per_K_norm[K] = norm_K

        def normalize_K(scores_raw, nk=norm_K):
            sn = np.zeros_like(scores_raw)
            for j in range(3):
                p5, p95 = nk[j]
                sn[:, j] = (scores_raw[:, j] - p5) / (p95 - p5)
            return sn

        has_ood_K = len(s_ood) > 0
        if has_ood_K:
            s_ood_for_platt = s_ood
            s_id_for_platt = s_cal
        else:
            id_dists = s_train[:, 0]
            thresh = np.percentile(id_dists, 90) if len(id_dists) > 0 else 0
            s_ood_for_platt = s_train[id_dists >= thresh]
            s_id_for_platt = s_train[id_dists < np.percentile(id_dists, 50)] if len(id_dists) > 0 else s_cal

        platt_K = {}
        if len(s_id_for_platt) > 0 and len(s_ood_for_platt) > 0:
            y_p = np.concatenate([np.zeros(len(s_id_for_platt)), np.ones(len(s_ood_for_platt))])
            raw_p = np.concatenate([s_id_for_platt, s_ood_for_platt], axis=0)
            norm_p = normalize_K(raw_p)

            for m_idx in range(6):
                sig_indices = FUSION_CONFIGS[m_idx][0]
                ftype = FUSION_CONFIGS[m_idx][1]
                if ftype == "individual":
                    s_in = raw_p[:, sig_indices[0]].reshape(-1, 1)
                else:
                    s_in = norm_p[:, sig_indices].mean(axis=1, keepdims=True)
                try:
                    plr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                    plr.fit(s_in, y_p)
                    platt_K[m_idx] = plr
                except Exception:
                    platt_K[m_idx] = None
        per_K_platt[K] = platt_K

        lr_K = {}
        X_fuse_K = np.concatenate([s_train, s_ood], axis=0) if has_ood_K else s_train
        y_fuse_K = np.concatenate([np.zeros(len(s_train)), np.ones(len(s_ood))]) if has_ood_K else np.zeros(len(s_train))
        if len(X_fuse_K) >= 5 and len(np.unique(y_fuse_K)) >= 2:
            perm_K = np.random.permutation(len(X_fuse_K))
            X_fuse_K = X_fuse_K[perm_K]
            y_fuse_K = y_fuse_K[perm_K]
            n_cv = min(5, max(2, int(min(np.sum(y_fuse_K==0), np.sum(y_fuse_K==1)))))
            for m_idx in range(6, 9):
                sig_indices = FUSION_CONFIGS[m_idx][0]
                try:
                    lr_k = LogisticRegressionCV(Cs=10, cv=n_cv, max_iter=1000,
                                                solver='lbfgs', class_weight='balanced')
                    lr_k.fit(X_fuse_K[:, sig_indices], y_fuse_K)
                    lr_K[m_idx] = lr_k
                except Exception:
                    lr_K[m_idx] = None
        per_K_lr[K] = lr_K

    models["per_K_platt"] = per_K_platt
    models["per_K_lr"] = per_K_lr
    models["per_K_norm"] = per_K_norm
    print(f"    Fitted per-K calibration for {len(per_K_platt)} horizons")

    models_path = os.path.join(output_dir, "fusion_models.pkl")
    with open(models_path, "wb") as fout:
        pickle.dump(models, fout)
    print(f"\n  Models saved to: {models_path}")

    return models, id_split


def _compute_fd_hybrid(feats, latents, n, models):
    """Compute FD score as hybrid of hidden-state GMM + K-dependent feature GMM.

    Component 1 (base): PCA+GMM on LSTM hidden state → K-independent, good discrimination
    Component 2 (K-dep): GMM on 6 dynamics features → K-dependent, adds horizon sensitivity

    Both components are percentile-normalized to [0,1]-ish scale, then summed.
    If only one component is available, falls back to that single component.
    """
    has_base = (models.get("pca") is not None and models.get("gmm") is not None
                and latents is not None and latents.ndim == 2 and len(latents) == n)
    has_kdep = (models.get("gmm_fd") is not None and models.get("scaler_fd") is not None)

    if has_base and has_kdep and models.get("fd_base_norm") is not None:
        lat_pca = models["pca"].transform(latents)
        base_fd = -models["gmm"].score_samples(lat_pca)

        fd_feats_s = models["scaler_fd"].transform(feats[:, K_DEP_INDICES])
        kdep_fd = -models["gmm_fd"].score_samples(fd_feats_s)

        b_lo, b_hi = models["fd_base_norm"]
        k_lo, k_hi = models["fd_kdep_norm"]
        base_fd_n = (base_fd - b_lo) / max(b_hi - b_lo, 1e-10)
        kdep_fd_n = (kdep_fd - k_lo) / max(k_hi - k_lo, 1e-10)

        kdep_fd_n = np.tanh(kdep_fd_n)

        return base_fd_n + kdep_fd_n

    elif has_base:
        lat_pca = models["pca"].transform(latents)
        return -models["gmm"].score_samples(lat_pca)

    elif has_kdep:
        fd_feats_s = models["scaler_fd"].transform(feats[:, K_DEP_INDICES])
        return -models["gmm_fd"].score_samples(fd_feats_s)

    else:
        return np.zeros(n, dtype=np.float32)


def compute_all_9_scores(feats, latents, models, K=None):
    """Compute anomaly scores for all 9 methods using FUSION_CONFIGS.

    Methods:
      0: OE (individual)       3: OE+MC (avg)       6: OE+MC (LR)
      1: MC (individual)       4: OE+FD (avg)       7: OE+FD (LR)
      2: FD (individual)       5: OE+MC+FD (avg)    8: OE+MC+FD (LR)

    If K is provided and per-K calibration models exist, uses per-K Platt scaling
    and LR fusion for better calibration at that horizon.

    Returns: scores (N, 9), probs (N, 9)
    """
    n = len(feats)
    if n == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 9), dtype=np.float32)

    raw_3 = np.zeros((n, 3), dtype=np.float32)
    X_s = models["mahal"]["scaler"].transform(feats)
    raw_3[:, 0] = mahalanobis_distance(X_s, models["mahal"]["mu"], models["mahal"]["cov_inv"])
    raw_3[:, 1] = (feats[:, 6] - feats[:, 4]) + feats[:, 7]

    raw_3[:, 2] = _compute_fd_hybrid(feats, latents, n, models)

    use_per_K = (K is not None
                 and models.get("per_K_platt") is not None
                 and K in models.get("per_K_platt", {}))

    if use_per_K:
        norm_params = models["per_K_norm"].get(K, models["norm_params"])
        platt_dict = models["per_K_platt"][K]
        lr_dict = models["per_K_lr"].get(K, models.get("lr_models", {}))
    else:
        norm_params = models["norm_params"]
        platt_dict = models["platt"]
        lr_dict = models.get("lr_models", {})

    raw_3_norm = np.zeros_like(raw_3)
    for j in range(3):
        p5, p95 = norm_params[j]
        raw_3_norm[:, j] = (raw_3[:, j] - p5) / (p95 - p5)

    scores = np.zeros((n, 9), dtype=np.float32)
    probs = np.zeros((n, 9), dtype=np.float32)

    for m_idx in range(3):
        scores[:, m_idx] = raw_3[:, m_idx]

    for m_idx in range(3, 6):
        sig_indices = FUSION_CONFIGS[m_idx][0]
        scores[:, m_idx] = raw_3_norm[:, sig_indices].mean(axis=1)

    for m_idx in range(6):
        platt = platt_dict.get(m_idx)
        s_in = scores[:, m_idx].reshape(-1, 1)
        if platt is not None:
            probs[:, m_idx] = platt.predict_proba(s_in)[:, 1]
        else:
            probs[:, m_idx] = 1.0 / (1.0 + np.exp(-s_in.flatten() + np.median(s_in)))

    for m_idx in range(6, 9):
        sig_indices = FUSION_CONFIGS[m_idx][0]
        clf = lr_dict.get(m_idx)
        if clf is not None:
            X_sel = raw_3[:, sig_indices]
            p = clf.predict_proba(X_sel)[:, 1]
        else:
            p = np.clip(raw_3_norm[:, sig_indices].mean(axis=1), 0, 1)
        scores[:, m_idx] = p
        probs[:, m_idx] = p

    return scores, probs


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() * abs(avg_acc - avg_conf)
    return ece / len(probs) if len(probs) > 0 else 0.0


def compute_ada_ece(probs, labels, n_bins=10):
    """Adaptive ECE (equal-mass bins)."""
    sorted_idx = np.argsort(probs)
    probs_sorted = probs[sorted_idx]
    labels_sorted = labels[sorted_idx]

    bin_size = max(1, len(probs) // n_bins)
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, len(probs))
        if start >= end:
            continue
        avg_conf = probs_sorted[start:end].mean()
        avg_acc = labels_sorted[start:end].mean()
        ece += (end - start) * abs(avg_acc - avg_conf)
    return ece / len(probs) if len(probs) > 0 else 0.0


def compute_mce(probs, labels, n_bins=10):
    """Maximum Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        mce = max(mce, abs(avg_acc - avg_conf))
    return mce


def compute_nll(probs, labels, eps=1e-7):
    """Negative Log-Likelihood (binary cross-entropy)."""
    p = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def compute_brier(probs, labels):
    """Brier Score."""
    return np.mean((probs - labels) ** 2)


def evaluate_one_setting(probs_id, probs_anomaly, scores_id, scores_anomaly):
    """Evaluate discrimination metrics for one method at one horizon.

    Args:
        probs_id: (N_id,) calibrated probabilities for ID (expected near 0)
        probs_anomaly: (N_anom,) calibrated probabilities for anomaly (expected near 1)
        scores_id: (N_id,) raw anomaly scores for ID
        scores_anomaly: (N_anom,) raw anomaly scores for anomaly

    Returns: dict of metrics
    """
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_anomaly))])
    all_scores = np.concatenate([scores_id, scores_anomaly])

    try:
        auroc = roc_auc_score(y_true, all_scores)
    except:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, all_scores)
    except:
        auprc = 0.0

    try:
        fpr, tpr, _ = roc_curve(y_true, all_scores)
        idx_5 = np.searchsorted(fpr, 0.05)
        tpr_at_5 = tpr[min(idx_5, len(tpr) - 1)]
    except:
        tpr_at_5 = 0.0

    mean_gap = float(np.mean(scores_anomaly) - np.mean(scores_id)) if len(scores_anomaly) > 0 else 0.0

    return {
        "AUROC": auroc, "AUPRC": auprc, "TPR@5%": tpr_at_5, "MeanGap": mean_gap,
    }


def evaluate_cnn_calibration(cnn_data_list):
    """Compute CNN-based calibration metrics from a list of CNN data dicts.
    Each dict has: actual_label, pred_conf, pred_prob_left, ...
    Evaluates: is the CNN confidence on PREDICTED (reconstructed) frames well-calibrated?

    Returns: dict {ECE, AdaECE, MCE, NLL, Brier, Accuracy}
    """
    if not cnn_data_list:
        return {"ECE": float('nan'), "AdaECE": float('nan'), "MCE": float('nan'),
                "NLL": float('nan'), "Brier": float('nan'), "CNN_Acc": float('nan')}

    confs = np.array([d['pred_conf'] for d in cnn_data_list])
    labels = np.array([d['actual_label'] for d in cnn_data_list])
    pred_pl = np.array([d['pred_prob_left'] for d in cnn_data_list])
    preds = np.where(pred_pl > 0.5, 0, 1)

    correct = (preds == labels).astype(float)

    ece = compute_ece(confs, correct)
    ada_ece = compute_ada_ece(confs, correct)
    mce = compute_mce(confs, correct)

    p_true_class = np.where(labels == 0, pred_pl, 1.0 - pred_pl)
    nll = -np.mean(np.log(np.clip(p_true_class, 1e-12, 1.0)))

    brier = np.mean((1.0 - p_true_class) ** 2)

    accuracy = float(np.mean(correct))

    return {"ECE": ece, "AdaECE": ada_ece, "MCE": mce,
            "NLL": nll, "Brier": brier, "CNN_Acc": accuracy}


DISCRIM_METRIC_NAMES = ["AUROC", "AUPRC", "TPR@5%", "MeanGap"]
CNN_CALIB_METRIC_NAMES = ["ECE", "AdaECE", "MCE", "NLL", "Brier", "CNN_Acc"]
METRIC_NAMES = DISCRIM_METRIC_NAMES + CNN_CALIB_METRIC_NAMES


def phase3_evaluate(all_data, models, id_split, output_dir):
    """Evaluate all 9 methods at all horizons on real bias/latency.
    Discrimination (AUROC/AUPRC/TPR@5%) from OE/MC/FD.
    Calibration (ECE/NLL/Brier) from CNN confidence on predicted (reconstructed) frames.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-Horizon Evaluation")
    print("  Discrimination: OE / MC / FD (9 methods)")
    print("  Calibration: CNN on predicted (LSTM→VAE reconstructed) frames")
    print("=" * 70)

    all_results = {}
    cnn_calib_results = {}

    for anomaly_type in ["real_bias", "real_latency", "real_dark", "real_blur"]:
        if anomaly_type not in all_data:
            print(f"  Skipping {anomaly_type} (not found)")
            continue

        print(f"\n--- {anomaly_type} ---")
        results_type = {}
        cnn_results_type = {}

        for K in HORIZONS:
            X_id_test = id_split["test_K"][K]
            lat_id_test = id_split["test_lat"][K]

            X_anom, lat_anom = concat_feats_and_latents_at_K(all_data[anomaly_type], K)

            if len(X_anom) == 0 or len(X_id_test) == 0:
                print(f"  K={K}: skipped (no data)")
                continue

            scores_id, probs_id = compute_all_9_scores(X_id_test, lat_id_test, models, K=K)
            scores_anom, probs_anom = compute_all_9_scores(X_anom, lat_anom, models, K=K)

            results_K = {}
            for m in range(9):
                metrics = evaluate_one_setting(
                    probs_id[:, m], probs_anom[:, m],
                    scores_id[:, m], scores_anom[:, m],
                )
                results_K[m] = metrics

            cnn_id_data = []
            for fname in id_split.get("test_files", []):
                if fname in all_data["id"]:
                    cnn_K = all_data["id"][fname].get("cnn_K", {})
                    if K in cnn_K:
                        cnn_id_data.extend(cnn_K[K])

            cnn_anom_data = concat_cnn_at_K(all_data[anomaly_type], K)

            cnn_id_metrics = evaluate_cnn_calibration(cnn_id_data)
            cnn_anom_metrics = evaluate_cnn_calibration(cnn_anom_data)

            id_cnn_anom_scores = [d['anomaly_score_cnn'] for d in cnn_id_data] if cnn_id_data else []
            anom_cnn_anom_scores = [d['anomaly_score_cnn'] for d in cnn_anom_data] if cnn_anom_data else []
            mean_id_cnn_as = float(np.mean(id_cnn_anom_scores)) if id_cnn_anom_scores else float('nan')
            mean_anom_cnn_as = float(np.mean(anom_cnn_anom_scores)) if anom_cnn_anom_scores else float('nan')

            for m in range(9):
                for mk, mv in cnn_id_metrics.items():
                    results_K[m][f"{mk}_id"] = mv
                for mk, mv in cnn_anom_metrics.items():
                    results_K[m][f"{mk}_anom"] = mv
                results_K[m]["anomaly_score_cnn_id"] = mean_id_cnn_as
                results_K[m]["anomaly_score_cnn_anom"] = mean_anom_cnn_as

            results_type[K] = results_K
            cnn_results_type[K] = {"cnn_id": cnn_id_metrics, "cnn_anom": cnn_anom_metrics}

            aurocs = [results_K[m]["AUROC"] for m in range(9)]
            best_m = int(np.argmax(aurocs))
            ece_id = cnn_id_metrics.get("ECE", float('nan'))
            ece_anom = cnn_anom_metrics.get("ECE", float('nan'))
            print(f"  K={K:>3}: best={METHOD_NAMES[best_m]} "
                  f"(AUROC={aurocs[best_m]:.4f}), "
                  f"CNN_ECE: ID={ece_id:.4f} Anom={ece_anom:.4f}, "
                  f"nID={len(X_id_test)}, nAnom={len(X_anom)}")

        all_results[anomaly_type] = results_type
        cnn_calib_results[anomaly_type] = cnn_results_type

    return all_results, cnn_calib_results


def phase4_plots(all_results, cnn_calib_results, output_dir):
    """Generate horizon comparison plots.
    Top rows: Discrimination metrics (9 methods, OE/MC/FD)
    Bottom rows: CNN calibration metrics (ID vs Anomaly)
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Generating Plots")
    print("=" * 70)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',  # OE, MC, FD (individual)
              '#d62728', '#9467bd', '#8c564b',    # OE+MC, OE+FD, OE+MC+FD (avg)
              '#e377c2', '#7f7f7f', '#bcbd22']    # OE+MC, OE+FD, OE+MC+FD (LR)
    linestyles = ['-', '-', '-',
                  '--', '--', '--',
                  ':', ':', ':']
    markers = ['o', 's', '^',
               'D', 'P', 'X',
               'v', '<', '>']

    for anomaly_type, results_type in all_results.items():
        if not results_type:
            continue

        label = anomaly_type.replace("real_", "").capitalize()
        horizons_available = sorted(results_type.keys())

        if len(horizons_available) < 2:
            print(f"  {anomaly_type}: not enough horizons, skipping plot")
            continue

        cnn_data = cnn_calib_results.get(anomaly_type, {})

        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle(f"Horizon Comparison — {label}\n"
                     f"Discrimination: OE/MC/FD (9 methods)  |  "
                     f"Calibration: CNN Lane Classifier",
                     fontsize=14, fontweight='bold')

        for ax_idx, metric_name in enumerate(["AUROC", "AUPRC", "TPR@5%"]):
            ax = axes[0, ax_idx]
            for m in range(9):
                xs, ys = [], []
                for K in horizons_available:
                    if m in results_type[K] and metric_name in results_type[K][m]:
                        xs.append(K)
                        ys.append(results_type[K][m][metric_name])
                if xs:
                    ax.plot(xs, ys, color=colors[m], linestyle=linestyles[m],
                            marker=markers[m], markersize=4, linewidth=1.5,
                            label=METHOD_NAMES[m], alpha=0.85)
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} (Discrimination)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(horizons_available)
            if ax_idx == 0:
                ax.legend(fontsize=6, ncol=3, loc='best')

        ax = axes[1, 0]
        for m in range(9):
            xs, ys = [], []
            for K in horizons_available:
                if m in results_type[K] and "MeanGap" in results_type[K][m]:
                    xs.append(K)
                    ys.append(results_type[K][m]["MeanGap"])
            if xs:
                ax.plot(xs, ys, color=colors[m], linestyle=linestyles[m],
                        marker=markers[m], markersize=3, linewidth=1.2,
                        label=METHOD_NAMES[m], alpha=0.7)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("MeanGap")
        ax.set_title("Mean Score Gap (OE/MC/FD)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons_available)

        ax = axes[1, 1]
        id_acc = [cnn_data[K]["cnn_id"].get("CNN_Acc", float('nan'))
                  for K in horizons_available if K in cnn_data]
        anom_acc = [cnn_data[K]["cnn_anom"].get("CNN_Acc", float('nan'))
                    for K in horizons_available if K in cnn_data]
        ks = [K for K in horizons_available if K in cnn_data]
        ax.plot(ks, id_acc, 'b-o', label='ID', linewidth=2, markersize=5)
        ax.plot(ks, anom_acc, 'r-^', label=label, linewidth=2, markersize=5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("CNN Accuracy")
        ax.set_title("CNN Classification Accuracy", fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons_available)

        ax = axes[1, 2]
        id_anom_score = []
        anom_anom_score = []
        for K in horizons_available:
            if K in cnn_data:
                id_cnn = cnn_data[K].get("cnn_id_data_raw", None)
                anom_cnn = cnn_data[K].get("cnn_anom_data_raw", None)
            if 0 in results_type.get(K, {}):
                id_as = results_type[K][0].get("anomaly_score_cnn_id", float('nan'))
                anom_as = results_type[K][0].get("anomaly_score_cnn_anom", float('nan'))
                id_anom_score.append(id_as)
                anom_anom_score.append(anom_as)
        if all(np.isnan(x) for x in id_anom_score if not isinstance(x, float)):
            id_anom_score = []
            anom_anom_score = []
        if not id_anom_score:
            for K in horizons_available:
                if K not in cnn_data:
                    continue
                id_anom_score.append(float('nan'))
                anom_anom_score.append(float('nan'))
        ax.plot(horizons_available[:len(id_anom_score)], id_anom_score,
                'b-o', label='ID', linewidth=2, markersize=5)
        ax.plot(horizons_available[:len(anom_anom_score)], anom_anom_score,
                'r-^', label=label, linewidth=2, markersize=5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("|P_pred - P_actual|")
        ax.set_title("CNN Anomaly Score (pred-actual gap)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons_available)
        ax.set_ylim(bottom=0)

        for col_idx, metric_name in enumerate(["ECE", "NLL", "Brier"]):
            ax = axes[2, col_idx]
            id_vals = [cnn_data[K]["cnn_id"].get(metric_name, float('nan'))
                       for K in horizons_available if K in cnn_data]
            anom_vals = [cnn_data[K]["cnn_anom"].get(metric_name, float('nan'))
                         for K in horizons_available if K in cnn_data]
            ks = [K for K in horizons_available if K in cnn_data]
            ax.plot(ks, id_vals, 'b-o', label='ID', linewidth=2, markersize=5)
            ax.plot(ks, anom_vals, 'r-^', label=label, linewidth=2, markersize=5)
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric_name)
            ax.set_title(f"CNN {metric_name} (Calibration)", fontsize=11, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(horizons_available)
            ax.set_ylim(bottom=0)

        for col_idx, metric_name in enumerate(["AdaECE", "MCE"]):
            ax = axes[3, col_idx]
            id_vals = [cnn_data[K]["cnn_id"].get(metric_name, float('nan'))
                       for K in horizons_available if K in cnn_data]
            anom_vals = [cnn_data[K]["cnn_anom"].get(metric_name, float('nan'))
                         for K in horizons_available if K in cnn_data]
            ks = [K for K in horizons_available if K in cnn_data]
            ax.plot(ks, id_vals, 'b-o', label='ID', linewidth=2, markersize=5)
            ax.plot(ks, anom_vals, 'r-^', label=label, linewidth=2, markersize=5)
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric_name)
            ax.set_title(f"CNN {metric_name} (Calibration)", fontsize=11, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(horizons_available)
            ax.set_ylim(bottom=0)

        ax = axes[3, 2]
        ax.axis('off')
        K0 = horizons_available[0]
        summary_lines = [f"Summary at K={K0}:"]
        aurocs = [results_type[K0][m]["AUROC"] for m in range(9) if m in results_type.get(K0, {})]
        if aurocs:
            best_m = int(np.argmax(aurocs))
            summary_lines.append(f"Best AUROC: {METHOD_NAMES[best_m]} = {aurocs[best_m]:.4f}")
        if K0 in cnn_data:
            summary_lines.append(f"CNN ECE (ID): {cnn_data[K0]['cnn_id'].get('ECE', float('nan')):.4f}")
            summary_lines.append(f"CNN ECE (Anom): {cnn_data[K0]['cnn_anom'].get('ECE', float('nan')):.4f}")
            summary_lines.append(f"CNN Acc (ID): {cnn_data[K0]['cnn_id'].get('CNN_Acc', float('nan')):.4f}")
            summary_lines.append(f"CNN Acc (Anom): {cnn_data[K0]['cnn_anom'].get('CNN_Acc', float('nan')):.4f}")
        ax.text(0.1, 0.5, "\n".join(summary_lines), transform=ax.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"{anomaly_type.replace('real_', '')}_horizon_comparison.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    K0 = HORIZONS[0]
    for anomaly_type, results_type in all_results.items():
        if K0 not in results_type:
            continue

        label = anomaly_type.replace("real_", "").capitalize()
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Reliability Diagrams at K={K0} — {label}",
                     fontsize=14, fontweight='bold')

        for m in range(9):
            ax = axes[m // 3, m % 3]
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')

            ece_val = results_type[K0][m].get("ECE_id", 0) if m in results_type[K0] else 0
            auroc_val = results_type[K0][m]["AUROC"] if m in results_type[K0] else 0.5

            ax.set_title(f"{METHOD_NAMES[m]}\nCNN_ECE(ID)={ece_val:.4f}, AUROC={auroc_val:.4f}",
                         fontsize=9)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"reliability_K{K0}_{anomaly_type.replace('real_', '')}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print("  (Score distribution plots require raw scores - generated from phase3 data)")


def phase4_reliability_detailed(all_data, models, id_split, output_dir):
    """Generate detailed reliability diagrams with actual probability bins."""
    print("\n  Generating detailed reliability diagrams...")
    K0 = HORIZONS[0]

    for anomaly_type in ["real_bias", "real_latency", "real_dark", "real_blur"]:
        if anomaly_type not in all_data:
            continue

        label = anomaly_type.replace("real_", "").capitalize()

        X_id_test = id_split["test_K"][K0]
        lat_id_test = id_split["test_lat"][K0]
        X_anom, lat_anom = concat_feats_and_latents_at_K(all_data[anomaly_type], K0)

        if len(X_anom) == 0 or len(X_id_test) == 0:
            continue

        _, probs_id = compute_all_9_scores(X_id_test, lat_id_test, models, K=K0)
        _, probs_anom = compute_all_9_scores(X_anom, lat_anom, models, K=K0)

        y_true = np.concatenate([np.zeros(len(probs_id)), np.ones(len(probs_anom))])

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Reliability Diagrams at K={K0} — {label}", fontsize=14, fontweight='bold')

        n_bins = 10
        for m in range(9):
            ax = axes[m // 3, m % 3]
            all_probs = np.concatenate([probs_id[:, m], probs_anom[:, m]])

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_accs = []
            bin_confs = []
            bin_counts = []

            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                if i == n_bins - 1:
                    mask = (all_probs >= lo) & (all_probs <= hi)
                else:
                    mask = (all_probs >= lo) & (all_probs < hi)
                if mask.sum() > 0:
                    bin_accs.append(y_true[mask].mean())
                    bin_confs.append(all_probs[mask].mean())
                    bin_counts.append(mask.sum())

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            if bin_confs:
                ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color='#1f77b4',
                       edgecolor='black', linewidth=0.5)
                ax.scatter(bin_confs, bin_accs, color='red', s=20, zorder=5)

            ece = compute_ece(all_probs, y_true)
            ax.set_title(f"{METHOD_NAMES[m]}\nECE={ece:.4f}", fontsize=9)
            ax.set_xlabel("Mean Predicted Prob")
            ax.set_ylabel("Fraction Positive")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"reliability_detailed_K{K0}_{anomaly_type.replace('real_', '')}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def phase4_score_distribution(all_data, models, id_split, output_dir):
    """Generate score distribution plots at K=20."""
    print("\n  Generating score distribution plots...")
    K0 = HORIZONS[0]

    X_id_test = id_split["test_K"][K0]
    lat_id_test = id_split["test_lat"][K0]

    scores_id, _ = compute_all_9_scores(X_id_test, lat_id_test, models, K=K0)

    anomaly_scores = {}
    for anomaly_type in ["real_bias", "real_latency", "real_dark", "real_blur"]:
        if anomaly_type not in all_data:
            continue
        X_anom, lat_anom = concat_feats_and_latents_at_K(all_data[anomaly_type], K0)
        if len(X_anom) == 0:
            continue
        s_anom, _ = compute_all_9_scores(X_anom, lat_anom, models, K=K0)
        anomaly_scores[anomaly_type] = s_anom

    if not anomaly_scores:
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Score Distributions at K={K0}", fontsize=14, fontweight='bold')

    for m in range(9):
        ax = axes[m // 3, m % 3]

        ax.hist(scores_id[:, m], bins=30, alpha=0.5, color='blue', label='ID', density=True)
        for atype, s_anom in anomaly_scores.items():
            lbl = atype.replace("real_", "")
            color = 'red' if 'bias' in atype else 'orange'
            ax.hist(s_anom[:, m], bins=30, alpha=0.5, color=color, label=lbl, density=True)

        ax.set_title(METHOD_NAMES[m], fontsize=10, fontweight='bold')
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"score_dist_K{K0}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def phase5_save(all_results, cnn_calib_results, output_dir):
    """Save comparison table CSV and JSON."""
    print("\n" + "=" * 70)
    print("PHASE 5: Saving Results")
    print("=" * 70)

    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        return obj

    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w") as fout:
        json.dump(to_json({"discrimination": all_results, "cnn_calibration": cnn_calib_results}),
                  fout, indent=2)
    print(f"  JSON: {json_path}")

    csv_discrim = ["AUROC", "AUPRC", "TPR@5%", "MeanGap"]
    csv_cnn = ["ECE_id", "NLL_id", "Brier_id", "CNN_Acc_id", "ECE_anom", "NLL_anom", "Brier_anom", "CNN_Acc_anom",
               "anomaly_score_cnn_id", "anomaly_score_cnn_anom"]
    csv_all = csv_discrim + csv_cnn

    csv_path = os.path.join(output_dir, "comparison_table.csv")
    with open(csv_path, "w") as fout:
        header = "anomaly_type,horizon,method," + ",".join(csv_all)
        fout.write(header + "\n")

        for atype, results_type in all_results.items():
            for K in sorted(results_type.keys()):
                for m in range(9):
                    if m not in results_type[K]:
                        continue
                    vals = []
                    for mn in csv_all:
                        v = results_type[K][m].get(mn, float('nan'))
                        vals.append(f"{v:.6f}" if not np.isnan(v) else "nan")
                    fout.write(f"{atype},{K},{METHOD_NAMES[m]}," + ",".join(vals) + "\n")
    print(f"  CSV: {csv_path}")

    print(f"\n{'='*120}")
    print("SUMMARY TABLE (K=20)")
    print(f"{'='*120}")
    K0 = HORIZONS[0]
    for atype, results_type in all_results.items():
        if K0 not in results_type:
            continue
        label = atype.replace("real_", "").upper()
        print(f"\n  --- {label} (Discrimination: OE/MC/FD) ---")
        print(f"  {'Method':<18} | {'AUROC':>7} | {'AUPRC':>7} | {'TPR@5%':>7} | {'MeanGap':>8}")
        print(f"  {'-'*60}")
        for m in range(9):
            r = results_type[K0][m]
            print(f"  {METHOD_NAMES[m]:<18} | {r['AUROC']:>7.4f} | {r['AUPRC']:>7.4f} | "
                  f"{r['TPR@5%']:>7.4f} | {r['MeanGap']:>8.4f}")

        r0 = results_type[K0].get(0, {})
        print(f"\n  --- {label} (CNN Calibration) ---")
        print(f"  {'Source':<10} | {'ECE':>7} | {'NLL':>7} | {'Brier':>7} | {'CNN_Acc':>7}")
        print(f"  {'-'*55}")
        for source in ["id", "anom"]:
            ece = r0.get(f"ECE_{source}", float('nan'))
            nll = r0.get(f"NLL_{source}", float('nan'))
            brier = r0.get(f"Brier_{source}", float('nan'))
            acc = r0.get(f"CNN_Acc_{source}", float('nan'))
            print(f"  {source.upper():<10} | {ece:>7.4f} | {nll:>7.4f} | {brier:>7.4f} | {acc:>7.4f}")

    print(f"\n{'='*120}")
    print("BEST METHOD PER HORIZON (by AUROC) + CNN Calibration")
    print(f"{'='*120}")
    for atype, results_type in all_results.items():
        label = atype.replace("real_", "").upper()
        cnn_data = cnn_calib_results.get(atype, {})
        print(f"\n  --- {label} ---")
        print(f"  {'Horizon':>8} | {'Best Method':<18} | {'AUROC':>7} | "
              f"{'CNN_ECE(ID)':>11} | {'CNN_ECE(Anom)':>13}")
        print(f"  {'-'*70}")
        for K in sorted(results_type.keys()):
            aurocs = [results_type[K][m]["AUROC"] for m in range(9)]
            best_m = int(np.argmax(aurocs))
            ece_id = cnn_data.get(K, {}).get("cnn_id", {}).get("ECE", float('nan'))
            ece_anom = cnn_data.get(K, {}).get("cnn_anom", {}).get("ECE", float('nan'))
            print(f"  {K:>8} | {METHOD_NAMES[best_m]:<18} | {aurocs[best_m]:>7.4f} | "
                  f"{ece_id:>11.4f} | {ece_anom:>13.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Signal Fusion Calibration Evaluation")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction, load from cache")
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Signal Fusion Anomaly Detection — Horizon Comparison")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Methods: {len(METHOD_NAMES)}")
    print(f"  Metrics: {len(METRIC_NAMES)}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, "multi_horizon_cache.pkl")

    if args.skip_extract and os.path.exists(cache_path):
        print(f"\nLoading cached features from {cache_path}...")
        with open(cache_path, "rb") as fin:
            all_data = pickle.load(fin)
        print("  Loaded!")
    else:
        all_data = phase1_extract(device, cache_path)

    models, id_split = phase2_train(all_data, OUTPUT_DIR)

    all_results, cnn_calib_results = phase3_evaluate(all_data, models, id_split, OUTPUT_DIR)

    phase4_plots(all_results, cnn_calib_results, OUTPUT_DIR)
    phase4_reliability_detailed(all_data, models, id_split, OUTPUT_DIR)
    phase4_score_distribution(all_data, models, id_split, OUTPUT_DIR)

    phase5_save(all_results, cnn_calib_results, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("ALL PHASES COMPLETE!")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
