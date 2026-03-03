# Original: evaluation/calibration/plot_method_comparison.py
#!/usr/bin/env python3
"""
Plot ECE (Expected Calibration Error) for all 9 fusion methods across horizons.

Evaluates CNN confidence on PREDICTED (LSTM→VAE reconstructed) frames.
Anomaly methods adjust pred_conf to improve calibration.

Usage:
    py -3.11 evaluation/calibration/plot_method_ece.py
"""

import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.calibration.eval_fusion_calibration import (
    HORIZONS, METHOD_NAMES, N_FEATURES, OUTPUT_DIR,
    ID_TRAIN_RATIO, ID_CAL_RATIO,
    concat_feats_and_latents_at_K, concat_cnn_at_K, file_level_split_mh,
    compute_all_9_scores, evaluate_cnn_calibration,
    compute_ece, compute_ada_ece, compute_nll, compute_brier,
)


def main():
    cache_path = os.path.join(OUTPUT_DIR, "multi_horizon_cache.pkl")
    models_path = os.path.join(OUTPUT_DIR, "fusion_models.pkl")

    print("Loading cached features...")
    with open(cache_path, "rb") as f:
        all_data = pickle.load(f)
    print("Loading trained models...")
    with open(models_path, "rb") as f:
        models = pickle.load(f)

    np.random.seed(42)
    id_split = file_level_split_mh(all_data["id"], ID_TRAIN_RATIO, ID_CAL_RATIO)

    print(f"ID test windows at K={HORIZONS[0]}: {len(id_split['test_K'][HORIZONS[0]])}")


    anomaly_types = ["real_bias", "real_latency", "real_dark", "real_blur"]

    results_adj = {}
    results_raw_cnn = {}

    for atype in anomaly_types:
        if atype not in all_data:
            print(f"  {atype}: not found, skipping")
            continue
        print(f"\n--- {atype} ---")
        results_adj[atype] = {}
        results_raw_cnn[atype] = {}

        for K in HORIZONS:
            X_id_test = id_split["test_K"][K]
            lat_id_test = id_split["test_lat"][K]
            X_anom, lat_anom = concat_feats_and_latents_at_K(all_data[atype], K)

            if len(X_anom) == 0 or len(X_id_test) == 0:
                continue

            scores_id, _ = compute_all_9_scores(X_id_test, lat_id_test, models, K=K)
            scores_anom, _ = compute_all_9_scores(X_anom, lat_anom, models, K=K)

            all_scores = np.concatenate([scores_id, scores_anom], axis=0)
            probs_raw = np.zeros_like(all_scores)
            for m in range(6):
                vals = all_scores[:, m]
                p5 = np.percentile(scores_id[:, m], 5) if len(scores_id) > 0 else 0
                p95 = np.percentile(scores_id[:, m], 95) if len(scores_id) > 0 else 1
                if p95 - p5 < 1e-12:
                    p95 = p5 + 1.0
                probs_raw[:, m] = np.clip((vals - p5) / (p95 - p5), 0.0, 1.0)
            for m in range(6, 9):
                probs_raw[:, m] = np.clip(all_scores[:, m], 0.0, 1.0)

            n_id = len(scores_id)
            probs_id = probs_raw[:n_id]
            probs_anom = probs_raw[n_id:]

            cnn_id_list = []
            for fname in id_split.get("test_files", []):
                if fname in all_data["id"]:
                    cnn_K = all_data["id"][fname].get("cnn_K", {})
                    if K in cnn_K and len(cnn_K[K]) > 0:
                        cnn_id_list.extend(cnn_K[K])

            cnn_anom_list = []
            for v in all_data[atype].values():
                fk = v["feats_K"].get(K)
                if fk is not None and len(fk) > 0:
                    cnn_K = v.get("cnn_K", {})
                    if K in cnn_K and len(cnn_K[K]) > 0:
                        cnn_anom_list.extend(cnn_K[K])

            n_id_cnn = min(len(cnn_id_list), len(probs_id))
            n_anom_cnn = min(len(cnn_anom_list), len(probs_anom))

            if n_id_cnn + n_anom_cnn == 0:
                continue

            all_cnn = cnn_id_list[:n_id_cnn] + cnn_anom_list[:n_anom_cnn]
            confs_raw = np.array([d['pred_conf'] for d in all_cnn])
            labels = np.array([d['actual_label'] for d in all_cnn])
            pred_prob_left = np.array([d['pred_prob_left'] for d in all_cnn])
            preds = np.where(pred_prob_left > 0.5, 0, 1)
            correct = (preds == labels).astype(float)

            p_true_raw = np.where(labels == 0, pred_prob_left, 1.0 - pred_prob_left)

            raw_ece = compute_ece(confs_raw, correct)
            raw_ada = compute_ada_ece(confs_raw, correct)
            raw_nll = -np.mean(np.log(np.clip(p_true_raw, 1e-12, 1.0)))
            raw_brier = np.mean((1.0 - p_true_raw) ** 2)
            results_raw_cnn[atype][K] = {
                "ECE": raw_ece, "AdaECE": raw_ada, "NLL": raw_nll, "Brier": raw_brier
            }

            p_anom_all = np.concatenate([probs_id[:n_id_cnn], probs_anom[:n_anom_cnn]], axis=0)

            results_K = {}
            for m in range(9):
                p_a = p_anom_all[:, m]
                confs_adj = confs_raw * (1.0 - p_a) + 0.5 * p_a
                p_true_adj = p_true_raw * (1.0 - p_a) + 0.5 * p_a

                ece = compute_ece(confs_adj, correct)
                ada_ece = compute_ada_ece(confs_adj, correct)
                nll = -np.mean(np.log(np.clip(p_true_adj, 1e-12, 1.0)))
                brier = np.mean((1.0 - p_true_adj) ** 2)
                results_K[m] = {"ECE": ece, "AdaECE": ada_ece, "NLL": nll, "Brier": brier}

            results_adj[atype][K] = results_K

            eces = [results_K[m]["ECE"] for m in range(9)]
            best_m = int(np.argmin(eces))
            print(f"  K={K:>3}: Raw CNN ECE={raw_ece:.4f} → best adjusted={METHOD_NAMES[best_m]} "
                  f"({eces[best_m]:.4f}), nFrames={n_id_cnn+n_anom_cnn}")

    results = results_adj

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22']
    linestyles = ['-', '-', '-', '--', '--', '--', ':', ':', ':']
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']

    for atype, results_type in results.items():
        if not results_type:
            continue

        raw_cnn_type = results_raw_cnn.get(atype, {})
        label = atype.replace("real_", "").capitalize()
        horizons_available = sorted(results_type.keys())

        if len(horizons_available) < 2:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"CNN on Predicted Frames — Calibration — {label}\n"
                     f"Black dashed = Raw CNN on predicted frames (uncalibrated)  |  "
                     f"Colored = adjusted by anomaly method",
                     fontsize=12, fontweight='bold')

        metric_list = [("ECE", "ECE (↓ better)"),
                       ("AdaECE", "Adaptive ECE (↓ better)"),
                       ("NLL", "Negative Log-Likelihood (↓ better)"),
                       ("Brier", "Brier Score (↓ better)")]

        for ax_idx, (metric_key, metric_label) in enumerate(metric_list):
            ax = axes[ax_idx // 2, ax_idx % 2]

            xs_raw, ys_raw = [], []
            for K in horizons_available:
                if K in raw_cnn_type and metric_key in raw_cnn_type[K]:
                    val = raw_cnn_type[K][metric_key]
                    if not np.isnan(val):
                        xs_raw.append(K)
                        ys_raw.append(val)
            if xs_raw:
                ax.plot(xs_raw, ys_raw, color='black', linestyle='--',
                        marker='*', markersize=9, linewidth=2.5,
                        label='Raw CNN on predicted frames', alpha=0.9, zorder=10)

            for m in range(9):
                xs, ys = [], []
                for K in horizons_available:
                    if m in results_type[K]:
                        xs.append(K)
                        ys.append(results_type[K][m][metric_key])
                if xs:
                    ax.plot(xs, ys, color=colors[m], linestyle=linestyles[m],
                            marker=markers[m], markersize=5, linewidth=1.8,
                            label=METHOD_NAMES[m], alpha=0.85)

            ax.set_xlabel("Horizon K", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(horizons_available)
            ax.set_ylim(bottom=0)
            if ax_idx == 0:
                ax.legend(fontsize=6.5, ncol=3, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.89])
        save_path = os.path.join(OUTPUT_DIR, f"method_ece_{atype.replace('real_', '')}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {save_path}")

    K0 = HORIZONS[0]
    n_atypes = len([a for a in anomaly_types if a in results and K0 in results[a]])
    fig, axes = plt.subplots(1, max(n_atypes, 1), figsize=(8 * max(n_atypes, 1), 6))
    if n_atypes <= 1:
        axes = [axes]
    fig.suptitle(f"Method ECE Comparison at K={K0}", fontsize=14, fontweight='bold')

    ax_counter = 0
    for atype in anomaly_types:
        if atype not in results or K0 not in results[atype]:
            continue

        ax = axes[ax_counter]
        ax_counter += 1
        label = atype.replace("real_", "").capitalize()

        eces = [results[atype][K0][m]["ECE"] for m in range(9)]
        x_pos = np.arange(9)

        bars = ax.bar(x_pos, eces, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        for i, (bar, val) in enumerate(zip(bars, eces)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(METHOD_NAMES, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("ECE (↓ better)", fontsize=11)
        ax.set_title(f"{label} — ECE at K={K0}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"method_ece_bar_K{K0}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    print(f"\n{'='*90}")
    print(f"CNN ON PREDICTED FRAMES — CALIBRATION SUMMARY at K={K0}")
    print(f"  conf = pred_conf (CNN on LSTM→VAE reconstructed frame)")
    print(f"  conf_adj = conf*(1-p_anom) + 0.5*p_anom")
    print(f"{'='*90}")
    for atype in anomaly_types:
        if atype not in results or K0 not in results[atype]:
            continue
        label = atype.replace("real_", "").upper()
        print(f"\n  --- {label} ---")
        print(f"  {'Method':<18} | {'ECE':>8} | {'AdaECE':>8} | {'NLL':>8} | {'Brier':>8}")
        print(f"  {'-'*58}")
        if atype in results_raw_cnn and K0 in results_raw_cnn[atype]:
            r = results_raw_cnn[atype][K0]
            print(f"  {'Raw pred_conf':<18} | {r['ECE']:>8.4f} | {r['AdaECE']:>8.4f} | "
                  f"{r['NLL']:>8.4f} | {r['Brier']:>8.4f}")
            print(f"  {'-'*58}")
        for m in range(9):
            r = results[atype][K0][m]
            print(f"  {METHOD_NAMES[m]:<18} | {r['ECE']:>8.4f} | {r['AdaECE']:>8.4f} | "
                  f"{r['NLL']:>8.4f} | {r['Brier']:>8.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
