## donkey_car – predictor + vae_recon (code-only)

This repository contains an **`Appendix.pdf`** with additional training details, features, calibration protocols, and evaluation metrics.

Aside from that, this repository is **code-only**: it tracks `predictor/` and `vae_recon/`. It does not contain other artifacts (datasets, frames, checkpoints, eval outputs) due to their prohibitive size.

### Project layout

- **`vae_recon/`**: VAE models + training script for image reconstruction (64×64 and 224×224 variants).
- **`predictor/`**: VAE-latent trajectory predictor (LSTM/GRU) + training/evaluation scripts.

### Setup

Create a venv and install deps:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Train VAE (reconstruction)

From repo root:

```bash
python vae_recon/train_enhanced.py --image_size 64 --data_dir ../npz_transfer --npz_files traj1.npz traj2.npz
```

### Train predictor (latent dynamics)

From repo root:

```bash
python predictor/train_predictor.py --data_dir ../npz_transfer --npz_files traj1.npz traj2.npz --sequence_length 16
```

### Evaluate predictor

```bash
python predictor/eval_predictor.py --model_path predictor/checkpoints/best_model.pt --data_dir ../npz_transfer --npz_files traj1.npz traj2.npz
```



