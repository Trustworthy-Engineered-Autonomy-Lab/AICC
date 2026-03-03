# Original: evaluation/calibration/benchmark_runtime.py
#!/usr/bin/env python3
"""Runtime benchmark: original vs optimized pipeline."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F

VAE_PATH  = "vae_recon/checkpoints_v2_rgb/best_model.pt"
LSTM_PATH = "predictor/checkpoints_v2_rgb_recdrop/best_model.pt"
CNN_PATH  = "lane_classifier/checkpoints/best_model_finetuned.pt"
CTX, MK, MC, NAUG = 100, 50, 3, 8
WU, REP = 20, 100
IMEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
ISTD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_models(dev):
    from vae_recon.vae_model_64x64_v2 import load_model_v2
    from predictor.core.vae_predictor_v2 import VAEPredictorV2
    from lane_classifier.models.cnn_model import LaneCNN

    vae = load_model_v2(VAE_PATH, dev); vae.eval()

    ck = torch.load(LSTM_PATH, map_location=dev, weights_only=False)
    c, a = ck.get("config", {}), ck.get("args", {})
    lstm = VAEPredictorV2(
        latent_dim=c.get("latent_dim", 64), spatial_size=c.get("spatial_size", 4),
        action_dim=c.get("action_dim", 2), hidden_channels=c.get("hidden_channels", 128),
        num_layers=c.get("num_layers", 2), dropout=a.get("dropout", 0.3),
        recurrent_dropout=c.get("recurrent_dropout", 0.0),
        residual=c.get("residual", True),
    ).to(dev)
    ps = {k: v for k, v in ck["model_state_dict"].items() if not k.startswith("vae_")}
    lstm.load_state_dict(ps, strict=False)

    cc = torch.load(CNN_PATH, map_location=dev, weights_only=False)
    cnn = LaneCNN(dropout_rate=cc.get("args", {}).get("dropout", 0.5)).to(dev)
    cnn.load_state_dict(cc["model_state_dict"]); cnn.eval()
    return vae, lstm, cnn


def apply_aug(x):
    import torchvision.transforms.functional as TF
    import random
    x = x.clone()
    x = TF.adjust_brightness(x, 1.0 + random.uniform(-0.25, 0.25))
    x = TF.adjust_contrast(x, 1.0 + random.uniform(-0.2, 0.2))
    x = TF.adjust_saturation(x, 1.0 + random.uniform(-0.15, 0.15))
    if random.random() < 0.5:
        x = (x + torch.randn_like(x) * random.uniform(0.01, 0.05)).clamp(0, 1)
    return x


class GpuTimer:
    def __init__(self, dev):
        self.cuda = dev.type == "cuda"
    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()
    def now(self):
        self.sync()
        return time.perf_counter()


def bench(timer, fn, warmup=WU, reps=REP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = timer.now()
        fn()
        ts.append(timer.now() - t0)
    return np.median(ts) * 1000


def run(dev):
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print(f"Device: {dev}  ({gpu})")
    print(f"CTX={CTX} K={MK} MC={MC} AUG={NAUG} WU={WU} REP={REP}")
    print()

    vae, lstm, cnn = load_models(dev)
    timer = GpuTimer(dev)
    mt = torch.tensor(IMEAN, device=dev)
    st = torch.tensor(ISTD, device=dev)

    fc = torch.rand(CTX, 3, 64, 64, device=dev)
    ff = torch.rand(CTX + MK, 3, 64, 64, device=dev)
    ac = torch.randn(1, CTX + MK, 2, device=dev)
    si = torch.rand(1, 3, 64, 64, device=dev)
    sn = (si - mt) / st

    R = {}

    def f_enc():
        with torch.no_grad():
            vae.encode(fc)
    R["VAE encode (100 frames)"] = bench(timer, f_enc)

    z1 = torch.randn(1, 64, 4, 4, device=dev)
    def f_dec():
        with torch.no_grad():
            vae.decode(z1)
    R["VAE decode (1 frame)"] = bench(timer, f_dec)

    with torch.no_grad():
        parts = []
        for i in range(0, len(ff), 32):
            mu, _, _ = vae.encode(ff[i:i+32])
            parts.append(mu)
        lat = torch.cat(parts, dim=0)
    cz = lat[:CTX].unsqueeze(0)
    cm = lat[:CTX]

    def f_phi_naive():
        with torch.no_grad():
            mu, _, _ = vae.encode(fc)
            recon = vae.decode(mu)
            F.mse_loss(recon, fc)
    R["phi naive (enc+dec 100fr)"] = bench(timer, f_phi_naive)

    def f_phi_opt():
        with torch.no_grad():
            recon = vae.decode(cm)
            F.mse_loss(recon, fc)
    R["phi opt (dec-only 100fr)"] = bench(timer, f_phi_opt)

    def f_lstm():
        lstm.eval()
        with torch.no_grad():
            lstm.predict_sequence(cz, ac, num_steps=MK, return_primed_hidden=True)
    R["LSTM det (100+50 steps)"] = bench(timer, f_lstm)

    def f_mc_seq():
        for _ in range(MC):
            lstm.train()
            with torch.no_grad():
                lstm.predict_sequence(cz, ac, num_steps=MK)
        lstm.eval()
    R["MC seq (3x150, OLD)"] = bench(timer, f_mc_seq)

    czm = cz.expand(MC, -1, -1, -1, -1).contiguous()
    acm = ac.expand(MC, -1, -1).contiguous()
    def f_mc_bat():
        lstm.train()
        with torch.no_grad():
            lstm.predict_sequence(czm, acm, num_steps=MK)
        lstm.eval()
    R["MC batch (B=3, NEW)"] = bench(timer, f_mc_bat)

    def f_cnn():
        with torch.no_grad():
            cnn(sn)
    R["CNN inference (1 image)"] = bench(timer, f_cnn)

    pc = si[0].cpu()
    def f_tta():
        with torch.no_grad():
            F.softmax(cnn((si - mt) / st), dim=1)
        for _ in range(NAUG):
            aug = apply_aug(pc).unsqueeze(0).to(dev)
            with torch.no_grad():
                F.softmax(cnn((aug - mt) / st), dim=1)
    R["TTA (9 CNN calls)"] = bench(timer, f_tta)

    fe = np.random.randn(24).astype(np.float32)
    md = np.random.randn(24).astype(np.float32)
    ci = np.eye(24, dtype=np.float32)
    def f_maha():
        d = fe - md
        np.sqrt(max(0.0, d @ ci @ d))
    R["delta: Mahalanobis (24d)"] = bench(timer, f_maha)

    pr, Tk = 0.73, 1.5
    lv = np.log(pr / (1 - pr))
    wp, ws, pn, sn2 = 0.3, 0.2, 0.4, 0.6

    def f_ts():
        1.0 / (1.0 + np.exp(-lv / Tk))
    R["Cal: Temp Scaling"] = bench(timer, f_ts)

    def f_ours():
        Te = Tk * np.exp(wp * pn + ws * sn2)
        1.0 / (1.0 + np.exp(-lv / Te))
    R["Cal: Ours (anom-cond)"] = bench(timer, f_ours)

    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=10).fit(np.random.randn(500, 24).astype(np.float32))
    def f_dac():
        d, _ = knn.kneighbors(fe.reshape(1, -1))
        Te = Tk * np.exp(2.0 * d.mean())
        1.0 / (1.0 + np.exp(-lv / Te))
    R["Cal: DAC (KNN k=10)"] = bench(timer, f_dac)

    enc = R["VAE encode (100 frames)"]
    dec = R["VAE decode (1 frame)"]
    lstm_t = R["LSTM det (100+50 steps)"]
    mc_old = R["MC seq (3x150, OLD)"]
    mc_new = R["MC batch (B=3, NEW)"]
    cnn_t = R["CNN inference (1 image)"]
    tta_t = R["TTA (9 CNN calls)"]
    phi_n = R["phi naive (enc+dec 100fr)"]
    phi_o = R["phi opt (dec-only 100fr)"]
    maha = R["delta: Mahalanobis (24d)"]
    c_ts = R["Cal: Temp Scaling"]
    c_o = R["Cal: Ours (anom-cond)"]

    R["--- Base (TS)"] = enc + lstm_t + dec + cnn_t + c_ts
    R["--- +TTA"] = enc + lstm_t + dec + tta_t + c_ts
    R["--- Full OLD"] = enc + lstm_t + mc_old + dec + tta_t + phi_n + maha + c_o
    R["--- Full NEW"] = enc + lstm_t + mc_new + dec + tta_t + phi_o + maha + c_o
    old_t = R["--- Full OLD"]
    new_t = R["--- Full NEW"]

    print("=" * 62)
    hdr = "Component"
    print(f"{hdr:<45} {'ms':>12}")
    print("=" * 62)
    for name, ms in R.items():
        if name.startswith("---"):
            print("-" * 62)
        print(f"{name:<45} {ms:>11.3f}")
    print("=" * 62)

    print()
    print("Model sizes:")
    for lb, m in [("VAE", vae), ("LSTM", lstm), ("CNN", cnn)]:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {lb}: {n:,} ({n * 4 / 1e6:.1f} MB)")

    print()
    print(f"OLD pipeline: {old_t:.0f} ms = {1000/old_t:.1f} samp/s")
    print(f"NEW pipeline: {new_t:.0f} ms = {1000/new_t:.1f} samp/s")
    ratio = old_t / new_t if new_t > 0 else 0
    print(f"Speedup: {ratio:.2f}x")


if __name__ == "__main__":
    run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
