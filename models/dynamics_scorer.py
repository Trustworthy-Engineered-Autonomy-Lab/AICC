# Original: predictor/lstm_dynamics_scorer.py
"""
LSTM Dynamics Anomaly Scorer

Uses MC Dropout to estimate LSTM prediction uncertainty (STD),
then maps STD to P(ID|STD) via KDE as a dynamics anomaly score.
"""

import os
import numpy as np
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn


class LSTMWithMCDropout(nn.Module):
    """LSTM predictor with MC Dropout"""
    
    def __init__(self, hidden_size=256, latent_flat_dim=1024, dropout_rate=0.2):
        super().__init__()
        self.latent_flat_dim = latent_flat_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.lstm = nn.LSTM(
            input_size=latent_flat_dim + 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm_out = nn.Linear(hidden_size, latent_flat_dim)
    
    def forward(self, x, hidden=None):
        """
        x: (B, T, latent_flat_dim + 2)
        Returns: output delta, (hn, cn)
        """
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.dropout(out)
        delta = self.lstm_out(out)
        return delta, (hn, cn)
    
    def enable_mc_dropout(self):
        """Force-enable dropout for MC sampling"""
        self.train()
        self.lstm.train()
        self.dropout.train()
    
    def predict_with_mc(self, latent_seq, action_seq, hidden=None, mc_samples=50):
        """
        MC Dropout prediction, returns results from multiple samples
        
        latent_seq: (B, T, latent_flat_dim)
        action_seq: (B, T, 2)
        mc_samples: number of samples
        
        Returns:
            mean: (B, latent_flat_dim) prediction mean
            std: (B, latent_flat_dim) prediction std
            samples: (mc_samples, B, latent_flat_dim) all MC samples
        """
        self.enable_mc_dropout()
        
        B, T, D = latent_seq.shape
        x = torch.cat([latent_seq, action_seq], dim=-1)
        
        if hidden is None:
            h0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        samples = []
        for _ in range(mc_samples):
            out, _ = self.lstm(x, hidden)
            out = self.dropout(out)
            delta = self.lstm_out(out[:, -1])
            pred = latent_seq[:, -1] + delta
            samples.append(pred)
        
        samples = torch.stack(samples, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std, samples


class LSTMDynamicsScorer:
    """
    LSTM dynamics anomaly scorer
    
    Maps LSTM prediction STD to P(ID|STD) via KDE
    """
    
    def __init__(self, results_path='predictor/lstm_std_results.npz'):
        """
        Args:
            results_path: Path to npz file containing ID and OOD STD distributions
        """
        self.results_path = results_path
        self.id_kde = None
        self.ood_kde = None
        self.id_prior = 0.5
        self.ood_prior = 0.5
        
        if os.path.exists(results_path):
            self.load_distributions(results_path)
    
    def load_distributions(self, results_path):
        """Load pre-computed STD distributions"""
        data = np.load(results_path)
        
        self.id_stds = data['id_stds']
        self.ood_stds = data['ood_stds']
        
        self.id_log_stds = np.log10(self.id_stds + 1e-8)
        self.ood_log_stds = np.log10(self.ood_stds + 1e-8)
        
        self.id_kde = gaussian_kde(self.id_log_stds, bw_method='scott')
        self.ood_kde = gaussian_kde(self.ood_log_stds, bw_method='scott')
        
        self.id_mean = np.mean(self.id_log_stds)
        self.id_std = np.std(self.id_log_stds)
        self.ood_mean = np.mean(self.ood_log_stds)
        self.ood_std = np.std(self.ood_log_stds)
        
        print(f"LSTMDynamicsScorer initialized:")
        print(f"  ID distribution: μ={self.id_mean:.4f}, σ={self.id_std:.4f}")
        print(f"  OOD distribution: μ={self.ood_mean:.4f}, σ={self.ood_std:.4f}")
    
    def get_id_probability(self, std_value, method='bayes'):
        """
        Compute P(ID|STD) - probability of being ID given STD
        
        Args:
            std_value: LSTM prediction STD (scalar or array)
            method: 'bayes' (Bayesian posterior), 'sigmoid', 'gaussian_cdf'
            
        Returns:
            P(ID|STD): probability in [0, 1]
        """
        if self.id_kde is None or self.ood_kde is None:
            raise ValueError("KDE not initialized. Run collect_std_distributions first.")
        
        log_std = np.log10(np.maximum(std_value, 1e-8))
        
        if method == 'bayes':
            p_std_given_id = self.id_kde(log_std)
            p_std_given_ood = self.ood_kde(log_std)
            
            p_std = p_std_given_id * self.id_prior + p_std_given_ood * self.ood_prior
            
            p_id_given_std = (p_std_given_id * self.id_prior) / (p_std + 1e-10)
            
            p_id_given_std = np.clip(p_id_given_std, 0.01, 0.99)
            
        elif method == 'sigmoid':
            z_score = (log_std - self.id_mean) / (self.id_std + 1e-8)
            p_id_given_std = 1.0 / (1.0 + np.exp(z_score))
            p_id_given_std = np.clip(p_id_given_std, 0.01, 0.99)
            
        elif method == 'gaussian_cdf':
            from scipy.stats import norm
            cdf = norm.cdf(log_std, loc=self.id_mean, scale=self.id_std)
            p_id_given_std = 1.0 - cdf
            p_id_given_std = np.clip(p_id_given_std, 0.01, 0.99)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(p_id_given_std) if np.isscalar(std_value) else p_id_given_std


def collect_std_distributions(
    vae, predictor, device,
    id_npz_files, ood_npz_files,
    output_path='predictor/lstm_std_results.npz',
    mc_samples=50, num_sequences=100, context_length=100
):
    """
    Collect LSTM prediction STD distributions for ID and OOD data
    
    Args:
        vae: VAE model
        predictor: LSTM predictor (must support MC Dropout)
        device: torch device
        id_npz_files: List of NPZ files for normal (ID) data
        ood_npz_files: List of NPZ files for OOD data
        output_path: Output file path
        mc_samples: Number of MC Dropout samples
        num_sequences: Number of sequences to sample per dataset
        context_length: Context length
    """
    import torch.nn.functional as F
    
    vae.eval()
    
    def compute_stds_for_dataset(npz_files, num_seq):
        """Compute LSTM prediction STD for a dataset"""
        all_stds = []
        
        all_frames = []
        all_actions = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            if 'frame' in data:
                frames = data['frame']
            elif 'frames' in data:
                frames = data['frames']
            else:
                frames = data[list(data.keys())[0]]
            
            actions = data['action'] if 'action' in data else np.zeros((len(frames), 2), dtype=np.float32)
            
            if frames.shape[-1] == 3:
                frames = np.transpose(frames, (0, 3, 1, 2))
            if frames.max() > 1.0:
                frames = frames.astype(np.float32) / 255.0
            
            all_frames.append(frames)
            all_actions.append(actions)
        
        all_frames = np.concatenate(all_frames, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        max_start = len(all_frames) - context_length - 1
        if max_start <= 0:
            print(f"Warning: Not enough frames for context_length={context_length}")
            return np.array([])
        
        starts = np.random.choice(max_start, min(num_seq, max_start), replace=False)
        
        with torch.no_grad():
            for i, start in enumerate(starts):
                if i % 20 == 0:
                    print(f"    Processing sequence {i+1}/{len(starts)}...")
                
                frames = all_frames[start:start + context_length]
                actions = all_actions[start:start + context_length]
                
                frames_t = torch.from_numpy(frames).float().to(device)
                actions_t = torch.from_numpy(actions).float().to(device)
                
                latents = []
                for t in range(context_length):
                    z, _, _ = vae.encode(frames_t[t:t+1])
                    latents.append(z.view(-1))
                latents = torch.stack(latents, dim=0).unsqueeze(0)
                
                predictor.enable_mc_dropout()
                _, C, H, W = z.shape
                D = C * H * W
                
                x = torch.cat([latents, actions_t.unsqueeze(0)], dim=-1)
                
                samples = []
                for _ in range(mc_samples):
                    h0 = torch.zeros(predictor.lstm.num_layers, 1, predictor.lstm.hidden_size, device=device)
                    c0 = torch.zeros(predictor.lstm.num_layers, 1, predictor.lstm.hidden_size, device=device)
                    
                    out, _ = predictor.lstm(x, (h0, c0))
                    out = predictor.dropout(out) if hasattr(predictor, 'dropout') else out
                    delta = predictor.lstm_out(out[:, -1])
                    samples.append(delta)
                
                samples = torch.stack(samples, dim=0)
                std = samples.std(dim=0).mean().item()
                all_stds.append(std)
        
        return np.array(all_stds)
    
    print("Collecting ID STD distribution...")
    id_stds = compute_stds_for_dataset(id_npz_files, num_sequences)
    
    print("Collecting OOD STD distribution...")
    ood_stds = compute_stds_for_dataset(ood_npz_files, num_sequences)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path,
             id_stds=id_stds,
             ood_stds=ood_stds,
             mc_samples=mc_samples,
             context_length=context_length)
    
    print(f"\nSaved STD distributions to {output_path}")
    print(f"  ID: {len(id_stds)} samples, mean={np.mean(id_stds):.6f}, std={np.std(id_stds):.6f}")
    print(f"  OOD: {len(ood_stds)} samples, mean={np.mean(ood_stds):.6f}, std={np.std(ood_stds):.6f}")
    
    return id_stds, ood_stds


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect LSTM STD distributions')
    parser.add_argument('--id_npz_files', nargs='+', required=True, help='ID data NPZ files')
    parser.add_argument('--ood_npz_files', nargs='+', required=True, help='OOD data NPZ files')
    parser.add_argument('--vae_path', default='vae_recon/finetuned_ctx100/best_model.pt')
    parser.add_argument('--predictor_path', default='predictor/checkpoints_ctx100/best_model.pt')
    parser.add_argument('--output_path', default='predictor/lstm_std_results.npz')
    parser.add_argument('--mc_samples', type=int, default=50)
    parser.add_argument('--num_sequences', type=int, default=100)
    parser.add_argument('--context_length', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    from vae_recon.vae_model_64x64 import SimpleVAE64x64
    print("Loading VAE...")
    vae_ckpt = torch.load(args.vae_path, map_location=device, weights_only=False)
    vae = SimpleVAE64x64(latent_dim=64)
    if 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    vae = vae.to(device).eval()
    
    print("Loading LSTM Predictor...")
    pred_ckpt = torch.load(args.predictor_path, map_location=device, weights_only=False)
    predictor = LSTMWithMCDropout(hidden_size=256, latent_flat_dim=1024, dropout_rate=0.2)
    state_dict = pred_ckpt['model_state_dict'] if 'model_state_dict' in pred_ckpt else pred_ckpt
    lstm_state = {k: v for k, v in state_dict.items() if k.startswith('lstm.') or k.startswith('lstm_out.')}
    predictor.load_state_dict(lstm_state, strict=False)
    predictor = predictor.to(device)
    
    collect_std_distributions(
        vae, predictor, device,
        args.id_npz_files, args.ood_npz_files,
        args.output_path,
        args.mc_samples, args.num_sequences, args.context_length
    )
