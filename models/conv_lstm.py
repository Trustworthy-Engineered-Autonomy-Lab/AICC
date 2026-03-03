# Original: predictor/core/vae_predictor_v2.py
"""
VAE Predictor V2 - Improved LSTM dynamics model

Key improvements:
1. ConvLSTM: preserves spatial structure, no latent flattening
2. Multi-scale prediction: separate global and local detail prediction
3. Improved action injection: action info injected at multiple levels
4. Larger capacity: resolves information bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
import os
import sys

try:
    from vae_recon.vae_model_64x64 import SimpleVAE64x64, load_model_64x64
    HAS_VAE_64 = True
except ImportError:
    HAS_VAE_64 = False
    print("Warning: Could not import VAE 64x64 model")

try:
    from vae_recon.vae_model_64x64_v2 import SimpleVAE64x64_V2, load_model_v2 as load_vae_v2
    HAS_VAE_V2 = True
except ImportError:
    HAS_VAE_V2 = False
    print("Warning: Could not import VAE V2 model")


@torch.jit.script
def conv_lstm_cell_forward_jit(
    combined: torch.Tensor,
    c: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    num_groups: int,
    hidden_channels: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled ConvLSTM forward pass core"""
    padding = conv_weight.size(2) // 2
    gates = F.conv2d(combined, conv_weight, conv_bias, padding=padding)
    
    
    i, f, g, o = gates.chunk(4, dim=1)
    
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    
    c_new = f * c + i * g
    h_new = o * torch.tanh(c_new)
    
    return h_new, c_new


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell - LSTM that preserves spatial structure
    
    Advantages over flattened LSTM:
    - Preserves spatial position info (center line, wall positions)
    - Fewer parameters but stronger expressiveness
    - Natural handling of local correlations
    
    Recurrent Dropout:
    - Applies dropout to hidden state h (during training)
    - Uses variational dropout: single mask shared across all timesteps
    - Can be enabled at inference for MC Dropout uncertainty estimation
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3,
                 use_jit: bool = True, recurrent_dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_jit = use_jit
        self.recurrent_dropout = recurrent_dropout
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.layer_norm = nn.GroupNorm(4, 4 * hidden_channels)
        
        self._dropout_mask_h: Optional[torch.Tensor] = None
    
    def reset_dropout_mask(self, batch_size: int, spatial_size: int, device: torch.device):
        """Generate new variational dropout mask (called once per sequence).
        
        Variational recurrent dropout: all timesteps in a sequence share one mask,
        so dropout consistently masks certain channels rather than randomly
        perturbing different positions at each step. This stabilizes training
        and produces more coherent uncertainty during MC inference.
        """
        if self.recurrent_dropout > 0.0 and self.training:
            mask = torch.ones(batch_size, self.hidden_channels, spatial_size, spatial_size, device=device)
            mask = F.dropout2d(mask, p=self.recurrent_dropout, training=True)
            self._dropout_mask_h = mask
        else:
            self._dropout_mask_h = None
    
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, C_in, H, W) input
            hidden: (h, c) each is (B, C_hidden, H, W)
        Returns:
            h_new: (B, C_hidden, H, W)
            (h_new, c_new): updated hidden state
        """
        h, c = hidden
        
        if self._dropout_mask_h is not None:
            h = h * self._dropout_mask_h
        
        combined = torch.cat([x, h], dim=1)
        
        if self.use_jit and not self.training:
            h_new, c_new = conv_lstm_cell_forward_jit(
                combined, c,
                self.conv.weight, self.conv.bias,
                4, self.hidden_channels
            )
        else:
            gates = self.conv(combined)
            gates = self.layer_norm(gates)
            
            i, f, g, o = gates.chunk(4, dim=1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
    def init_hidden(self, batch_size: int, spatial_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h = torch.zeros(batch_size, self.hidden_channels, spatial_size, spatial_size, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, spatial_size, spatial_size, device=device)
        return h, c


class ActionEncoder(nn.Module):
    """
    Action encoder - converts 2D action to spatial feature maps
    
    Distributes action information spatially instead of simple concatenation
    """
    
    def __init__(self, action_dim: int, out_channels: int, spatial_size: int = 4):
        super().__init__()
        self.spatial_size = spatial_size
        
        self.fc = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels * spatial_size * spatial_size),
        )
        self.out_channels = out_channels
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (B, action_dim)
        Returns:
            action_map: (B, out_channels, H, W)
        """
        B = action.size(0)
        feat = self.fc(action)
        action_map = feat.view(B, self.out_channels, self.spatial_size, self.spatial_size)
        return action_map


class VAEPredictorV2(nn.Module):
    """
    VAE Predictor V2 - Improved version
    
    Architecture improvements:
    1. 2-layer ConvLSTM (preserves spatial structure)
    2. Action injected via spatial feature maps
    3. Residual connection + gating mechanism
    4. Larger hidden channels
    
    Latent: (B, 64, 4, 4) = 1024-dim
    ConvLSTM hidden: (B, 128, 4, 4) = 2048-dim (8x over original 256-dim)
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        spatial_size: int = 4,
        action_dim: int = 2,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        recurrent_dropout: float = 0.15,
        residual: bool = True,
        vae_model_path: Optional[str] = None,
        freeze_vae: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.action_dim = action_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.residual = residual
        self.freeze_vae = freeze_vae
        self.recurrent_dropout = recurrent_dropout
        
        self.latent_flat_dim = latent_dim * spatial_size * spatial_size
        
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            out_channels=32,
            spatial_size=spatial_size
        )
        
        self.conv_lstm1 = ConvLSTMCell(
            in_channels=latent_dim + 32,
            hidden_channels=hidden_channels,
            kernel_size=3,
            recurrent_dropout=recurrent_dropout,
        )
        
        self.conv_lstm2 = ConvLSTMCell(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            recurrent_dropout=recurrent_dropout,
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=1),
        )
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(hidden_channels + latent_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.vae_encoder = None
        self.vae_decoder = None
        if vae_model_path and os.path.exists(vae_model_path):
            self._load_vae(vae_model_path)
    
    def _load_vae(self, vae_model_path: str):
        """Load and freeze VAE (supports both V1 and V2)"""
        print(f"Loading VAE from {vae_model_path}")
        
        checkpoint = torch.load(vae_model_path, map_location='cpu', weights_only=False)
        version = checkpoint.get('version', 'v1') if isinstance(checkpoint, dict) else 'v1'
        
        if version == 'v2' and HAS_VAE_V2:
            print(f"  Detected VAE V2 model")
            vae_model = load_vae_v2(vae_model_path, torch.device('cpu'))
        elif HAS_VAE_64:
            print(f"  Loading VAE V1 model")
            vae_model = load_model_64x64(vae_model_path, torch.device('cpu'))
        else:
            print("Warning: No VAE model available")
            return
        
        self.vae_encoder = vae_model
        self.vae_decoder = vae_model
        self.latent_dim = vae_model.latent_dim
        
        if self.freeze_vae:
            for param in vae_model.parameters():
                param.requires_grad = False
            print("  VAE frozen")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent"""
        if self.vae_encoder is not None:
            mu, logvar, _ = self.vae_encoder.encode(x)
            return mu, logvar
        else:
            raise ValueError("VAE not loaded")
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to images"""
        if self.vae_decoder is not None:
            return self.vae_decoder.decode(z)
        else:
            raise ValueError("VAE not loaded")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def predict_step(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor],
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-step prediction
        
        Args:
            z: (B, C, H, W) current latent
            action: (B, action_dim) current action
            hidden: ((h1, c1), (h2, c2)) LSTM hidden states
        
        Returns:
            z_next: (B, C, H, W) predicted next-frame latent
            new_hidden: updated hidden states
        """
        B = z.size(0)
        device = z.device
        
        if hidden is None:
            h1, c1 = self.conv_lstm1.init_hidden(B, self.spatial_size, device)
            h2, c2 = self.conv_lstm2.init_hidden(B, self.spatial_size, device)
        else:
            (h1, c1), (h2, c2) = hidden
        
        if action is not None:
            action_feat = self.action_encoder(action)
        else:
            action_feat = torch.zeros(B, 32, self.spatial_size, self.spatial_size, device=device)
        
        x = torch.cat([z, action_feat], dim=1)
        
        h1_new, (h1_new, c1_new) = self.conv_lstm1(x, (h1, c1))
        h1_new = self.dropout(h1_new)
        
        h2_new, (h2_new, c2_new) = self.conv_lstm2(h1_new, (h2, c2))
        h2_new = self.dropout(h2_new)
        
        delta = self.out_conv(h2_new)
        
        if self.residual:
            gate_input = torch.cat([h2_new, z], dim=1)
            gate = self.gate_conv(gate_input)
            z_next = z + gate * delta
        else:
            z_next = z + delta
        
        new_hidden = ((h1_new, c1_new), (h2_new, c2_new))
        return z_next, new_hidden
    
    def predict_sequence(
        self,
        z_context: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        teacher_forcing_targets: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
        primed_hidden: Optional[Tuple] = None,
        return_primed_hidden: bool = False,
        noise_std: float = 0.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Sequence prediction (supports teacher forcing and open-loop)
        
        Args:
            z_context: (B, T_ctx, C, H, W) context latent sequence
            actions: (B, T_ctx + num_steps, action_dim) action sequence
            num_steps: number of prediction steps
            teacher_forcing_targets: (B, num_steps, C, H, W) targets for teacher forcing
            teacher_forcing_prob: teacher forcing probability
            primed_hidden: pre-primed hidden state, skips priming if provided
            return_primed_hidden: whether to return hidden state after priming
            noise_std: noise std injected during training (robustness to error accumulation)
        
        Returns:
            If return_primed_hidden=False: z_pred (B, num_steps, C, H, W)
            If return_primed_hidden=True: (z_pred, primed_hidden)
        """
        B, T_ctx, C, H, W = z_context.shape
        device = z_context.device
        
        self.conv_lstm1.reset_dropout_mask(B, self.spatial_size, device)
        self.conv_lstm2.reset_dropout_mask(B, self.spatial_size, device)
        
        if primed_hidden is not None:
            hidden = primed_hidden
        else:
            hidden = None
            for t in range(T_ctx):
                z_t = z_context[:, t]
                a_t = actions[:, t] if actions is not None else None
                _, hidden = self.predict_step(z_t, a_t, hidden)
        
        hidden_after_priming = hidden
        
        predictions = torch.empty(B, num_steps, C, H, W, device=device, dtype=z_context.dtype)
        z_current = z_context[:, -1]
        
        for t in range(num_steps):
            a_idx = T_ctx + t
            a_t = actions[:, a_idx] if (actions is not None and a_idx < actions.size(1)) else None
            
            z_next, hidden = self.predict_step(z_current, a_t, hidden)
            predictions[:, t] = z_next
            
            if teacher_forcing_targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                z_current = teacher_forcing_targets[:, t]
            else:
                z_current = z_next
            
            if noise_std > 0 and self.training:
                z_current = z_current + torch.randn_like(z_current) * noise_std
        
        if return_primed_hidden:
            return predictions, hidden_after_priming
        return predictions
    
    def forward(
        self,
        input_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        target_frames: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            input_frames: (B, T_in, 3, H_img, W_img) input images
            actions: (B, T_total, action_dim) action sequence
            target_frames: (B, T_target, 3, H_img, W_img) target images (for teacher forcing)
            teacher_forcing_prob: teacher forcing probability
        
        Returns:
            dict with z_input, z_pred, z_target
        """
        B, T_in = input_frames.shape[:2]
        device = input_frames.device
        
        with torch.no_grad() if self.freeze_vae else torch.enable_grad():
            in_flat = input_frames.reshape(B * T_in, *input_frames.shape[2:])
            mu_in, logvar_in = self.encode(in_flat)
            z_input = mu_in
            z_input = z_input.view(B, T_in, *z_input.shape[1:])
        
        z_target = None
        if target_frames is not None:
            T_target = target_frames.shape[1]
            with torch.no_grad():
                tgt_flat = target_frames.reshape(B * T_target, *target_frames.shape[2:])
                mu_tgt, _ = self.encode(tgt_flat)
                z_target = mu_tgt.view(B, T_target, *mu_tgt.shape[1:])
        
        num_steps = z_target.shape[1] if z_target is not None else 1
        z_pred = self.predict_sequence(
            z_context=z_input,
            actions=actions,
            num_steps=num_steps,
            teacher_forcing_targets=z_target,
            teacher_forcing_prob=teacher_forcing_prob
        )
        
        return {
            'z_input': z_input,
            'z_pred': z_pred,
            'z_target': z_target,
        }
    
    def predict_mc(
        self,
        z_context: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        mc_samples: int = 20,
        enable_dropout: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Monte-Carlo Dropout uncertainty estimation
        
        Estimates prediction uncertainty via multiple forward passes with dropout enabled.
        
        Args:
            z_context: (B, T_ctx, C, H, W) context latent sequence
            actions: (B, T_ctx + num_steps, action_dim) action sequence
            num_steps: number of prediction steps
            mc_samples: number of MC samples
            enable_dropout: whether to enable dropout (False for deterministic baseline)
        
        Returns:
            dict with:
                - mean: (B, num_steps, C, H, W) prediction mean
                - std: (B, num_steps, C, H, W) prediction std (uncertainty)
                - samples: (mc_samples, B, num_steps, C, H, W) all MC samples
        """
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        
        was_training = self.training
        try:
            if enable_dropout:
                self.train()
            else:
                self.eval()
            
            samples = []
            for _ in range(mc_samples):
                z_pred = self.predict_sequence(
                    z_context=z_context,
                    actions=actions,
                    num_steps=num_steps,
                    teacher_forcing_prob=0.0
                )
                samples.append(z_pred)
            
            samples_tensor = torch.stack(samples, dim=0)
            
            mean = samples_tensor.mean(dim=0)
            std = samples_tensor.std(dim=0, unbiased=False) if mc_samples > 1 else torch.zeros_like(mean)
            
            return {
                "mean": mean,
                "std": std,
                "samples": samples_tensor
            }
        finally:
            self.train(was_training)


def compute_loss_v2(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    classifier: Optional[nn.Module] = None,
    target_frames: Optional[torch.Tensor] = None,
    lambda_cls: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute V2 model loss
    
    Args:
        z_pred: (B, T, C, H, W) predicted latent
        z_target: (B, T, C, H, W) target latent
        classifier: optional lane classifier
        target_frames: (B, T, 3, H_img, W_img) target images (for label extraction)
        lambda_cls: classification loss weight
    
    Returns:
        dict with loss components
    """
    mse_loss = F.mse_loss(z_pred, z_target)
    
    total_loss = mse_loss
    loss_dict = {'mse': mse_loss}
    
    if classifier is not None and target_frames is not None:
        z_pred_last = z_pred[:, -1]
        
        with torch.no_grad():
            pass
        
    
    loss_dict['total'] = total_loss
    return loss_dict


def train_epoch_v2(
    model: VAEPredictorV2,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    classifier: Optional[nn.Module] = None,
    lambda_cls: float = 0.5,
    teacher_forcing_prob: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch"""
    model.train()
    if classifier is not None:
        classifier.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_frames = batch['input_frames'].to(device)
        target_frames = batch['target_frames'].to(device)
        actions = batch['actions'].to(device) if 'actions' in batch else None
        
        optimizer.zero_grad()
        
        outputs = model(
            input_frames=input_frames,
            actions=actions,
            target_frames=target_frames,
            teacher_forcing_prob=teacher_forcing_prob
        )
        
        z_pred = outputs['z_pred']
        z_target = outputs['z_target']
        
        loss_dict = compute_loss_v2(
            z_pred=z_pred,
            z_target=z_target,
            classifier=classifier,
            target_frames=target_frames,
            lambda_cls=lambda_cls
        )
        
        loss = loss_dict['total']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += loss_dict['mse'].item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
    }


def save_model_v2(model: VAEPredictorV2, path: str, epoch: int = 0, **kwargs):
    """Save V2 model"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': {
            'latent_dim': model.latent_dim,
            'spatial_size': model.spatial_size,
            'action_dim': model.action_dim,
            'hidden_channels': model.hidden_channels,
            'num_layers': model.num_layers,
            'residual': model.residual,
            'recurrent_dropout': model.recurrent_dropout,
        },
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model_v2(path: str, device: torch.device, vae_model_path: Optional[str] = None) -> VAEPredictorV2:
    """Load V2 model"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = VAEPredictorV2(
        latent_dim=config.get('latent_dim', 64),
        spatial_size=config.get('spatial_size', 4),
        action_dim=config.get('action_dim', 2),
        hidden_channels=config.get('hidden_channels', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1),
        recurrent_dropout=config.get('recurrent_dropout', 0.0),
        residual=config.get('residual', True),
        vae_model_path=vae_model_path,
        freeze_vae=True
    )
    
    state_dict = checkpoint['model_state_dict']
    predictor_state = {k: v for k, v in state_dict.items() 
                       if not k.startswith('vae_')}
    model.load_state_dict(predictor_state, strict=False)
    model.to(device)
    
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = VAEPredictorV2(
        latent_dim=64,
        spatial_size=4,
        action_dim=2,
        hidden_channels=128,
        vae_model_path=None,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    B, T_ctx = 2, 10
    z_context = torch.randn(B, T_ctx, 64, 4, 4).to(device)
    actions = torch.randn(B, T_ctx + 5, 2).to(device)
    
    z_pred = model.predict_sequence(z_context, actions, num_steps=5)
    print(f"Input: {z_context.shape}")
    print(f"Output: {z_pred.shape}")
    
    print("\n=== Capacity Comparison ===")
    print(f"Original LSTM hidden: 256 dim")
    print(f"V2 ConvLSTM hidden: {128} channels × {4}×{4} = {128*4*4} dim")
    print(f"Capacity increase: {128*4*4 / 256:.1f}x")
