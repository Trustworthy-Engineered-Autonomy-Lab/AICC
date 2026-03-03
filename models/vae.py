# Original: vae_recon/vae_model_64x64_v2.py
"""
VAE model for 64x64 images - V2
Uses Upsample + Conv instead of ConvTranspose2d to avoid checkerboard artifacts
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class SimpleVAE64x64_V2(nn.Module):
    """
    Improved VAE for 64x64 images
    - Uses Upsample + Conv instead of ConvTranspose2d (avoids checkerboard artifacts)
    - Increased decoder capacity
    """
    
    def __init__(self, latent_dim: int = 64, channels: int = 3):
        super(SimpleVAE64x64_V2, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.to_mu = nn.Conv2d(512, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv2d(512, latent_dim, kernel_size=1)
        
        self.decoder_start = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar, []
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, skip_features: list = None) -> torch.Tensor:
        h = self.decoder_start(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return self.output(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


class ImageDataset64x64(Dataset):
    """Dataset for 64x64 images from NPZ files"""
    
    def __init__(self, npz_paths, normalize=True, target_size=64):
        self.images = []
        self.normalize = normalize
        
        for npz_path in npz_paths:
            data = np.load(npz_path, allow_pickle=True)
            
            for key in ['frame', 'frames', 'images', 'image']:
                if key in data:
                    frames = data[key]
                    break
            else:
                print(f"Warning: No image key found in {npz_path}")
                continue
            
            self.images.append(frames)
            print(f"Loaded {npz_path}: {len(frames)} images")
        
        if self.images:
            self.images = np.concatenate(self.images, axis=0)
            print(f"Total images: {len(self.images)}")
        else:
            raise ValueError("No images loaded")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        if self.normalize:
            img = img / 255.0
        return torch.from_numpy(img)


def save_model_v2(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim,
        'channels': model.channels,
        'version': 'v2',
    }, path)


def load_model_v2(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SimpleVAE64x64_V2(
        latent_dim=checkpoint.get('latent_dim', 64),
        channels=checkpoint.get('channels', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
