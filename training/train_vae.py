# Original: vae_recon/train_enhanced.py
#!/usr/bin/env python3.11
"""
Enhanced training script for VAE reconstruction
- Perceptual Loss (VGG features) for better detail preservation
- Deeper architecture with larger capacity
- Larger latent dimension (512)
- Stronger reconstruction emphasis
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import json

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from vae_recon.vae_model_enhanced import EnhancedVAE, ImageDataset, enhanced_vae_loss, PerceptualLoss, save_model, load_model
    from vae_recon.vae_model_64x64 import SimpleVAE64x64, ImageDataset64x64, save_model_64x64, load_model_64x64
except Exception:
    from vae_model_enhanced import EnhancedVAE, ImageDataset, enhanced_vae_loss, PerceptualLoss, save_model, load_model
    from vae_model_64x64 import SimpleVAE64x64, ImageDataset64x64, save_model_64x64, load_model_64x64


def get_beta_schedule(epoch, total_epochs, anneal_type='linear', beta_max=1.0, warmup_epochs=5, beta_start=0.1):
    """
    Improved KL annealing schedule
    - Starts from beta_start (not 0) for faster stabilization
    - Linear increase to beta_max during warmup
    """
    if anneal_type == 'linear':
        if epoch < warmup_epochs:
            progress = (epoch + 1) / warmup_epochs
            return beta_start + (beta_max - beta_start) * progress
        else:
            return beta_max
    elif anneal_type == 'sigmoid':
        progress = min(1.0, epoch / warmup_epochs)
        sigmoid_val = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
        return beta_start + (beta_max - beta_start) * sigmoid_val
    else:
        return beta_max


def train_epoch(model, dataloader, optimizer, device, beta=1.0, free_bits=2.0, 
                use_perceptual=True, perceptual_weight=0.1, perceptual_loss_fn=None, debug=False):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_perceptual = 0.0
    num_batches = 0
    
    for batch_idx, x in enumerate(dataloader):
        x = x.to(device)
        
        if batch_idx == 0 and debug:
            print(f"Input shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
        
        x_recon, mu, logvar, z = model(x)
        
        if batch_idx == 0 and debug:
            print(f"Output shape: {x_recon.shape}, dtype: {x_recon.dtype}, range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
        
        loss, recon_loss, kl_loss, perceptual_loss = enhanced_vae_loss(
            x_recon, x, mu, logvar, 
            beta=beta, 
            free_bits=free_bits,
            use_perceptual=use_perceptual,
            perceptual_weight=perceptual_weight,
            l1_weight=0.7,
            mse_weight=0.3,
            perceptual_loss_fn=perceptual_loss_fn,
            debug=debug and (batch_idx == 0)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_perceptual += perceptual_loss.item() if use_perceptual else 0.0
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            perceptual_val = perceptual_loss.item() if use_perceptual else 0.0
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.6f}, "
                  f"Recon: {recon_loss.item():.6f}, "
                  f"Perceptual: {perceptual_val:.6f}, "
                  f"KL: {kl_loss.item():.6f}, "
                  f"Beta: {beta:.4f}")
        
        if batch_idx == 0 and debug:
            print(f"  [Debug] First batch - Recon: {recon_loss.item():.6f}, "
                  f"Perceptual: {perceptual_loss.item():.6f if use_perceptual else 0.0:.6f}, "
                  f"KL: {kl_loss.item():.6f}")
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches,
        'perceptual_loss': total_perceptual / num_batches if use_perceptual else 0.0,
    }


def validate(model, dataloader, device, beta=1.0, free_bits=2.0, 
             use_perceptual=True, perceptual_weight=0.1, perceptual_loss_fn=None, debug=False):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_perceptual = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            
            x_recon, mu, logvar, z = model(x)
            
            loss, recon_loss, kl_loss, perceptual_loss = enhanced_vae_loss(
                x_recon, x, mu, logvar, 
                beta=beta, 
                free_bits=free_bits,
                use_perceptual=use_perceptual,
                perceptual_weight=perceptual_weight,
                l1_weight=0.7,
                mse_weight=0.3,
                perceptual_loss_fn=perceptual_loss_fn,
                debug=debug and (batch_idx == 0)
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_perceptual += perceptual_loss.item() if use_perceptual else 0.0
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches,
        'perceptual_loss': total_perceptual / num_batches if use_perceptual else 0.0,
    }


def save_samples(model, dataloader, device, save_dir, epoch, num_samples=8):
    """Save reconstruction samples"""
    if not HAS_MATPLOTLIB:
        return
    
    model.eval()
    with torch.no_grad():
        x = next(iter(dataloader))[:num_samples].to(device)
        x_recon, mu, logvar, z = model(x)
        
        if x_recon.min() < 0:
            x_recon_scaled = (x_recon + 1.0) / 2.0
            x_recon_scaled = torch.clamp(x_recon_scaled, 0, 1)
        else:
            x_recon_scaled = torch.clamp(x_recon, 0, 1)
        
        x_np = x.cpu().numpy()
        if x_np.min() < 0:
            x_np = (x_np + 1.0) / 2.0
        x_np = np.clip(x_np, 0, 1)
        
        x_recon_np = x_recon_scaled.cpu().numpy()
        x_recon_np = np.clip(x_recon_np, 0, 1)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            orig = x_np[i].transpose(1, 2, 0)
            axes[0, i].imshow(orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            recon = x_recon_np[i].transpose(1, 2, 0)
            axes[1, i].imshow(recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction samples to {save_dir}/reconstruction_epoch_{epoch:03d}.png")


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced VAE for image reconstruction')
    parser.add_argument('--data_dir', type=str, default='../npz_transfer',
                       help='Directory containing NPZ files')
    parser.add_argument('--npz_files', nargs='+', default=['traj1.npz', 'traj2.npz'],
                       help='NPZ files to use for training')
    parser.add_argument('--image_size', type=int, default=224, choices=[64, 224],
                       help='Image size: 64 for faster training, 224 for higher quality')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (reduced for larger model)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (reduced for more stable training)')
    parser.add_argument('--beta_max', type=float, default=0.0,
                       help='Maximum beta parameter for KL loss weighting (set to 0 to disable KL completely)')
    parser.add_argument('--beta_warmup', type=int, default=40,
                       help='Number of epochs for KL annealing warmup (very slow for stability)')
    parser.add_argument('--beta_start', type=float, default=0.0,
                       help='Starting beta value (use 0.0 to start fully reconstruction-focused)')
    parser.add_argument('--kl_freeze_epochs', type=int, default=10,
                       help='Number of initial epochs to keep beta forced to 0 (freeze KL, increased for better reconstruction)')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'sigmoid'],
                       help='KL annealing schedule type')
    parser.add_argument('--free_bits', type=float, default=0.0,
                       help='Free bits in nats per latent element (set 0 to disable)')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent space channels (convolutional latent, 64 for 64x64 images)')
    parser.add_argument('--no_skip_connections', action='store_false', dest='use_skip_connections',
                       help='Disable U-Net style skip connections (enabled by default)')
    # Note: action='store_false' means if --no_skip_connections is used, use_skip_connections becomes False
    parser.add_argument('--enable_perceptual', action='store_true',
                       help='Enable perceptual loss (disabled by default)')
    parser.set_defaults(use_perceptual=False)
    parser.add_argument('--perceptual_weight', type=float, default=0.01,
                       help='Weight for perceptual loss (reduced due to larger scale)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_enhanced',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs (to reduce disk usage)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    if not hasattr(args, 'use_skip_connections') or args.use_skip_connections is None:
        args.use_skip_connections = True
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, data_dir))
    npz_paths = [os.path.join(data_dir, f) for f in args.npz_files]
    print(f"Loading data from: {npz_paths}")
    
    print("Creating dataset...")
    if args.image_size == 64:
        npz_paths_64 = []
        for path in npz_paths:
            base = os.path.splitext(path)[0]
            path_64 = f"{base}_64x64.npz"
            if os.path.exists(path_64):
                npz_paths_64.append(path_64)
            else:
                print(f"Warning: {path_64} not found, trying original {path}")
                npz_paths_64.append(path)
        dataset = ImageDataset64x64(npz_paths=npz_paths_64, normalize=True)
        expected_shape = (3, 64, 64)
    else:
        dataset = ImageDataset(npz_paths=npz_paths, normalize=True)
        expected_shape = (3, 224, 224)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}, range: [{sample.min():.3f}, {sample.max():.3f}]")
        if sample.shape != expected_shape:
            raise ValueError(f"Dataset sample shape mismatch: {sample.shape}, expected {expected_shape}")
    
    total_size = len(dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")
    
    if args.image_size == 64:
        print("Using SimpleVAE64x64 model")
        model = SimpleVAE64x64(
            latent_dim=args.latent_dim,
            channels=3,
            use_skip_connections=False
        ).to(device)
        if args.use_perceptual:
            print("Warning: Perceptual loss disabled for 64x64 images (VGG expects 224x224)")
            args.use_perceptual = False
    else:
        print("Using EnhancedVAE model (224x224)")
        model = EnhancedVAE(
            latent_dim=args.latent_dim,
            image_size=224,
            channels=3,
            use_skip_connections=args.use_skip_connections
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    perceptual_loss_fn = None
    if args.use_perceptual:
        print("Initializing Perceptual Loss (VGG features)...")
        try:
            perceptual_loss_fn = PerceptualLoss().to(device)
            perceptual_loss_fn.eval()
            print("✓ Perceptual Loss initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize Perceptual Loss: {e}")
            print("Falling back to pixel-level loss only")
            args.use_perceptual = False
    
    initial_lr = args.lr
    if args.image_size == 64 and args.lr == 1e-4:
        initial_lr = 1e-3
        print(f"Using higher initial LR for 64x64: {initial_lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    if args.image_size == 64:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40], gamma=0.5
        )
        scheduler_type = 'multistep'
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        scheduler_type = 'plateau'
    
    start_epoch = 0
    best_loss = float('inf')
    train_history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_perceptual_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_perceptual_loss': [],
        'val_kl_loss': [],
        'beta': [],
    }
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        train_history = checkpoint.get('train_history', train_history)
        print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
    
    print(f"\nStarting VAE training for {args.epochs} epochs...")
    print("=" * 70)
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Beta: {args.beta_start:.1f} -> {args.beta_max:.1f} (warmup: {args.beta_warmup} epochs)")
    print(f"Free bits: {args.free_bits}, Latent channels: {args.latent_dim} (convolutional)")
    if args.image_size == 224:
        print(f"Skip connections: {args.use_skip_connections}")
    print(f"Use Perceptual: {args.use_perceptual}, Perceptual weight: {args.perceptual_weight}")
    print(f"Loss weights: L1=0.7, MSE=0.3")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        current_beta = get_beta_schedule(
            epoch, args.epochs, 
            anneal_type=args.beta_schedule,
            beta_max=args.beta_max,
            warmup_epochs=args.beta_warmup,
            beta_start=args.beta_start
        )
        if epoch < args.kl_freeze_epochs:
            current_beta = 0.0
            if epoch == 0:
                print(f"KL frozen for first {args.kl_freeze_epochs} epochs (beta forced to 0).")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, 
            beta=current_beta, 
            free_bits=args.free_bits,
            use_perceptual=args.use_perceptual,
            perceptual_weight=args.perceptual_weight,
            perceptual_loss_fn=perceptual_loss_fn,
            debug=args.debug and (epoch == 0)
        )
        
        val_metrics = validate(
            model, val_loader, device, 
            beta=current_beta, 
            free_bits=args.free_bits,
            use_perceptual=args.use_perceptual,
            perceptual_weight=args.perceptual_weight,
            perceptual_loss_fn=perceptual_loss_fn,
            debug=args.debug and (epoch == 0)
        )
        
        if scheduler_type == 'multistep':
            scheduler.step()
        else:
            scheduler.step(val_metrics['loss'])
        
        train_history['train_loss'].append(train_metrics['loss'])
        train_history['train_recon_loss'].append(train_metrics['recon_loss'])
        train_history['train_perceptual_loss'].append(train_metrics['perceptual_loss'])
        train_history['train_kl_loss'].append(train_metrics['kl_loss'])
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_recon_loss'].append(val_metrics['recon_loss'])
        train_history['val_perceptual_loss'].append(val_metrics['perceptual_loss'])
        train_history['val_kl_loss'].append(val_metrics['kl_loss'])
        train_history['beta'].append(current_beta)
        
        print(f"Train Loss: {train_metrics['loss']:.6f} "
              f"(Recon: {train_metrics['recon_loss']:.6f}, "
              f"Perceptual: {train_metrics['perceptual_loss']:.6f}, "
              f"KL: {train_metrics['kl_loss']:.6f})")
        print(f"Val Loss: {val_metrics['loss']:.6f} "
              f"(Recon: {val_metrics['recon_loss']:.6f}, "
              f"Perceptual: {val_metrics['perceptual_loss']:.6f}, "
              f"KL: {val_metrics['kl_loss']:.6f})")
        print(f"Beta: {current_beta:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        is_best = val_metrics['loss'] < best_loss
        if is_best:
            best_loss = val_metrics['loss']
            print(f"New best validation loss: {best_loss:.6f}")
        
        save_checkpoint = (epoch + 1) % args.save_freq == 0 or is_best or (epoch + 1) == args.epochs
        
        if save_checkpoint:
            try:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'train_history': train_history,
                    'args': vars(args)
                }
                
                if is_best:
                    best_path = os.path.join(args.save_dir, 'best_model.pt')
                    torch.save(checkpoint, best_path)
                    print(f"Saved best model to {best_path}")
                
                if (epoch + 1) % args.save_freq == 0:
                    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                
                if (epoch + 1) % args.save_freq == 0 or is_best:
                    if args.image_size == 64:
                        model_path = os.path.join(args.save_dir, f'vae_64x64_epoch_{epoch + 1}.pt')
                        save_func = save_model_64x64
                    else:
                        model_path = os.path.join(args.save_dir, f'vae_enhanced_epoch_{epoch + 1}.pt')
                        save_func = save_model
                    
                    try:
                        save_func(model, model_path)
                    except Exception as e:
                        print(f"Warning: Failed to save model state: {e}")
                        try:
                            torch.save(model.state_dict(), model_path.replace('.pt', '_state_dict.pt'))
                            print(f"Saved model state_dict to {model_path.replace('.pt', '_state_dict.pt')}")
                        except Exception as e2:
                            print(f"Error: Failed to save model: {e2}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
        
        if (epoch + 1) % args.save_freq == 0 or (is_best and (epoch + 1) % 2 == 0):
            try:
                save_samples(model, val_loader, device, args.save_dir, epoch + 1)
            except Exception as e:
                print(f"Warning: Failed to save samples: {e}")
                if "No space left" in str(e) or "Errno 28" in str(e):
                    print("Disk full - skipping sample saving for remaining epochs")
                    args.save_freq = 999999
    
    final_model_path = os.path.join(args.save_dir, 'vae_enhanced_final.pt')
    try:
        save_model(model, final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Warning: Failed to save final model: {e}")
        try:
            torch.save(model.state_dict(), final_model_path.replace('.pt', '_state_dict.pt'))
            print(f"Saved final model state_dict to {final_model_path.replace('.pt', '_state_dict.pt')}")
        except Exception as e2:
            print(f"Error: Failed to save final model: {e2}")
        print("\nTraining completed!")
    
    if HAS_MATPLOTLIB:
        try:
            plot_training_history(train_history, args.save_dir)
        except Exception as e:
            print(f"Warning: Failed to save training history plot: {e}")
            if "No space left" in str(e) or "Errno 28" in str(e):
                print("Disk full - skipping training history plot")
    
    try:
        config_path = os.path.join(args.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Failed to save training config: {e}")
        if "No space left" in str(e) or "Errno 28" in str(e):
            print("Disk full - skipping training config save")


def plot_training_history(history, save_dir):
    """Plot and save training history"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_recon_loss'], label='Train Recon Loss')
    axes[0, 1].plot(history['val_recon_loss'], label='Val Recon Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    if 'train_perceptual_loss' in history:
        axes[1, 0].plot(history['train_perceptual_loss'], label='Train Perceptual Loss')
        axes[1, 0].plot(history['val_perceptual_loss'], label='Val Perceptual Loss')
        axes[1, 0].set_title('Perceptual Loss (VGG)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['train_kl_loss'], label='Train KL Loss')
    axes[1, 1].plot(history['val_kl_loss'], label='Val KL Loss')
    axes[1, 1].set_title('KL Divergence Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {plot_path}")


if __name__ == "__main__":
    main()

