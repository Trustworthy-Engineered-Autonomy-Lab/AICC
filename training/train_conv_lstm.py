# Original: predictor/core/train_predictor_v2.py
#!/usr/bin/env python3
"""
Train VAE Predictor V2 (ConvLSTM version)

Key improvements:
1. ConvLSTM preserves spatial structure (no latent flattening)
2. Larger hidden capacity: 128 channels x 4x4 = 2048 dim (was 256 dim)
3. Action injected via spatial feature maps
4. Gated residual connections

Performance optimizations:
- cuDNN benchmark auto-selects fastest convolution algorithm
- Avoids double priming in open-loop (reuses primed hidden state)
- Pre-allocates tensors to reduce memory allocation overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime

torch.backends.cudnn.benchmark = True

from predictor.core.vae_predictor import TrajectoryDataset
from predictor.core.vae_predictor_v2 import (
    VAEPredictorV2, 
    train_epoch_v2, 
    save_model_v2,
    compute_loss_v2
)


def main():
    parser = argparse.ArgumentParser(description='Train VAE Predictor V2 (ConvLSTM)')
    
    parser.add_argument('--data_dir', type=str, default='npz_data')
    parser.add_argument('--npz_files', nargs='+', default=['traj1_64x64.npz', 'traj2_64x64.npz'])
    parser.add_argument('--sequence_length', type=int, default=220,
                       help='Total sequence length (context + target + buffer)')
    parser.add_argument('--input_length', type=int, default=100,
                       help='Context length (frames to prime LSTM)')
    parser.add_argument('--target_length', type=int, default=100,
                       help='Number of frames to predict')
    parser.add_argument('--target_offset', type=int, default=1)
    
    parser.add_argument('--vae_model_path', type=str, default='vae_recon/finetuned_ctx100/best_model.pt')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='ConvLSTM hidden channels (effective capacity = channels x 4 x 4)')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate between layers (increased from 0.1 for better regularization)')
    parser.add_argument('--recurrent_dropout', type=float, default=0.15,
                       help='Recurrent dropout rate (variational, applied to hidden state h)')
    parser.add_argument('--residual', action='store_true', default=True)
    
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (increased from 1e-5 for better regularization)')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers (increase if CPU has cores)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training (faster + less memory)')
    parser.add_argument('--compile', action='store_true', default=False,
                       help='Use torch.compile (PyTorch 2.0+, can be 20-50%% faster)')
    
    parser.add_argument('--scheduled_sampling', action='store_true', default=True)
    parser.add_argument('--ss_start_prob', type=float, default=1.0)
    parser.add_argument('--ss_end_prob', type=float, default=0.0)
    parser.add_argument('--ss_decay_epochs', type=int, default=40)
    
    parser.add_argument('--open_loop_curriculum', action='store_true', default=True)
    parser.add_argument('--open_loop_schedule', type=str, default='0:20,10:40,20:60,30:100',
                       help='Format: epoch:steps,epoch:steps,... (aggressive schedule)')
    parser.add_argument('--open_loop_weight', type=float, default=0.5)
    
    parser.add_argument('--curriculum_learning', action='store_true', default=False,
                       help='Enable curriculum learning for prediction steps')
    parser.add_argument('--curriculum_schedule', type=str, default='0:20,10:40,20:60,30:100',
                       help='Format: epoch:max_steps - aggressive schedule')
    
    parser.add_argument('--multi_scale_loss', action='store_true', default=False,
                       help='Use multi-scale prediction loss (short+medium+long term)')
    parser.add_argument('--multi_scale_weights', type=str, default='0.3,0.3,0.4',
                       help='Weights for 10-step, 30-step, and full prediction loss')
    
    parser.add_argument('--diversity_loss', action='store_true', default=False,
                       help='Add diversity loss to prevent prediction collapse')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                       help='Weight for diversity loss')
    
    parser.add_argument('--velocity_loss', action='store_true', default=False,
                       help='Match frame-to-frame velocity (direction + magnitude)')
    parser.add_argument('--velocity_weight', type=float, default=1.0,
                       help='Weight for velocity loss')
    
    parser.add_argument('--temporal_weight', action='store_true', default=False,
                       help='Apply increasing weight to later timesteps')
    parser.add_argument('--temporal_weight_power', type=float, default=1.0,
                       help='Power for temporal weight (1.0=linear, 2.0=quadratic)')
    
    parser.add_argument('--noise_injection', action='store_true', default=False,
                       help='Add noise to latent during training for robustness')
    parser.add_argument('--noise_schedule', type=str, default='0:0.1,30:0.05,60:0.02',
                       help='Noise schedule: epoch:std pairs (decays over training)')
    
    parser.add_argument('--recon_loss', action='store_true', default=False,
                       help='Add image reconstruction loss (decode z_pred and compare)')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    
    parser.add_argument('--class_loss', action='store_true', default=False,
                       help='Add lane classification loss on predicted latent')
    parser.add_argument('--class_weight', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--classifier_path', type=str, default='lane_classifier/checkpoints/best_model.pt',
                       help='Path to trained LatentClassifier')
    
    parser.add_argument('--save_dir', type=str, default='predictor/checkpoints_v2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("VAE Predictor V2 (ConvLSTM) Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"  -> Actual capacity: {args.hidden_channels} × 4 × 4 = {args.hidden_channels * 16} dim")
    print(f"  -> vs Original LSTM: 256 dim")
    print(f"  -> Capacity increase: {args.hidden_channels * 16 / 256:.1f}x")
    
    print(f"\n[1/3] Loading data...")
    npz_paths = [os.path.join(args.data_dir, f) for f in args.npz_files]
    
    full_dataset = TrajectoryDataset(
        npz_paths=npz_paths,
        sequence_length=args.sequence_length,
        image_size=64,
        normalize=True,
        input_length=args.input_length,
        target_length=args.target_length,
        target_offset=args.target_offset
    )
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0,
        'prefetch_factor': 2 if args.num_workers > 0 else None,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val:   {len(val_dataset)} sequences")
    
    print(f"\n[2/3] Creating model...")
    model = VAEPredictorV2(
        latent_dim=64,
        spatial_size=4,
        action_dim=2,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        residual=args.residual,
        vae_model_path=args.vae_model_path,
        freeze_vae=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total params: {total_params:,}")
    print(f"Trainable (predictor): {trainable_params:,}")
    print(f"Frozen (VAE): {frozen_params:,}")
    
    start_epoch = 1
    best_val_loss = float('inf')
    train_history = {'loss': [], 'mse': [], 'val_loss': [], 'val_mse': []}
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        state_dict = ckpt.get('model_state_dict', ckpt)
        predictor_state = {k: v for k, v in state_dict.items() if not k.startswith('vae_')}
        model.load_state_dict(predictor_state, strict=False)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        train_history = ckpt.get('train_history', train_history)
        print(f"  -> Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.6f}")
    
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("Using AMP (mixed precision) training")
    
    print(f"\n[3/3] Setting up training...")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    open_loop_schedule = []
    if args.open_loop_curriculum and args.open_loop_schedule:
        for item in args.open_loop_schedule.split(','):
            epoch, steps = map(int, item.strip().split(':'))
            open_loop_schedule.append((epoch, steps))
        print(f"Open-loop curriculum: {open_loop_schedule}")
    
    curriculum_schedule = []
    if args.curriculum_learning and args.curriculum_schedule:
        for item in args.curriculum_schedule.split(','):
            epoch, max_steps = map(int, item.strip().split(':'))
            curriculum_schedule.append((epoch, max_steps))
        print(f"Curriculum learning: {curriculum_schedule}")
    
    multi_scale_weights = [float(w) for w in args.multi_scale_weights.split(',')]
    if args.multi_scale_loss:
        print(f"Multi-scale loss weights (10/30/full): {multi_scale_weights}")
    
    if args.velocity_loss:
        print(f"Velocity loss enabled (weight={args.velocity_weight})")
    
    if args.diversity_loss:
        print(f"Diversity loss enabled (weight={args.diversity_weight})")
    
    if args.temporal_weight:
        print(f"Temporal weight enabled (power={args.temporal_weight_power})")
    
    noise_schedule = []
    if args.noise_injection and args.noise_schedule:
        for item in args.noise_schedule.split(','):
            epoch, std = item.strip().split(':')
            noise_schedule.append((int(epoch), float(std)))
        print(f"Noise injection enabled: {noise_schedule}")
    
    latent_classifier = None
    if args.class_loss:
        try:
            from lane_classifier.latent_classifier import LatentClassifier
            latent_classifier = LatentClassifier(latent_dim=64, latent_spatial_size=4)
            if os.path.exists(args.classifier_path):
                ckpt = torch.load(args.classifier_path, map_location=device, weights_only=False)
                latent_classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
                latent_classifier = latent_classifier.to(device)
                latent_classifier.eval()
                for p in latent_classifier.parameters():
                    p.requires_grad = False
                print(f"Loaded LatentClassifier from {args.classifier_path}")
            else:
                print(f"Warning: Classifier not found at {args.classifier_path}, disabling class_loss")
                args.class_loss = False
        except Exception as e:
            print(f"Warning: Could not load LatentClassifier: {e}")
            args.class_loss = False
    
    vae_decoder = None
    if args.recon_loss:
        try:
            from vae_recon.vae_model_64x64 import SimpleVAE64x64
            vae_full = SimpleVAE64x64(latent_dim=64)
            if os.path.exists(args.vae_model_path):
                ckpt = torch.load(args.vae_model_path, map_location=device, weights_only=False)
                vae_full.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
                vae_full = vae_full.to(device)
                vae_full.eval()
                for p in vae_full.parameters():
                    p.requires_grad = False
                vae_decoder = vae_full
                print(f"Loaded VAE decoder for reconstruction loss")
            else:
                print(f"Warning: VAE not found, disabling recon_loss")
                args.recon_loss = False
        except Exception as e:
            print(f"Warning: Could not load VAE decoder: {e}")
            args.recon_loss = False
    
    print(f"\n{'=' * 70}")
    print(f"Starting Training (from epoch {start_epoch})")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        if args.scheduled_sampling:
            decay_epochs = args.ss_decay_epochs or args.epochs
            progress = min(1.0, epoch / decay_epochs)
            tf_prob = args.ss_start_prob - progress * (args.ss_start_prob - args.ss_end_prob)
        else:
            tf_prob = 1.0
        
        noise_std = 0.0
        if args.noise_injection and noise_schedule:
            for e, std in noise_schedule:
                if epoch >= e:
                    noise_std = std
        
        open_loop_steps = 0
        for e, steps in open_loop_schedule:
            if epoch >= e:
                open_loop_steps = steps
        
        curriculum_max_steps = None
        if args.curriculum_learning and curriculum_schedule:
            for e, max_steps in curriculum_schedule:
                if epoch >= e:
                    curriculum_max_steps = max_steps
        
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_frames = batch['input_frames'].to(device, non_blocking=True)
            target_frames = batch['target_frames'].to(device, non_blocking=True)
            actions = batch['actions'].to(device, non_blocking=True) if 'actions' in batch else None
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                B, T_in = input_frames.shape[:2]
                T_tgt = target_frames.shape[1]
                
                with torch.no_grad():
                    in_flat = input_frames.reshape(B * T_in, *input_frames.shape[2:])
                    mu_in, _ = model.encode(in_flat)
                    z_input = mu_in.view(B, T_in, *mu_in.shape[1:])
                    
                    tgt_flat = target_frames.reshape(B * T_tgt, *target_frames.shape[2:])
                    mu_tgt, _ = model.encode(tgt_flat)
                    z_target = mu_tgt.view(B, T_tgt, *mu_tgt.shape[1:])
                
                num_predict_full = T_tgt - 1 if args.target_offset == 1 else T_tgt
                
                if curriculum_max_steps is not None:
                    num_predict = min(curriculum_max_steps, num_predict_full)
                else:
                    num_predict = num_predict_full
                
                need_open_loop = open_loop_steps > 0 and args.open_loop_weight > 0
                z_pred, primed_hidden = model.predict_sequence(
                    z_context=z_input,
                    actions=actions,
                    num_steps=num_predict,
                    teacher_forcing_targets=z_target[:, 1:1+num_predict] if args.target_offset == 1 else z_target[:, :num_predict],
                    teacher_forcing_prob=tf_prob,
                    return_primed_hidden=True,
                    noise_std=noise_std,
                )
                
                z_target_aligned = z_target[:, 1:1+num_predict] if args.target_offset == 1 else z_target[:, :num_predict]
                
                if args.temporal_weight:
                    t_weights = torch.linspace(1.0, 2.0, num_predict, device=device)
                    t_weights = t_weights ** args.temporal_weight_power
                    t_weights = t_weights / t_weights.mean()
                    
                    mse_per_step = ((z_pred - z_target_aligned) ** 2).mean(dim=(2, 3, 4))
                    mse_loss = (mse_per_step * t_weights.unsqueeze(0)).mean()
                elif args.multi_scale_loss and num_predict >= 30:
                    w_short, w_med, w_full = multi_scale_weights
                    
                    n_short = min(10, num_predict)
                    loss_short = F.mse_loss(z_pred[:, :n_short], z_target_aligned[:, :n_short])
                    
                    n_med = min(30, num_predict)
                    loss_med = F.mse_loss(z_pred[:, :n_med], z_target_aligned[:, :n_med])
                    
                    loss_full = F.mse_loss(z_pred, z_target_aligned)
                    
                    mse_loss = w_short * loss_short + w_med * loss_med + w_full * loss_full
                else:
                    mse_loss = F.mse_loss(z_pred, z_target_aligned)
                
                velocity_loss = torch.tensor(0.0, device=device)
                if args.velocity_loss and num_predict > 1:
                    pred_velocity = z_pred[:, 1:] - z_pred[:, :-1]
                    target_velocity = z_target_aligned[:, 1:] - z_target_aligned[:, :-1]
                    
                    velocity_loss = F.mse_loss(pred_velocity, target_velocity)
                
                diversity_loss = torch.tensor(0.0, device=device)
                if args.diversity_loss and num_predict > 1:
                    pred_diff = (z_pred[:, 1:] - z_pred[:, :-1]).abs()
                    pred_diff_mean = pred_diff.mean(dim=(2, 3, 4))
                    
                    target_diff = (z_target_aligned[:, 1:] - z_target_aligned[:, :-1]).abs()
                    target_diff_mean = target_diff.mean(dim=(2, 3, 4))
                    
                    diversity_loss = F.relu(target_diff_mean - pred_diff_mean).mean()
                
                if need_open_loop:
                    z_pred_ol = model.predict_sequence(
                        z_context=z_input,
                        actions=actions,
                        num_steps=min(open_loop_steps, num_predict),
                        teacher_forcing_prob=0.0,
                        primed_hidden=primed_hidden,
                        noise_std=noise_std,
                    )
                    z_target_ol = z_target_aligned[:, :z_pred_ol.size(1)]
                    ol_loss = F.mse_loss(z_pred_ol, z_target_ol)
                    loss = mse_loss + args.open_loop_weight * ol_loss
                else:
                    loss = mse_loss
                
                if args.velocity_loss:
                    loss = loss + args.velocity_weight * velocity_loss
                
                if args.diversity_loss:
                    loss = loss + args.diversity_weight * diversity_loss
                
                if args.recon_loss and vae_decoder is not None:
                    sample_indices = [0, num_predict // 4, num_predict // 2, 3 * num_predict // 4, num_predict - 1]
                    sample_indices = [i for i in sample_indices if i < num_predict]
                    
                    z_pred_sample = z_pred[:, sample_indices].reshape(-1, 64, 4, 4)
                    z_target_sample = z_target_aligned[:, sample_indices].reshape(-1, 64, 4, 4)
                    
                    with torch.no_grad():
                        img_target = vae_decoder.decode(z_target_sample)
                    img_pred = vae_decoder.decode(z_pred_sample)
                    
                    recon_loss = F.mse_loss(img_pred, img_target)
                    loss = loss + args.recon_weight * recon_loss
                
                if args.class_loss and latent_classifier is not None:
                    sample_indices = [0, num_predict // 4, num_predict // 2, 3 * num_predict // 4, num_predict - 1]
                    sample_indices = [i for i in sample_indices if i < num_predict]
                    
                    z_pred_sample = z_pred[:, sample_indices].reshape(-1, 64, 4, 4)
                    z_target_sample = z_target_aligned[:, sample_indices].reshape(-1, 64, 4, 4)
                    
                    with torch.no_grad():
                        target_logits = latent_classifier(z_target_sample)
                        target_probs = F.softmax(target_logits, dim=-1)
                    
                    pred_logits = latent_classifier(z_pred_sample)
                    
                    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
                    class_loss = F.kl_div(pred_log_probs, target_probs, reduction='batchmean')
                    loss = loss + args.class_weight * class_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_mse += mse_loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_mse /= num_batches
        
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_frames = batch['input_frames'].to(device, non_blocking=True)
                target_frames = batch['target_frames'].to(device, non_blocking=True)
                actions = batch['actions'].to(device, non_blocking=True) if 'actions' in batch else None
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    B, T_in = input_frames.shape[:2]
                    T_tgt = target_frames.shape[1]
                    
                    in_flat = input_frames.reshape(B * T_in, *input_frames.shape[2:])
                    mu_in, _ = model.encode(in_flat)
                    z_input = mu_in.view(B, T_in, *mu_in.shape[1:])
                    
                    tgt_flat = target_frames.reshape(B * T_tgt, *target_frames.shape[2:])
                    mu_tgt, _ = model.encode(tgt_flat)
                    z_target = mu_tgt.view(B, T_tgt, *mu_tgt.shape[1:])
                    
                    num_predict = T_tgt - 1 if args.target_offset == 1 else T_tgt
                    
                    z_pred = model.predict_sequence(
                        z_context=z_input,
                        actions=actions,
                        num_steps=num_predict,
                        teacher_forcing_prob=0.0
                    )
                    
                    z_target_aligned = z_target[:, 1:] if args.target_offset == 1 else z_target
                    mse = F.mse_loss(z_pred, z_target_aligned)
                
                val_loss += mse.item()
                val_mse += mse.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_mse /= val_batches
        
        scheduler.step()
        
        train_history['loss'].append(train_loss)
        train_history['mse'].append(train_mse)
        train_history['val_loss'].append(val_loss)
        train_history['val_mse'].append(val_mse)
        
        lr_current = optimizer.param_groups[0]['lr']
        noise_info = f" | Noise: {noise_std:.3f}" if args.noise_injection else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"TF: {tf_prob:.2f} | OL: {open_loop_steps}{noise_info} | LR: {lr_current:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_v2(
                model, 
                os.path.join(args.save_dir, 'best_model.pt'),
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_history=train_history,
                args=vars(args)
            )
            print(f"  -> New best model saved! (val_loss: {val_loss:.6f})")
        
        if epoch % 10 == 0:
            save_model_v2(
                model,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'),
                epoch=epoch,
                train_history=train_history,
                args=vars(args)
            )
    
    save_model_v2(
        model,
        os.path.join(args.save_dir, 'final_model.pt'),
        epoch=args.epochs,
        train_history=train_history,
        args=vars(args)
    )
    
    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
