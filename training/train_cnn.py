# Original: lane_classifier/training/train.py
"""
Training script for Lane CNN Binary Classifier
"""

import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from lane_classifier.cnn_model import get_model
from lane_classifier.dataset_visual import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 3:
            images, labels, ctes = batch_data
        else:
            images, labels = batch_data
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} Acc: {acc:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0, 0]
    class_total = [0, 0]
    
    all_predictions = []
    all_labels = []
    all_ctes = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:
                images, labels, ctes = batch_data
            else:
                images, labels = batch_data
                ctes = None
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if ctes is not None:
                all_ctes.extend(ctes.numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    left_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    right_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'left_acc': left_acc,
        'right_acc': right_acc,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'ctes': np.array(all_ctes)
    }
    
    return metrics


def plot_confusion_matrix(predictions, labels, save_path):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_left_acc'], label='Left')
    axes[1, 0].plot(history['val_right_acc'], label='Right')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Per-Class Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Lane CNN Classifier')
    
    parser.add_argument('--data_dir', type=str, default='npz_data',
                        help='Directory containing NPZ files')
    parser.add_argument('--npz_files', nargs='+', 
                        default=['traj1_64x64.npz', 'traj2_64x64.npz'],
                        help='NPZ files to load')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--balance_classes', action='store_true', default=True,
                        help='Balance left/right classes')
    
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'lightweight'],
                        help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='Use class weights in loss function')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='Learning rate scheduler')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    parser.add_argument('--save_dir', type=str, default='lane_classifier/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='lane_classifier/logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("Lane CNN Binary Classifier Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', data_dir))
    
    npz_paths = [os.path.join(data_dir, f) for f in args.npz_files]
    print(f"\nLoading data from: {npz_paths}")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        npz_paths=npz_paths,
        batch_size=args.batch_size,
        val_split=args.val_split,
        target_size=64,
        balance_classes=args.balance_classes,
        num_workers=args.num_workers
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    print(f"\nCreating {args.model_type} model...")
    model = get_model(model_type=args.model_type, dropout_rate=args.dropout)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.use_class_weights:
        class_weights = class_weights.to(device)
        print(f"\nUsing class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                          weight_decay=args.weight_decay)
    
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
    else:
        scheduler = None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_left_acc': [],
        'val_right_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print("\nValidating...")
        val_metrics = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_left_acc'].append(val_metrics['left_acc'])
        history['val_right_acc'].append(val_metrics['right_acc'])
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}  Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Left Acc: {val_metrics['left_acc']:.2f}%  Right Acc: {val_metrics['right_acc']:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val_left', val_metrics['left_acc'], epoch)
        writer.add_scalar('Accuracy/val_right', val_metrics['right_acc'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, best_model_path)
            print(f"  [*] Saved best model (acc: {best_val_acc:.2f}%)")
        
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, checkpoint_path)
            print(f"  [*] Saved checkpoint")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    final_model_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'args': vars(args)
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    curves_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_curves(history, curves_path)
    
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(val_metrics['predictions'], val_metrics['labels'], cm_path)
    
    writer.close()
    
    print(f"\nTensorboard logs saved to: {log_dir}")
    print(f"To view: tensorboard --logdir={args.log_dir}")
    
    print("\n[*] Training complete!")


if __name__ == '__main__':
    main()
