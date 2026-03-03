# Original: lane_classifier/data/dataset_visual.py
"""
Visual-based Lane Dataset
Uses image analysis (red lane line position) instead of CTE for label generation
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


def detect_red_line_position(img_uint8):
    """
    Detect the horizontal position of the red lane line
    
    Args:
        img_uint8: (H, W, C) numpy array in uint8 [0, 255]
    
    Returns:
        float: Horizontal center of mass of red pixels (0=left, 1=right)
    """
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    h, w = red_mask.shape
    y_coords, x_coords = np.where(red_mask > 0)
    
    if len(x_coords) == 0:
        return 0.5
    
    red_x_normalized = np.mean(x_coords) / w
    return red_x_normalized


def get_visual_label(red_x):
    """
    Convert red line x position to label
    
    Args:
        red_x: Normalized x position (0=left, 1=right)
    
    Returns:
        int: 0 for Left, 1 for Right
    """
    return 0 if red_x > 0.5 else 1


class LaneDatasetVisual(Dataset):
    """
    Lane position dataset using visual analysis
    - Detects red lane line position in image
    - If red line is on RIGHT side → car is on LEFT
    - If red line is on LEFT side → car is on RIGHT
    """
    
    def __init__(self, npz_files, balance=True, debug=False):
        """
        Args:
            npz_files: List of paths to NPZ files
            balance: Whether to balance classes by undersampling
            debug: Print debug information
        """
        self.balance = balance
        self.debug = debug
        
        all_frames = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            frames = data['frame']
            frames = frames.astype(np.float32) / 255.0
            all_frames.append(frames)
            
            if self.debug:
                print(f"Loaded {npz_file}: {frames.shape[0]} frames")
        
        self.frames = np.concatenate(all_frames, axis=0)
        
        print(f"Total frames loaded: {len(self.frames)}")
        
        print("Analyzing red lane line positions...")
        self.labels = self._generate_visual_labels()
        
        left_count = np.sum(self.labels == 0)
        right_count = np.sum(self.labels == 1)
        print(f"Visual label distribution:")
        print(f"  Left (0): {left_count} ({100*left_count/len(self.labels):.1f}%)")
        print(f"  Right (1): {right_count} ({100*right_count/len(self.labels):.1f}%)")
        
        if balance and left_count != right_count:
            self._balance_classes()
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _detect_red_line_position(self, frame):
        """
        Detect the horizontal position of the red lane line
        
        Args:
            frame: (C, H, W) numpy array in [0, 1]
        
        Returns:
            float: Horizontal center of mass of red pixels (0=left, 1=right)
        """
        img = np.transpose(frame, (1, 2, 0))
        img_uint8 = (img * 255).astype(np.uint8)
        
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        h, w = red_mask.shape
        y_coords, x_coords = np.where(red_mask > 0)
        
        if len(x_coords) == 0:
            return 0.5
        
        x_center = np.mean(x_coords) / w
        
        return x_center
    
    def _generate_visual_labels(self):
        """
        Generate labels based on red line position
        
        Returns:
            np.array: Labels (0=Left, 1=Right)
        """
        labels = []
        red_positions = []
        
        for i, frame in enumerate(self.frames):
            red_x = self._detect_red_line_position(frame)
            red_positions.append(red_x)
            
            label = 0 if red_x > 0.5 else 1
            labels.append(label)
            
            if self.debug and i < 10:
                print(f"Frame {i}: red_x={red_x:.3f} → {'Left' if label==0 else 'Right'}")
        
        red_positions = np.array(red_positions)
        print(f"\nRed line position statistics:")
        print(f"  Mean: {np.mean(red_positions):.3f}")
        print(f"  Std: {np.std(red_positions):.3f}")
        print(f"  Min: {np.min(red_positions):.3f}")
        print(f"  Max: {np.max(red_positions):.3f}")
        
        return np.array(labels, dtype=np.int64)
    
    def _balance_classes(self):
        """Balance dataset by undersampling majority class"""
        left_indices = np.where(self.labels == 0)[0]
        right_indices = np.where(self.labels == 1)[0]
        
        min_count = min(len(left_indices), len(right_indices))
        
        left_sample = np.random.choice(left_indices, min_count, replace=False)
        right_sample = np.random.choice(right_indices, min_count, replace=False)
        
        keep_indices = np.concatenate([left_sample, right_sample])
        np.random.shuffle(keep_indices)
        
        self.frames = self.frames[keep_indices]
        self.labels = self.labels[keep_indices]
        
        print(f"Balanced dataset: {len(self.frames)} samples per class")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (C, H, W) normalized
            label: int (0=Left, 1=Right)
        """
        frame = self.frames[idx]
        label = self.labels[idx]
        
        frame_tensor = torch.from_numpy(frame).float()
        frame_tensor = self.transform(frame_tensor)
        
        return frame_tensor, label


def create_dataloaders(npz_paths, batch_size=32, val_split=0.2, 
                       target_size=64, balance_classes=True, num_workers=0):
    """
    Create train and validation dataloaders with visual labels
    
    Args:
        npz_paths: list of paths to NPZ files
        batch_size: batch size
        val_split: fraction of data for validation
        target_size: image size (not used, for compatibility)
        balance_classes: whether to balance classes
        num_workers: number of workers for dataloader
    
    Returns:
        train_loader, val_loader, None (for compatibility)
    """
    full_dataset = LaneDatasetVisual(
        npz_files=npz_paths,
        balance=balance_classes,
        debug=False
    )
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, None
