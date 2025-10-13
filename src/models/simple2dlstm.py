"""
Simple 2D CNN + LSTM model for video classification.

This module implements a custom neural network architecture that combines
2D convolutional layers (applied per frame) for spatial feature extraction 
with LSTM for temporal modeling.
"""
import torch
import torch.nn as nn


class Simple2DLSTM(nn.Module):
    """
    2D CNN + LSTM architecture for video classification.
    
    The model uses three 2D convolutional blocks applied per frame followed 
    by an LSTM layer and a fully connected head for binary classification.
    
    Architecture:
        - 3x Conv2D blocks with ReLU, MaxPool2D, BatchNorm2D, and Dropout2D
        - LSTM layer for temporal modeling
        - Fully connected head with dropout
    
    Args:
        max_frames: Number of frames per video (temporal dimension)
        target_size: Tuple of (height, width) for input frames
        p_drop: Dropout probability for 2D convolutional layers (default: 0.3)
        p_head: Dropout probability for fully connected layers (default: 0.5)
    """
    
    def __init__(self, max_frames, target_size, p_drop: float = 0.3, p_head: float = 0.5):
        super().__init__()
        C, T, H, W = 3, max_frames, *target_size
        
        # 2D Convolutional Feature Extractor (applied per frame)
        self.cnn = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p_drop),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p_drop),
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p_drop)
        )
        
        # Calculate feature dimension after CNN
        feat_dim = 128 * (H // 8) * (W // 8)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(feat_dim, 64, batch_first=True)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_head),
            nn.Linear(32, 1)  # Output logits (no sigmoid)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch, channels, time, height, width)
        
        Returns:
            Logits tensor of shape (batch,) for binary classification
        """
        B, C, T, H, W = x.size()
        
        # Process each frame independently through 2D CNN
        # Reshape to (B*T, C, H, W) to process all frames in batch
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Extract spatial features
        f = self.cnn(x)  # (B*T, 128, H/8, W/8)
        
        # Reshape back to (B, T, features)
        feat_dim = f.size(1) * f.size(2) * f.size(3)
        f = f.reshape(B, T, feat_dim)  # (B, T, feat_dim)
        
        # LSTM processing
        out, _ = self.lstm(f)  # (B, T, 64)
        
        # Use last timestep for classification
        return self.head(out[:, -1, :]).view(-1)  # (B,)
