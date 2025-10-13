"""
Simple 3D CNN + LSTM model for video classification.

This module implements a custom neural network architecture that combines
3D convolutional layers for spatial-temporal feature extraction with LSTM
for temporal modeling.
"""
import torch.nn as nn


class Simple3DLSTM(nn.Module):
    """
    3D CNN + LSTM architecture for video classification.
    
    The model uses three 3D convolutional blocks followed by an LSTM layer
    and a fully connected head for binary classification.
    
    Architecture:
        - 3x Conv3D blocks with ReLU, MaxPool3D, BatchNorm3D, and Dropout3D
        - LSTM layer for temporal modeling
        - Fully connected head with dropout
    
    Args:
        max_frames: Number of frames per video (temporal dimension)
        target_size: Tuple of (height, width) for input frames
        p_drop: Dropout probability for 3D convolutional layers (default: 0.3)
        p_head: Dropout probability for fully connected layers (default: 0.5)
    """
    
    def __init__(self, max_frames, target_size, p_drop: float = 0.3, p_head: float = 0.5):
        super().__init__()
        C, T, H, W = 3, max_frames, *target_size
        
        # 3D Convolutional Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv3d(3, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.Dropout3d(p_drop),
            
            # Block 2: 32 -> 64 channels
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.Dropout3d(p_drop),
            
            # Block 3: 64 -> 128 channels
            nn.Conv3d(64, 128, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p_drop)
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
        B = x.size(0)
        
        # Extract spatial-temporal features
        f = self.cnn(x)  # (B, 128, T, H/8, W/8)
        
        # Reshape for LSTM: (B, T, features)
        f = f.permute(0, 2, 1, 3, 4)  # (B, T, 128, H/8, W/8)
        f = f.reshape(B, f.size(1), -1)  # (B, T, feat_dim)
        
        # LSTM processing
        out, _ = self.lstm(f)  # (B, T, 64)
        
        # Use last timestep for classification
        return self.head(out[:, -1, :]).view(-1)  # (B,)

