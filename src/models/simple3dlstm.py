# models/simple3dlstm.py
import torch.nn as nn

class Simple3DLSTM(nn.Module):
    def __init__(self, max_frames, target_size):
        super().__init__()
        C,T,H,W = 3, max_frames, *target_size
        self.cnn = nn.Sequential(
            nn.Conv3d(3,32,(3,3,3),padding=1), nn.ReLU(),
            nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(32),
            nn.Conv3d(32,64,(3,3,3),padding=1), nn.ReLU(),
            nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(64),
            nn.Conv3d(64,128,(3,3,3),padding=1), nn.ReLU(),
            nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(128),
        )
        feat_dim = 128 * (H//8) * (W//8)
        self.lstm = nn.LSTM(feat_dim, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32,1),  nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        f = self.cnn(x)                   # (B,128,T,H/8,W/8)
        f = f.permute(0,2,1,3,4)          # (B,T,128,H/8,W/8)
        f = f.reshape(B, f.size(1), -1)   # (B,T,feat_dim)
        out,_ = self.lstm(f)              # (B,T,64)
        return self.head(out[:,-1,:]).view(-1)  
