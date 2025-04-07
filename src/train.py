import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configuration
CONFIG = {
    "data_path": "./",  # Base path for your video folders
    "target_size": (112, 112),
    "max_frames": 60,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-4,
    "num_workers": 0,  # For debugging; set higher when training on GPU
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_names": ["negative", "positive"]
}

# -----------------------------
# 1. Data Augmentation & Preprocessing
# -----------------------------
class AddGaussianNoise(object):
    def __call__(self, x):
        return x + torch.randn_like(x) * 0.02

class VideoTransform(object):
    """
    Applies a given image transformation to each frame in the video.
    Expects video tensor of shape (C, T, H, W).
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video):
        C, T, H, W = video.shape
        transformed_video = torch.empty_like(video)
        for t in range(T):
            frame = video[:, t, :, :]  # Shape: (C, H, W)
            # Convert tensor to PIL image for augmentation (if needed)
            # Note: transforms like ColorJitter expect a PIL Image
            frame = self.transform(frame)
            transformed_video[:, t, :, :] = frame
        return transformed_video

# Define a per-frame transform using a composition of augmentations.
# Here we include: ColorJitter, RandomHorizontalFlip, RandomRotation, conversion back to tensor,
# and adding Gaussian noise.
frame_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    AddGaussianNoise()
])

# Wrap the per-frame transform in our VideoTransform class.
video_transform = VideoTransform(frame_transforms)

# -----------------------------
# 2. Dataset Definition
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < CONFIG["max_frames"]:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, CONFIG["target_size"])
            frames.append(frame)
        cap.release()
        
        # If video has fewer frames than required, pad with black frames.
        if len(frames) < CONFIG["max_frames"]:
            if len(frames) > 0:
                pad = [np.zeros_like(frames[0])] * (CONFIG["max_frames"] - len(frames))
            else:
                pad = [np.zeros((CONFIG["target_size"][1], CONFIG["target_size"][0], 3), dtype=np.uint8)] * CONFIG["max_frames"]
            frames += pad
            
        video = np.array(frames, dtype=np.float32) / 255.0  # Normalize to [0,1]
        video = torch.from_numpy(video).permute(3, 0, 1, 2)   # Shape: (C, T, H, W)
        
        if self.transform:
            video = self.transform(video)
            
        return video, torch.tensor(label, dtype=torch.float32)

# -----------------------------
# 3. Model Architecture & Training Improvements
# -----------------------------
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(128)
        )
        
        # Compute the flattened feature size after CNN layers:
        # With three pooling layers (each reducing H and W by a factor of 2),
        # the spatial dimensions become (target_size[0]//8, target_size[1]//8).
        flattened_size = 128 * (CONFIG["target_size"][0] // 8) * (CONFIG["target_size"][1] // 8)
        
        self.lstm = nn.LSTM(
            input_size=flattened_size,
            hidden_size=64,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)  # Expected shape: (B, 128, T, H/8, W/8)
        # Permute to (B, T, 128, H/8, W/8)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, CONFIG["max_frames"], -1)  # Flatten CNN features per frame
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)

def train_model(model, train_loader, val_loader):
    model.to(CONFIG["device"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
    # Scheduler reduces LR when val_loss plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    epochs_no_improve = 0
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        
        for videos, labels in train_loader:
            videos = videos.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])
                
                outputs = model(videos)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * videos.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Early stopping based on validation loss improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
                
    return model

# -----------------------------
# 4. Main Routine: Data Loading, Training, and Inference
# -----------------------------
if __name__ == "__main__":
    video_paths = []
    labels = []
    for class_idx, class_name in enumerate(CONFIG["class_names"]):
        class_dir = os.path.join(CONFIG["data_path"], class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.mp4', '.avi', '.webm')):
                video_paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = VideoDataset(train_paths, train_labels, transform=video_transform)
    val_dataset = VideoDataset(val_paths, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    model = AutismDetectionModel()
    trained_model = train_model(model, train_loader, val_loader)

    def predict_video(model, video_path):
        model.eval()
        # Apply the same transform at inference if desired.
        video_dataset = VideoDataset([video_path], [0], transform=video_transform)
        video, _ = video_dataset[0]
        video = video.unsqueeze(0).to(CONFIG["device"])
        with torch.no_grad():
            pred = model(video).item()
        return f"Autism probability: {pred:.2%}"

    # Replace with the path to a sample video.
    sample_video = "./positive/20240927085658.webm"
    print(predict_video(trained_model, sample_video))
