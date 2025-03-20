import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# Dummy dataset for demonstration
class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # Return a random tensor (simulate an image) and a label (0 or 1)
        image = torch.randn(3, 224, 224)
        label = torch.randint(0, 2, (1,)).item()
        return image, label

# Simple CNN-based classifier using pre-trained ResNet18
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    def forward(self, x):
        return self.resnet(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader)}")

if __name__ == "__main__":
    train()
