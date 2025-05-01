# utils/earlystop.py
import os, torch

class EarlyStopper:
    def __init__(self, patience, ckpt_dir):
        self.patience, self.best_loss, self.counter = patience, float('inf'), 0
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir,"best.pth")

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.ckpt_path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
