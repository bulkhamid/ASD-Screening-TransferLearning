# utils/wandb_logger.py
import wandb

def log_epoch(stage, epoch, train_metrics, val_metrics):
    entry = {f"{stage}/train_{k}":v for k,v in train_metrics.items()}
    entry.update({f"{stage}/val_{k}":v   for k,v in val_metrics.items()})
    entry["epoch"] = epoch
    wandb.log(entry)
