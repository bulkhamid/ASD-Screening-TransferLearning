"""Optional WandB logging without import-time network or filesystem effects."""


def log_epoch(stage, epoch, train_metrics, val_metrics, run=None):
    if run is None:
        return
    entry = {f"{stage}/train_{key}": value for key, value in train_metrics.items()}
    entry.update({f"{stage}/val_{key}": value for key, value in val_metrics.items()})
    entry["epoch"] = epoch
    run.log(entry, step=epoch)
