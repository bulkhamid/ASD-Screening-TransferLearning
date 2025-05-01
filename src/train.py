# train.py
import os, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split

from data.dataloader       import make_dataloaders
from models.simple3dlstm   import Simple3DLSTM
from models.pretrained_r3d import PretrainedR3D
from utils.metrics         import compute_metrics
from utils.earlystop       import EarlyStopper
from utils.wandb_logger    import log_epoch

def get_model(name, cfg):
    if name=="simple3dlstm":
        return Simple3DLSTM(cfg.max_frames, cfg.target_size)
    if name=="r3d":
        return PretrainedR3D()
    raise ValueError(f"Unknown model: {name}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=["simple3dlstm","r3d"], required=True)
    p.add_argument("--data_path",  type=str,   default="./")
    p.add_argument("--max_frames", type=int,   default=60)
    p.add_argument("--target_size",type=int,   nargs=2, default=[112,112])
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    wandb.init(project="ASD-Screening", config=vars(args))
    cfg = wandb.config

    # collect & split
    pos = [os.path.join(cfg.data_path,"Positive_Trimmed",f) for f in os.listdir(os.path.join(cfg.data_path,"Positive_Trimmed"))]
    neg = [os.path.join(cfg.data_path,"Negative_Trimmed",f) for f in os.listdir(os.path.join(cfg.data_path,"Negative_Trimmed"))]
    paths, labels = pos+neg, [1]*len(pos)+[0]*len(neg)
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
    val_p, te_p, val_l, te_l = train_test_split(tmp_p, tmp_l, test_size=2/3, stratify=tmp_l, random_state=42)

    # dataloaders (no per-frame augment by default)
    # train_loader, val_loader = make_dataloaders(tr_p, tr_l, val_p, val_l,
    #                                             cfg.max_frames, tuple(cfg.target_size),
    #                                             cfg.batch_size, cfg.workers,
    #                                             transform=None)
    train_loader, val_loader = make_dataloaders(
        tr_p, tr_l,
        val_p,   val_l,
        max_frames=60,
        target_size=(112,112),
        batch_size=8,
        num_workers=4,
        augment=True   # toggle on/off your spatial augms
    )
    # model, loss, optimizer, earlystopper
    model     = get_model(cfg.model, cfg).to(cfg.device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    stopper   = EarlyStopper(cfg.patience, os.path.join("checkpoints", cfg.model))

    # training loop
    for epoch in range(1, cfg.epochs+1):
        # train
        model.train(); losses=[]
        for X,y in train_loader:
            X,y = X.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            p = model(X); loss = criterion(p, y)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        tr_loss = np.mean(losses)

        # evaluate
        def eval_split(dl):
            ys, ps = [], []
            with torch.no_grad():
                model.eval()
                for X, y in dl:
                    X,y = X.to(cfg.device), y.to(cfg.device)
                    out = model(X)
                    ys.extend(y.cpu().numpy()); ps.extend(out.cpu().numpy())
            return compute_metrics(np.array(ys), np.array(ps))

        tr_metrics  = eval_split(train_loader)
        val_metrics = eval_split(val_loader)

        log_epoch(cfg.model, epoch, tr_metrics, val_metrics)
        print(f"[{cfg.model}] epoch {epoch}: train_auc={tr_metrics['auc']:.4f}, val_auc={val_metrics['auc']:.4f}")

        if stopper.step(val_metrics["loss"], model):
            print(f"→ Early stopping at epoch {epoch}")
            break

    # final test
    test_metrics = eval_split(val_loader)  # replace with true test loader if you have one
    print("Test metrics:", test_metrics)
    wandb.log({f"test_{k}":v for k,v in test_metrics.items()})
    wandb.finish()
    print("Done. Best model at:", stopper.ckpt_path)
