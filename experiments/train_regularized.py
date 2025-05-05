#!/usr/bin/env python
"""
Regularised quick‑run for either  Simple3DLSTM  or  Pretrained‑R3D
-----------------------------------------------------------------
▸ weight‑decay AdamW            (‑‑wd)
▸ optional 3‑D dropout in LSTM  (‑‑drop   – ignored for R3D)
▸ cosine LR + mixed precision
▸ early‑stopping on val‑loss    (‑‑patience)
▸ temperature‑scaling           (‑‑temp)
▸ threshold picked on val F1 → reported on TEST
"""

import time, argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)

# ------------------------------------------------------------------ #
#  local imports
# ------------------------------------------------------------------ #
from src.utils.data_cached import make_train_loader, make_eval_loader
from src.utils.temperature_scaling import TemperatureScaler
from src.models.simple3dlstm      import Simple3DLSTM
from src.models.pretrained_r3d    import PretrainedR3D


# ── reproducibility ─────────────────────────────────────────
SEED = 42                        # pick your favourite prime
import random, numpy as np, torch

random.seed(SEED)                # Python stdlib
np.random.seed(SEED)             # NumPy
torch.manual_seed(SEED)          # CPU RNG
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True   # slower but repeatable
torch.backends.cudnn.benchmark     = False


# ───────────────────────── helpers ──────────────────────────
def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true))>1 else 0.5,
        acc = accuracy_score (y_true, y_pred),
        prec= precision_score(y_true, y_pred, zero_division=0),
        rec = recall_score   (y_true, y_pred, zero_division=0),
        f1  = f1_score       (y_true, y_pred, zero_division=0)
    )

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        logit = model(X.to(device))
        ys.extend(y.numpy())
        ps.extend(torch.sigmoid(logit).cpu().numpy())
    return np.array(ys), np.array(ps)

# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   choices=["simple3dlstm","r3d"],
                    required=True)
    ap.add_argument("--batch",   type=int,  default=8)
    ap.add_argument("--epochs",  type=int,  default=25)
    ap.add_argument("--patience",type=int,  default=10)
    ap.add_argument("--wd",      type=float,default=1e-4)
    # Simple3DLSTM‑specific
    ap.add_argument("--drop",    type=float,default=0.0,
                    help="3‑D dropout prob in conv blocks (ignored for R3D)")
    # image size
    ap.add_argument("--target",  nargs=2, type=int, default=[112,112])
    # temperature scaling
    ap.add_argument("--temp",    action="store_true")
    # R3D fine‑tune stage2 flag (optional)
    ap.add_argument("--finetune",action="store_true",
                    help="if set, unfreezes layer4 after head warm‑up")
    cfg = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader = make_train_loader(cfg.batch, target_size=tuple(cfg.target))
    vl_loader = make_eval_loader("val",  cfg.batch)
    te_loader = make_eval_loader("test", cfg.batch)

    # ------------------------------------------------------------------ #
    #  build model + optimiser per architecture
    # ------------------------------------------------------------------ #
    if cfg.model == "simple3dlstm":
        model = Simple3DLSTM(60, tuple(cfg.target), p_drop=cfg.drop).to(device)
        opt   = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=cfg.wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cfg.epochs, eta_min=1e-6)
        stages = [("full", cfg.epochs, model.parameters())]

    else:  # ── R3D quick‑run ────────────────────────────────────────────
        model = PretrainedR3D().to(device)

        # stage‑1: train only the newly‑initialised fc layer
        for p in model.parameters(): p.requires_grad_(False)
        model.net.fc.requires_grad_(True)

        opt = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                          lr=1e-3, weight_decay=cfg.wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cfg.epochs//2, eta_min=1e-6)
        stages = [("head", cfg.epochs//2,
                   filter(lambda p:p.requires_grad, model.parameters()))]

        # stage‑2: optionally unfreeze layer4 for light fine‑tuning
        if cfg.finetune:
            stages.append(("ft", cfg.epochs//2, []))   # placeholder – will fill later

    crit   = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # ------------------------------------------------------------------ #
    #  training stages
    # ------------------------------------------------------------------ #
    for stage_name, n_ep, params in stages:
        if stage_name == "ft":  # unlock layer4 & fc
            for n,p in model.net.named_parameters():
                if "layer4" in n or "fc" in n:
                    p.requires_grad_(True)
                    params.append(p)
            opt = optim.AdamW(params, lr=1e-4, weight_decay=cfg.wd)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=n_ep, eta_min=1e-6)

        best_val, wait = float("inf"), 0
        print(f"\n▶  Stage **{stage_name}**  "
              f"(epochs={n_ep}, wd={cfg.wd})\n")

        for ep in range(1, n_ep+1):
            t0 = time.time();  model.train()
            for X,y in tr_loader:
                X,y = X.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=device.type):
                    loss = crit(model(X), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            sched.step()

            # --- validation loss (raw logits) ---
            ys_val, ps_val = eval_loader(model, vl_loader, device)
            val_loss = crit(
                torch.from_numpy(ps_val).logit(),
                torch.from_numpy(ys_val)
            ).item()
            print(f"[{stage_name}] ep {ep:02d}  "
                  f"val‑loss {val_loss:.4f}  "
                  f"time={time.time()-t0:.1f}s")

            if val_loss < best_val:
                best_val, wait = val_loss, 0
                torch.save(model.state_dict(), f"best_{stage_name}.pth")
            else:
                wait += 1
                if wait >= cfg.patience:
                    print("↯  early‑stopping this stage")
                    break

        # reload best for next stage / inference
        model.load_state_dict(torch.load(f"best_{stage_name}.pth",
                                         map_location=device))

    # ------------------------------------------------------------------ #
    #  calibration
    # ------------------------------------------------------------------ #
    # ────────────────────── calibration ─────────────────────────
    if cfg.temp:
        infer_net = TemperatureScaler(model).set_temperature(vl_loader, device)
        print(f"✓  temperature scaling applied  "
        f"(T = {infer_net.temperature.item():.2f})")
    else:
        infer_net = model

    # ── pick the decision threshold on calibrated VAL ───────────
    ys_val, ps_val = eval_loader(infer_net, vl_loader, device)
    ths  = np.linspace(0.05, 0.95, 19)
    best_t = float(ths[int(np.argmax([f1_score(ys_val, ps_val >= t) for t in ths]))])
    print(f"\n◎  chosen threshold = {best_t:.2f}")

    # we still have logits inside infer_net, just grab them
    # ----- post‑TS validation diagnostics -----
    with torch.no_grad():
        logits_val, ys_val_t = [], []
        for X, y in vl_loader:
            logits_val.append( infer_net(X.to(device)) )   # cuda
            ys_val_t.append( y.to(device) )                # <─ move labels too

        logits_val = torch.cat(logits_val)                 # cuda
        ys_val_t   = torch.cat(ys_val_t)                   # cuda

    val_loss_post = crit(logits_val, ys_val_t).item()
    val_auc_post  = roc_auc_score(
                    ys_val_t.cpu().numpy(),
                    torch.sigmoid(logits_val).cpu().numpy()
            )
    print(f"val‑loss={val_loss_post:.3f}  val‑AUC={val_auc_post:.3f}")

    # <<<

    # ────────────────────── TEST metrics ────────────────────────
    ys_te, ps_te = eval_loader(infer_net, te_loader, device)
    mets = compute_metrics(ys_te, ps_te, thr=best_t)
    print("\n=== TEST metrics ===")
    for k, v in mets.items():
        print(f"{k:7s}: {v:.3f}")


# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
