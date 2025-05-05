# temperature_scaling.py ----------------------------------------------------
import torch, torch.nn as nn, torch.optim as optim
from torch.nn import functional as F

class _LogitWrapper(nn.Module):
    """Expose the model’s raw logit (pre‑sigmoid) for BCE logits‑loss."""
    def __init__(self, net):           # net returns probability or logit?
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        # If the model already outputs a logit tensor of shape (B,1) leave it.
        # If it outputs a probability in [0,1], convert to logit domain.
        if (out.detach() <= 0).any() or (out.detach() >= 1).any():
            return out          # assume logits
        logit = torch.logit(out.clamp(1e-6, 1-(1e-6)))
        return logit

class TemperatureScaler(nn.Module):
    """Learns a single scalar T > 0 to rescale logits: logit/T."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.net   = model
        # old  : self.temperature = nn.Parameter(torch.ones(1))
        # new ▼ log_T initialised at log(1.0) = 0.0
        # self.log_T = nn.Parameter(torch.zeros(1))
        self.log_T = nn.Parameter(torch.zeros(1))   # log T (so   T = e^{log_T})

        # ── cosmetic alias so other code can still access `.temperature`
        #    (used only for printing, not in forward pass)
        self.register_buffer("temperature", torch.exp(self.log_T).detach())

    def forward(self, x):
        logits = self.model(x)
        T = self.log_T.exp()
        return logits / T

    # ------------------------------------------------------------------
    def set_temperature(self, loader, device='cuda'):
        """Fit T on a held‑out calib/val loader by minimising NLL."""
        self.to(device)
        self.eval()

        nll_criterion = nn.BCELoss()

        # Collect all logits & labels ----------------------------------
        logits_list, labels_list = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = self.net(X).view(-1)
                logits_list.append(logits)
                labels_list.append(y.float().view(-1))
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # single‑parameter optimisation -------------------------------
        # LBFGS is picky on GPU; run it on CPU for numerical safety
        self.log_T.data = self.log_T.data.cpu()
        optimiser = optim.LBFGS([self.log_T], lr=0.05, max_iter=50)

        def _eval():
            optimiser.zero_grad()
            T    = self.log_T.exp()                    # ← convert to real temperature
            loss = nll_criterion(torch.sigmoid(logits.cpu() / T), labels.cpu())
            loss.backward()
            return loss

        optimiser.step(_eval)
        # move back to original device & refresh the alias
        self.log_T.data = self.log_T.data.to(device)
        self.temperature = self.log_T.exp().detach()
        print(f'>> optimal T = {self.temperature.item():.3f}')
        return self
