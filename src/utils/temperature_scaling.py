"""
Temperature scaling for model calibration.

Temperature scaling is a post-processing calibration method that rescales
the logits of a trained neural network to produce better-calibrated
probability estimates.

Reference:
    Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Temperature scaling wrapper for neural network calibration.
    
    This module wraps a trained model and learns a single scalar temperature
    parameter T > 0 that rescales the model's logits (logit/T) to improve
    probability calibration. The temperature is learned by minimizing the
    negative log-likelihood on a held-out calibration set.
    
    Args:
        model: Pre-trained PyTorch model that outputs logits
    
    Attributes:
        model: The wrapped model
        log_T: Learnable parameter (log of temperature) initialized to 0
        temperature: Buffer storing exp(log_T) for convenience
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Store log(T) instead of T directly for numerical stability
        # Initialized at log(1.0) = 0.0, so initial T = 1.0 (no scaling)
        self.log_T = nn.Parameter(torch.zeros(1))
        
        # Cosmetic alias for printing (not used in forward pass)
        self.register_buffer("temperature", torch.exp(self.log_T).detach())

    def forward(self, x):
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Input tensor
        
        Returns:
            Temperature-scaled logits
        """
        logits = self.model(x)
        T = self.log_T.exp()
        return logits / T

    def set_temperature(self, loader, device='cuda'):
        """
        Learn optimal temperature on a calibration/validation set.
        
        Fits the temperature parameter T by minimizing negative log-likelihood
        (binary cross-entropy) on the provided data loader. Uses L-BFGS optimizer
        for single-parameter optimization.
        
        Args:
            loader: DataLoader for calibration set
            device: Device to run optimization on (default: 'cuda')
        
        Returns:
            self (for method chaining)
        """
        self.to(device)
        self.eval()

        nll_criterion = nn.BCELoss()

        # Collect all logits and labels
        logits_list, labels_list = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = self.model(X).view(-1)
                logits_list.append(logits)
                labels_list.append(y.float().view(-1))
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Single-parameter optimization using L-BFGS
        # Run on CPU for numerical stability
        self.log_T.data = self.log_T.data.cpu()
        optimiser = optim.LBFGS([self.log_T], lr=0.05, max_iter=50)

        def _eval():
            """Closure for L-BFGS optimizer."""
            optimiser.zero_grad()
            T = self.log_T.exp()
            loss = nll_criterion(
                torch.sigmoid(logits.cpu() / T),
                labels.cpu()
            )
            loss.backward()
            return loss

        optimiser.step(_eval)
        
        # Move back to original device and update temperature buffer
        self.log_T.data = self.log_T.data.to(device)
        self.temperature = self.log_T.exp().detach()
        
        print(f'Optimal temperature: T = {self.temperature.item():.3f}')
        return self
