"""
PACE-ASD — Platt Scaling Calibration

Fits a temperature + bias term on out-of-fold validation logits
(Dryad-only) using LBFGS. Applied after each fold's best checkpoint
is selected. Per protocol Section 3.
"""

import numpy as np
import torch
from torch import nn, optim


class PlattScaler(nn.Module):
    """
    Learns a strictly positive temperature recalibration with bias:
        logit_cal = exp(log_temperature) * logit + bias

    Ensures rank-order preservation (AUC invariant under positive scaling)
    and robust calibration on small validation cohorts.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # init T = exp(0) = 1.0
        self.bias            = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.temperature + self.bias

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.05, max_iter: int = 100, l2_reg: float = 0.01) -> None:
        """Fit positive temperature and bias with L2 regularization."""
        logits_t = torch.from_numpy(logits.astype(np.float32))
        labels_t = torch.from_numpy(labels.astype(np.float32))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS(
            [self.log_temperature, self.bias],
            lr=lr, max_iter=max_iter,
        )

        def closure():
            optimizer.zero_grad()
            pred = self.forward(logits_t)
            nll  = criterion(pred, labels_t)
            reg  = l2_reg * (self.log_temperature ** 2 + self.bias ** 2)
            loss = nll + reg
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            # Fallback if LBFGS encounters numerical issues on small sets
            opt_adam = optim.Adam([self.log_temperature, self.bias], lr=0.01)
            for _ in range(50):
                opt_adam.zero_grad()
                loss = criterion(self.forward(logits_t), labels_t) + l2_reg * (self.log_temperature ** 2 + self.bias ** 2)
                loss.backward()
                opt_adam.step()

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for an array of logits."""
        with torch.no_grad():
            scaled = self.forward(torch.from_numpy(logits.astype(np.float32)))
            return torch.sigmoid(scaled).numpy()
