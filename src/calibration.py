"""
ASDMotion — Platt Scaling Calibration

With a robust validation set (130 subjects), we can safely use Platt Scaling
to learn a temperature and bias term, significantly improving Expected Calibration Error (ECE).
"""

import numpy as np
import torch
from torch import optim, nn

class AnalyticalScaler:
    """Fallback stub to avoid import errors if train.py hasn't reloaded."""
    pass

class PlattScaler(nn.Module):
    """
    Learns to calibrate logits using Logistic Regression (Platt Scaling).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits * self.temperature + self.bias

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Fit Platt scaler using validation set logits."""
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).float()
        
        optimizer = optim.LBFGS([self.temperature, self.bias], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss
            
        optimizer.step(eval)

    def calibrate(self, logits):
        """Apply scaling and return probabilities."""
        with torch.no_grad():
            logits_t = torch.from_numpy(logits).float()
            scaled_logits = self.forward(logits_t)
            probs = torch.sigmoid(scaled_logits).numpy()
        return probs
