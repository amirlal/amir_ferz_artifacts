# activation.py
import torch
import torch.nn as nn

class LegacyTransmute(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        symbolic_gate = x * torch.tanh(self.alpha * x)
        recursive_uplift = self.beta * torch.sigmoid(self.gamma * x)
        return symbolic_gate + recursive_uplift
