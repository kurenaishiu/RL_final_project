import torch
import torch.nn as nn
import numpy as np

class StudentAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(384,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self,x): return self.net(x)

