import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim // num_heads

        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt))
        # for i in range(num_heads):
        #     setattr(self, f'fc{i}', nn.Linear(self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt * num_heads, fdim)
        self.dropout = nn.Dropout(0)

    # fx_in: Nx x fdim   fp_in: Np x fdim
    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)
        d = math.sqrt(self.fatt)

        Nx = len(fx_in)
        f = torch.cat([fx_in, fp_in])  # f: Nx + Np, fdim
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        fx, fp = f[:, :Nx], f[:, Nx:]  # head x Nx  x fatt    head x Np x fatt

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.matmul(w, fp)  # head x Nx x 2*fatt
        # fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w

