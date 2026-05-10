import math
import torch
import torch.nn as nn

try:
    import merlin as ML
except ImportError:
    print("Warning: MerLin not found. Ensure you are running this in the Quandela environment.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
ALPHA = 0.1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exact_u(x, t):
    return torch.exp(-ALPHA * math.pi**2 * t) * torch.sin(math.pi * x)

def make_xt(x, t):
    return torch.cat([x, t], dim=1)

def sample_interior(n):
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    xt = make_xt(x, t)
    xt.requires_grad_(True)
    return xt

def sample_initial(n):
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.zeros_like(x)
    return make_xt(x, t)

def sample_boundary(n):
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    half = n // 2
    x0 = torch.zeros(half, 1, device=DEVICE, dtype=DTYPE)
    x1 = torch.ones(n - half, 1, device=DEVICE, dtype=DTYPE)
    x = torch.cat([x0, x1], dim=0)
    return make_xt(x, t)

def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

class ModelA_QPINN_Trick(nn.Module):
    def __init__(self, feature_size=6, hidden=19):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size))
        self.quantum = ML.QuantumLayer.simple(input_size=feature_size, output_size=feature_size)
        self.readout = nn.Sequential(nn.Linear(feature_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        q_u, ux_hat = self.readout(self.quantum(self.feature_map(xt)))[:, 0:1], self.readout(self.quantum(self.feature_map(xt)))[:, 1:2]
        return xt[:, 0:1] * (1.0 - xt[:, 0:1]) * q_u, ux_hat

class ModelB_Classical_Trick(nn.Module):
    def __init__(self, hidden=13):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        out = self.net(xt)
        return xt[:, 0:1] * (1.0 - xt[:, 0:1]) * out[:, 0:1], out[:, 1:2]

class ModelC_Classical_Direct(nn.Module):
    def __init__(self, hidden=13):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, xt):
        return xt[:, 0:1] * (1.0 - xt[:, 0:1]) * self.net(xt)