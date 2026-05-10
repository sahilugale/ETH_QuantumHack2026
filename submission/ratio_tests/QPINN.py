# %%

import math
import importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# MerLin package import name is usually `merlin`.
try:
    import merlin as ML
except Exception as exc:
    raise ImportError(
        "Could not import MerLin. Install with: pip install merlinquantum"
    ) from exc

print("torch", torch.__version__)
print("merlin", getattr(ML, "__version__", "version unknown"))

# %%

# Reproducibility and dtype
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# MerLin's high-level layers commonly return float32 tensors.
# Keep the whole notebook in float32 to avoid errors such as:
# RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double
torch.set_default_dtype(torch.float32)
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("dtype:", dtype)



# %%

# PDE constants
alpha = 0.1
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

def exact_u(x, t):
    return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)

def sample_interior(n):
    x = torch.rand(n, 1, device=device, dtype=torch.get_default_dtype())
    t = torch.rand(n, 1, device=device, dtype=torch.get_default_dtype())
    xt = torch.cat([x, t], dim=1)
    xt.requires_grad_(True)
    return xt

def sample_initial(n):
    x = torch.rand(n, 1, device=device, dtype=torch.get_default_dtype())
    t = torch.zeros_like(x)
    xt = torch.cat([x, t], dim=1)
    return xt

def sample_boundary(n):
    t = torch.rand(n, 1, device=device, dtype=torch.get_default_dtype())
    half = n // 2
    x0 = torch.zeros(half, 1, device=device, dtype=torch.get_default_dtype())
    x1 = torch.ones(n - half, 1, device=device, dtype=torch.get_default_dtype())
    x = torch.cat([x0, x1], dim=0)
    xt = torch.cat([x, t], dim=1)
    return xt

# %%

class MerlinHeatQPINN(nn.Module):
    def __init__(self, feature_size=14, quantum_output_size=14, hidden=11):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, feature_size),
        )

        # High-level MerLin photonic layer.
        # If your installed MerLin exposes extra architecture arguments, you can add them here.
        self.quantum = ML.QuantumLayer.simple(
            input_size=feature_size,
            output_size=quantum_output_size,
        )

        self.readout = nn.Sequential(
            nn.Linear(quantum_output_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)

        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        # Enforce homogeneous Dirichlet boundary conditions exactly.
        u = x * (1.0 - x) * q_u
        return u, ux_hat

model = MerlinHeatQPINN().to(device=device, dtype=dtype)

# Defensive dtype alignment: if MerLin creates/returns float32 internally,
# the surrounding PyTorch layers must also be float32.
for p in model.parameters():
    if p.is_floating_point():
        p.data = p.data.to(dtype)

print(model)
print("first parameter dtype:", next(model.parameters()).dtype)


# %%
total_params = sum(p.numel() for p in model.parameters())

quantum_params = sum(
    p.numel() for p in model.quantum.parameters()
)

classical_params = total_params - quantum_params

print("Total:", total_params)
print("Quantum:", quantum_params)
print("Classical:", classical_params)

print("Quantum Parameter Ratio:", quantum_params / total_params)

# %%

def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def pde_and_consistency_residuals(model, xt):
    u, ux_hat = model(xt)

    grad_u = gradients(u, xt)
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_ux_hat = gradients(ux_hat, xt)
    ux_hat_x = grad_ux_hat[:, 0:1]

    pde_residual = u_t - alpha * ux_hat_x
    consistency_residual = u_x - ux_hat
    return pde_residual, consistency_residual


# %%

# Loss weights and training settings
n_f = 64      # interior points
n_i = 64      # initial condition points
n_b = 64      # boundary points, mostly redundant due to hard BC but kept as a check

epochs = 300
lr = 1e-2
lambda_f = 1.0
lambda_c = 0.1
lambda_i = 10.0
lambda_b = 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = nn.MSELoss()

history = []

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Trainable Parameters: {count_parameters(model)}")


# %%

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    xt_f = sample_interior(n_f)
    xt_i = sample_initial(n_i)
    xt_b = sample_boundary(n_b)

    # Physics and consistency losses
    r_f, r_c = pde_and_consistency_residuals(model, xt_f)
    loss_f = mse(r_f, torch.zeros_like(r_f))
    loss_c = mse(r_c, torch.zeros_like(r_c))

    # Initial condition loss
    u_i, _ = model(xt_i)
    x_i = xt_i[:, 0:1]
    t_i = xt_i[:, 1:2]
    loss_i = mse(u_i, exact_u(x_i, t_i))

    # Boundary loss: should already be near zero by construction.
    u_b, _ = model(xt_b)
    loss_b = mse(u_b, torch.zeros_like(u_b))

    loss = lambda_f * loss_f + lambda_c * loss_c + lambda_i * loss_i + lambda_b * loss_b
    loss.backward()
    optimizer.step()

    history.append([loss.item(), loss_f.item(), loss_c.item(), loss_i.item(), loss_b.item()])

    if epoch == 1 or epoch % 25 == 0:
        print(
            f"epoch {epoch:04d} | loss={loss.item():.4e} | "
            f"pde={loss_f.item():.2e} | cons={loss_c.item():.2e} | "
            f"ic={loss_i.item():.2e} | bc={loss_b.item():.2e}"
        )


# %%  

# Plot training losses
hist = np.array(history)
plt.figure(figsize=(7, 4))
plt.semilogy(hist[:, 0], label="total")
plt.semilogy(hist[:, 1], label="PDE")
plt.semilogy(hist[:, 2], label="consistency")
plt.semilogy(hist[:, 3], label="initial")
plt.semilogy(hist[:, 4], label="boundary")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("MerLin DV-photonic QPINN surrogate training")
plt.show()


# %%

# Evaluate on a grid
nx, nt = 60, 60
x = torch.linspace(0, 1, nx, device=device, dtype=dtype).reshape(-1, 1)
t = torch.linspace(0, 1, nt, device=device, dtype=dtype).reshape(-1, 1)
X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

with torch.no_grad():
    U_pred, UX_hat = model(xt_grid)
    U_pred = U_pred.reshape(nx, nt).cpu()
    U_true = exact_u(xt_grid[:, 0:1], xt_grid[:, 1:2]).reshape(nx, nt).cpu()

# %%

# Visual comparison
plt.figure(figsize=(6, 5))
plt.imshow(U_true.numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
plt.colorbar(label="u")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Exact heat equation solution")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(U_pred.numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
plt.colorbar(label="u")
plt.xlabel("t")
plt.ylabel("x")
plt.title("MerLin DV-photonic QPINN prediction")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow((U_pred - U_true).numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
plt.colorbar(label="error")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Prediction error")
plt.show()

# %%

# Metrics

rel_l2 = torch.linalg.norm(U_pred - U_true) / torch.linalg.norm(U_true)
print("relative L2 error:", float(rel_l2))


rmse = torch.sqrt(torch.mean((U_pred - U_true)**2))
print("RMSE:", float(rmse))

mae = torch.mean(torch.abs(U_pred - U_true))
print("MAE:", float(mae))

l_infinity = torch.max(torch.abs(U_pred - U_true))
print("L-infinity error:", float(l_infinity))


nmse = torch.sum((U_true - U_pred)**2) / torch.sum(U_true**2)
print("NMSE:", float(nmse))

num_layers = len(list(model.modules()))
print(f"Total internal modules/layers: {num_layers}")
# %%



