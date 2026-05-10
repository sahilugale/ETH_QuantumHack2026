
# %%

import math
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print("Device:", DEVICE)
print("Torch version:", torch.__version__)


# %%


alpha = 0.1


def exact_u(x, t):
    """Exact solution of the benchmark heat equation."""
    return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)


def make_xt(x, t):
    """Concatenate x and t into a two-column input tensor."""
    return torch.cat([x, t], dim=1)


# %%


def sample_interior(n):
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    xt = make_xt(x, t)
    xt.requires_grad_(True)
    return xt


def sample_initial(n):
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.zeros_like(x)
    xt = make_xt(x, t)
    y = exact_u(x, t)
    return xt, y


def sample_boundary(n):
    n0 = n // 2
    n1 = n - n0
    t0 = torch.rand(n0, 1, device=DEVICE, dtype=DTYPE)
    t1 = torch.rand(n1, 1, device=DEVICE, dtype=DTYPE)
    x0 = torch.zeros_like(t0)
    x1 = torch.ones_like(t1)
    xt = torch.cat([make_xt(x0, t0), make_xt(x1, t1)], dim=0)
    y = torch.zeros(n, 1, device=DEVICE, dtype=DTYPE)
    return xt, y


# %%


class MLP_PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, depth=4, out_dim=1, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        return self.net(xt)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total trainable parameters in MLP_PINN:", count_parameters(MLP_PINN()))
# %%


def pde_residual_direct(model, xt):
    """Compute u_t - alpha * u_xx using PyTorch automatic differentiation."""
    u = model(xt)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    grads2 = torch.autograd.grad(
        outputs=u_x,
        inputs=xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = grads2[:, 0:1]
    return u_t - alpha * u_xx


# %%


class MLP_AuxPINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, depth=4, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, 2))  # output: [u, u_x_hat]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        y = self.net(xt)
        u = y[:, 0:1]
        ux_hat = y[:, 1:2]
        return u, ux_hat
    


def pde_residual_aux(model, xt):
    """Compute u_t - alpha * d/dx(ux_hat), plus consistency u_x - ux_hat."""
    u, ux_hat = model(xt)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_ux_hat = torch.autograd.grad(
        outputs=ux_hat,
        inputs=xt,
        grad_outputs=torch.ones_like(ux_hat),
        create_graph=True,
        retain_graph=True,
    )[0]
    ux_hat_x = grad_ux_hat[:, 0:1]

    residual = u_t - alpha * ux_hat_x
    consistency = u_x - ux_hat
    return residual, consistency

class MLP_AuxPINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, depth=4, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, 2))  # output: [u, u_x_hat]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        y = self.net(xt)
        u = y[:, 0:1]
        ux_hat = y[:, 1:2]
        return u, ux_hat


def pde_residual_aux(model, xt):
    """Compute u_t - alpha * d/dx(ux_hat), plus consistency u_x - ux_hat."""
    u, ux_hat = model(xt)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_ux_hat = torch.autograd.grad(
        outputs=ux_hat,
        inputs=xt,
        grad_outputs=torch.ones_like(ux_hat),
        create_graph=True,
        retain_graph=True,
    )[0]
    ux_hat_x = grad_ux_hat[:, 0:1]

    residual = u_t - alpha * ux_hat_x
    consistency = u_x - ux_hat
    return residual, consistency

print("Total trainable parameters in MLP_AuxPINN:", count_parameters(MLP_AuxPINN()))

# %%


@dataclass
class TrainConfig:
    epochs: int = 300
    n_f: int = 256
    n_i: int = 128
    n_b: int = 128
    lr: float = 1e-3
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    lambda_consistency: float = 1.0
    print_every: int = 300



USE_AUXILIARY_DERIVATIVE = True  # Set to False to train the direct PINN instead of the auxiliary one
config = TrainConfig()

if USE_AUXILIARY_DERIVATIVE:
    model = MLP_AuxPINN(hidden_dim=10, depth=4).to(device=DEVICE, dtype=DTYPE)
else:
    model = MLP_PINN(hidden_dim=10, depth=4).to(device=DEVICE, dtype=DTYPE)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
mse = nn.MSELoss()

print("Using auxiliary derivative:", USE_AUXILIARY_DERIVATIVE)
print("Trainable parameters:", count_parameters(model))


# %%


history = {"total": [], "pde": [], "ic": [], "bc": [], "consistency": []}
start = time.time()

for epoch in range(1, config.epochs + 1):
    optimizer.zero_grad()

    xt_f = sample_interior(config.n_f)
    xt_i, y_i = sample_initial(config.n_i)
    xt_b, y_b = sample_boundary(config.n_b)

    if USE_AUXILIARY_DERIVATIVE:
        res, consistency = pde_residual_aux(model, xt_f)
        u_i, _ = model(xt_i)
        u_b, _ = model(xt_b)
        loss_consistency = mse(consistency, torch.zeros_like(consistency))
    else:
        res = pde_residual_direct(model, xt_f)
        u_i = model(xt_i)
        u_b = model(xt_b)
        loss_consistency = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    loss_pde = mse(res, torch.zeros_like(res))
    loss_ic = mse(u_i, y_i)
    loss_bc = mse(u_b, y_b)

    loss = (
        config.lambda_pde * loss_pde
        + config.lambda_ic * loss_ic
        + config.lambda_bc * loss_bc
        + config.lambda_consistency * loss_consistency
    )

    loss.backward()
    optimizer.step()

    history["total"].append(loss.item())
    history["pde"].append(loss_pde.item())
    history["ic"].append(loss_ic.item())
    history["bc"].append(loss_bc.item())
    history["consistency"].append(loss_consistency.item())

    if epoch % config.print_every == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d} | loss={loss.item():.3e} | "
            f"pde={loss_pde.item():.3e} | ic={loss_ic.item():.3e} | "
            f"bc={loss_bc.item():.3e} | cons={loss_consistency.item():.3e}"
        )

elapsed = time.time() - start
print(f"Training time: {elapsed:.1f} s")


# %%


plt.figure(figsize=(7, 4))
for key, values in history.items():
    if key == "consistency" and not USE_AUXILIARY_DERIVATIVE:
        continue
    plt.semilogy(values, label=key)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training losses")
plt.tight_layout()
plt.show()

# %%


@torch.no_grad()
def predict_u(model, xt):
    if USE_AUXILIARY_DERIVATIVE:
        u, _ = model(xt)
        return u
    return model(xt)


def evaluate_on_grid(model, nx=101, nt=101):
    x = torch.linspace(0, 1, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    t = torch.linspace(0, 1, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    u_pred = predict_u(model, xt).reshape(nx, nt)
    u_true = exact_u(X, T)

    rel_l2 = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
    max_abs = torch.max(torch.abs(u_pred - u_true))
    return X.detach().cpu(), T.detach().cpu(), u_pred.detach().cpu(), u_true.detach().cpu(), rel_l2.item(), max_abs.item()


X, T, U_pred, U_true, rel_l2, max_abs = evaluate_on_grid(model)
print(f"Relative L2 error: {rel_l2:.4e}")
print(f"Max absolute error: {max_abs:.4e}")


# %%


plt.figure(figsize=(6, 4))
plt.contourf(T.numpy(), X.numpy(), U_true.numpy(), levels=50)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Exact solution")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.contourf(T.numpy(), X.numpy(), U_pred.numpy(), levels=50)
plt.xlabel("t")
plt.ylabel("x")
plt.title("PINN prediction")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.contourf(T.numpy(), X.numpy(), (U_pred - U_true).numpy(), levels=50)
plt.xlabel("t")
plt.ylabel("x")
plt.title("Prediction error")
plt.colorbar()
plt.tight_layout()
plt.show()


# %%

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
