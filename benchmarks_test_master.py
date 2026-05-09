import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

try:
    import merlin as ML
except ImportError:
    print("Warning: MerLin not found. Ensure you are running this in the Quandela hackathon environment.")

# ==========================================
# 1. CONFIGURATION, CONSTANTS & DIRECTORIES
# ==========================================
SEEDS = [1234] 
EPOCHS = 300
LR = 1e-2
N_F = 64
N_I = 64
N_B = 64
ALPHA = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
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
    y = exact_u(x, t)
    return make_xt(x, t), y

def sample_boundary(n):
    n0, n1 = n // 2, n - n // 2
    t0 = torch.rand(n0, 1, device=DEVICE, dtype=DTYPE)
    t1 = torch.rand(n1, 1, device=DEVICE, dtype=DTYPE)
    x0, x1 = torch.zeros_like(t0), torch.ones_like(t1)
    xt = torch.cat([make_xt(x0, t0), make_xt(x1, t1)], dim=0)
    return xt, torch.zeros(n, 1, device=DEVICE, dtype=DTYPE)

# ==========================================
# 3. MODEL DEFINITIONS (Original Architecture)
# ==========================================
class MerlinHeatQPINN(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size))
        self.quantum = ML.QuantumLayer.simple(input_size=feature_size, output_size=quantum_output_size)
        self.readout = nn.Sequential(nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        x, z = xt[:, 0:1], self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)
        q_u, ux_hat = out[:, 0:1], out[:, 1:2]
        u = x * (1.0 - x) * q_u
        return u, ux_hat

class SurrogateClassicalPINN(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size))
        self.quantum_replacement = nn.Sequential(nn.Linear(feature_size, hidden), nn.Tanh(), nn.Linear(hidden, quantum_output_size))
        self.readout = nn.Sequential(nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        x, z = xt[:, 0:1], self.feature_map(xt)
        q = self.quantum_replacement(z)
        out = self.readout(q)
        q_u, ux_hat = out[:, 0:1], out[:, 1:2]
        u = x * (1.0 - x) * q_u
        return u, ux_hat

class MLP_AuxPINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        y = self.net(xt)
        return y[:, 0:1], y[:, 1:2]

class MLP_PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        return self.net(xt)

# ==========================================
# 4. LOSS & RESIDUAL FUNCTIONS
# ==========================================
def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def pde_residual_aux(u, ux_hat, xt):
    grad_u = gradients(u, xt)
    grad_ux_hat = gradients(ux_hat, xt)
    return grad_u[:, 1:2] - ALPHA * grad_ux_hat[:, 0:1], grad_u[:, 0:1] - ux_hat

def pde_residual_direct(u, xt):
    grad_u = gradients(u, xt)
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = gradients(u_x, xt)[:, 0:1]
    return u_t - ALPHA * u_xx

# ==========================================
# 5. EVALUATION LOOP
# ==========================================
@torch.no_grad()
def evaluate_on_grid(model, nx=60, nt=60):
    x = torch.linspace(0, 1, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    t = torch.linspace(0, 1, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    
    out = model(xt_grid)
    u_pred = out[0] if isinstance(out, tuple) else out
    u_pred = u_pred.reshape(nx, nt).cpu()
    u_true = exact_u(xt_grid[:, 0:1], xt_grid[:, 1:2]).reshape(nx, nt).cpu()
    
    rel_l2 = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
    return rel_l2.item()

def train_and_evaluate(model_fn, use_aux, freeze_quantum=False):
    metrics = {"l2": [], "pde": [], "time": [], "losses": [], "models": {}, "params": 0}
    mse = nn.MSELoss()
    
    temp_model = model_fn()
    metrics["params"] = count_parameters(temp_model)
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_fn().to(DEVICE, dtype=DTYPE)
        
        for p in model.parameters():
            if p.is_floating_point(): p.data = p.data.to(DTYPE)
                
        if freeze_quantum and hasattr(model, 'quantum'):
            for param in model.quantum.parameters():
                param.requires_grad = False
                
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        
        start_time = time.time()
        seed_history = []
        
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad()
            xt_f = sample_interior(N_F)
            xt_i, y_i = sample_initial(N_I)
            xt_b, y_b = sample_boundary(N_B)
            
            out_f = model(xt_f)
            if use_aux:
                u, ux_hat = out_f[0], out_f[1]
                r_f, r_c = pde_residual_aux(u, ux_hat, xt_f)
                loss_c = mse(r_c, torch.zeros_like(r_c))
            else:
                u = out_f[0] if isinstance(out_f, tuple) else out_f
                r_f = pde_residual_direct(u, xt_f)
                loss_c = torch.tensor(0.0, device=DEVICE)
                
            loss_f = mse(r_f, torch.zeros_like(r_f))
            
            out_i, out_b = model(xt_i), model(xt_b)
            u_i = out_i[0] if isinstance(out_i, tuple) else out_i
            u_b = out_b[0] if isinstance(out_b, tuple) else out_b
            
            loss_i = mse(u_i, y_i)
            loss_b = mse(u_b, y_b)
            
            loss = 1.0 * loss_f + 0.1 * loss_c + 10.0 * loss_i + 1.0 * loss_b
            loss.backward()
            optimizer.step()
            seed_history.append(loss.item())
            
        metrics["time"].append(time.time() - start_time)
        metrics["pde"].append(loss_f.item())
        metrics["l2"].append(evaluate_on_grid(model))
        metrics["losses"].append(seed_history)
        metrics["models"][seed] = model 
        
    return {
        "l2_mean": np.mean(metrics["l2"]), "l2_std": np.std(metrics["l2"]),
        "pde_mean": np.mean(metrics["pde"]), "pde_std": np.std(metrics["pde"]),
        "time_mean": np.mean(metrics["time"]), "time_std": np.std(metrics["time"]),
        "avg_loss_curve": np.mean(metrics["losses"], axis=0),
        "models": metrics["models"],
        "params": metrics["params"]
    }

# ==========================================
# 6. RUN EXPERIMENTS
# ==========================================
experiments = {
    "MerLin QPINN with trainable quantum layer":  {"fn": lambda: MerlinHeatQPINN(), "aux": True, "freeze": False},
    "MerLin QPINN with frozen quantum layer":     {"fn": lambda: MerlinHeatQPINN(), "aux": True, "freeze": True},
    "MerLin QPINN with classical linear layer":   {"fn": lambda: SurrogateClassicalPINN(), "aux": True, "freeze": False},
    "Same-parameter MLP baseline":                {"fn": lambda: MLP_AuxPINN(hidden_dim=10, depth=3), "aux": True, "freeze": False},
    "Auxiliary-derivative MLP PINN":              {"fn": lambda: MLP_AuxPINN(hidden_dim=32, depth=4), "aux": True, "freeze": False},
    "Direct-derivative MLP PINN":                 {"fn": lambda: MLP_PINN(hidden_dim=32, depth=4), "aux": False, "freeze": False},
}

results = {}
print("Starting rigorous ablation tests...")
for name, config in experiments.items():
    print(f"Training: {name}...")
    results[name] = train_and_evaluate(config["fn"], config["aux"], config["freeze"])

# ==========================================
# 7. SAVE DATA & PLOTS
# ==========================================
names = list(results.keys())
l2_means = [results[n]["l2_mean"] for n in names]
l2_stds = [results[n]["l2_std"] for n in names]
pde_means = [results[n]["pde_mean"] for n in names]
pde_stds = [results[n]["pde_std"] for n in names]
param_counts = [results[n]["params"] for n in names]
time_means = [results[n]["time_mean"] for n in names]

# Save text file
txt_path = os.path.join(DATA_DIR, "benchmark_results.txt")
with open(txt_path, "w") as f:
    f.write("=== HACKATHON BENCHMARK RESULTS ===\n")
    f.write(f"Seeds: {SEEDS} | Epochs: {EPOCHS} | LR: {LR}\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'Experiment Name':<45} | {'Params':<6} | {'Rel L2 Error':<12} | {'PDE Residual':<12} | {'Time (s)'}\n")
    f.write("-" * 110 + "\n")
    for i, name in enumerate(names):
        f.write(f"{name:<45} | {param_counts[i]:<6} | {l2_means[i]:.4e}   | {pde_means[i]:.4e}   | {time_means[i]:.2f}\n")

# PLOT 1: Relative L2 Error
plt.figure(figsize=(14, 7))
bars = plt.bar(names, l2_means, yerr=l2_stds, capsize=5, color='skyblue', edgecolor='black')
plt.ylabel('Relative L2 Error (Mean ± Std)')
plt.title(f'Benchmarking L2 Error across {len(SEEDS)} Seeds\nEpochs: {EPOCHS}')
plt.xticks(rotation=45, ha='right')
for bar, p_count in zip(bars, param_counts):
    plt.text(bar.get_x() + bar.get_width()/2, 0, f"Params: {p_count}", ha='center', va='bottom', rotation=90, color='black')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "benchmark_l2_error.png")) 
plt.close()

# PLOT 2: Final PDE Residual
plt.figure(figsize=(14, 7))
bars = plt.bar(names, pde_means, yerr=pde_stds, capsize=5, color='salmon', edgecolor='black')
plt.yscale('log')
plt.ylabel('PDE Residual (Mean ± Std) [Log Scale]')
plt.title(f'Benchmarking PDE Residual across {len(SEEDS)} Seeds\nEpochs: {EPOCHS}')
plt.xticks(rotation=45, ha='right')
for bar, p_count in zip(bars, param_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.1, f"Params: {p_count}", ha='center', va='bottom', rotation=90, color='black')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "benchmark_pde_residual.png"))
plt.close()

# PLOT 3: Loss Curves
plt.figure(figsize=(10, 6))
for name in names:
    plt.semilogy(results[name]["avg_loss_curve"], label=name)
plt.xlabel('Epoch')
plt.ylabel('Total Loss (Log Scale)')
plt.title('Average Training Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "benchmark_loss_curves.png"))
plt.close()

# ==========================================
# 8. PLOT PDE HEATMAPS PER SEED
# ==========================================
def plot_pde_heatmaps(model, title_prefix, filename, seed, epochs, params):
    nx, nt = 100, 100
    x = torch.linspace(0, 1, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
    t = torch.linspace(0, 1, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    with torch.no_grad():
        out = model(xt_grid)
        u_pred = out[0] if isinstance(out, tuple) else out
        u_pred = u_pred.reshape(nx, nt).cpu()
        u_true = exact_u(xt_grid[:, 0:1], xt_grid[:, 1:2]).reshape(nx, nt).cpu()

    error = torch.abs(u_pred - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{title_prefix}\nSeed: {seed} | Epochs: {epochs} | Parameters: {params}", fontsize=14, fontweight='bold')

    c1 = axes[0].contourf(T.numpy(), X.numpy(), u_true.numpy(), levels=50, cmap='viridis')
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("Exact Solution")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")

    c2 = axes[1].contourf(T.numpy(), X.numpy(), u_pred.numpy(), levels=50, cmap='viridis')
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")

    c3 = axes[2].contourf(T.numpy(), X.numpy(), error.numpy(), levels=50, cmap='magma')
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    plt.savefig(filename, dpi=300)
    plt.close() 

for name in names:
    params = results[name]["params"]
    
    for current_seed in SEEDS:
        trained_model = results[name]["models"][current_seed]
        
        seed_dir = os.path.join(DATA_DIR, f"seed_{current_seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        safe_filename = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        file_path = os.path.join(seed_dir, f"heatmap_{safe_filename}.png")
        
        plot_pde_heatmaps(trained_model, title_prefix=name, filename=file_path, seed=current_seed, epochs=EPOCHS, params=params)

print(f"\nExecution complete. All results, images, and heatmaps saved to the '{DATA_DIR}' directory.")