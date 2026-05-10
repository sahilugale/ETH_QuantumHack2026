import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from shared_utils import (
    DEVICE, DTYPE, ALPHA, count_parameters, exact_u, gradients,
    sample_interior, sample_initial, sample_boundary,
    ModelA_QPINN_Trick, ModelB_Classical_Trick, ModelC_Classical_Direct
)

SEEDS = [1234, 42, 999]
DATA_DIR = "phase1_ablation_results"
os.makedirs(DATA_DIR, exist_ok=True)

EPOCHS = 300
LR = 1e-2
N_F, N_I, N_B = 64, 64, 64
torch.set_default_dtype(DTYPE)

def calc_metrics(pred, true):
    """Calculates all 5 required accuracy metrics."""
    rmse = float(torch.sqrt(torch.mean((pred - true)**2)))
    mae = float(torch.mean(torch.abs(pred - true)))
    l_inf = float(torch.max(torch.abs(pred - true)))
    nmse = float(torch.sum((true - pred)**2) / torch.sum(true**2))
    rel_l2 = float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))
    return rel_l2, rmse, mae, l_inf, nmse

def train_model(model, name, use_trick, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Training {name} [Seed {seed}] (Params: {count_parameters(model)})")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    mse = nn.MSELoss()
    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        xt_f, xt_i, xt_b = sample_interior(N_F), sample_initial(N_I), sample_boundary(N_B)

        if use_trick:
            u, ux_hat = model(xt_f)
            grad_u = gradients(u, xt_f)
            pde_res = grad_u[:, 1:2] - ALPHA * gradients(ux_hat, xt_f)[:, 0:1]
            cons_res = grad_u[:, 0:1] - ux_hat
            loss_pde = mse(pde_res, torch.zeros_like(pde_res))
            loss_cons = mse(cons_res, torch.zeros_like(cons_res))
            u_i, _ = model(xt_i)
            u_b, _ = model(xt_b)
        else:
            u = model(xt_f)
            grad_u = gradients(u, xt_f)
            pde_res = grad_u[:, 1:2] - ALPHA * gradients(grad_u[:, 0:1], xt_f)[:, 0:1]
            loss_pde = mse(pde_res, torch.zeros_like(pde_res))
            loss_cons = torch.tensor(0.0, device=DEVICE)
            u_i, u_b = model(xt_i), model(xt_b)

        loss_ic = mse(u_i, exact_u(xt_i[:, 0:1], xt_i[:, 1:2]))
        loss_bc = mse(u_b, torch.zeros_like(u_b))
        loss = 1.0 * loss_pde + 10.0 * loss_ic + 1.0 * loss_bc + (0.1 * loss_cons if use_trick else 0.0)

        loss.backward()
        optimizer.step()
        history.append([loss.item(), loss_pde.item(), loss_cons.item(), loss_ic.item(), loss_bc.item()])

    return np.array(history), time.time() - start_time

metrics_file_path = os.path.join(DATA_DIR, "ablation_metrics_summary.csv")
with open(metrics_file_path, "w") as f:
    f.write("Seed,Model_Name,Params,Train_Time_s,Rel_L2_Int,RMSE_Int,MAE_Int,Linf_Int,NMSE_Int,Rel_L2_Ext,RMSE_Ext,MAE_Ext,Linf_Ext,NMSE_Ext\n")

aggregate_times = {"Model_A_Quantum_Active": [], "Model_Frozen_Quantum": [], "Model_B_Classical_Trick": [], "Model_C_Classical_Direct": []}

for seed in SEEDS:
    torch.manual_seed(seed)
    
    model_a = ModelA_QPINN_Trick().to(DEVICE, dtype=DTYPE)
    model_frozen = ModelA_QPINN_Trick().to(DEVICE, dtype=DTYPE)
    
    for p in model_frozen.quantum.parameters():
        p.requires_grad = False

    models = {
        "Model_A_Quantum_Active": (model_a, True),
        "Model_Frozen_Quantum": (model_frozen, True),
        "Model_B_Classical_Trick": (ModelB_Classical_Trick().to(DEVICE, dtype=DTYPE), True),
        "Model_C_Classical_Direct": (ModelC_Classical_Direct().to(DEVICE, dtype=DTYPE), False),
    }

    for p in models["Model_A_Quantum_Active"][0].parameters():
        if p.is_floating_point(): p.data = p.data.to(DTYPE)
    for p in models["Model_Frozen_Quantum"][0].parameters():
        if p.is_floating_point(): p.data = p.data.to(DTYPE)

    results = {}
    for name, (model, use_trick) in models.items():
        history, train_time = train_model(model, name, use_trick, seed)
        aggregate_times[name].append(train_time)
        
        nx, nt = 60, 60
        x, t = torch.linspace(0, 1, nx, device=DEVICE), torch.linspace(0, 1, nt, device=DEVICE)
        X, T = torch.meshgrid(x, t, indexing="ij")
        xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        t_ext = torch.linspace(1.0, 2.0, nt, device=DEVICE)
        X_ext, T_ext = torch.meshgrid(x, t_ext, indexing="ij")
        xt_ext_grid = torch.stack([X_ext.reshape(-1), T_ext.reshape(-1)], dim=1)

        with torch.no_grad():
            U_pred = model(xt_grid)[0].reshape(nx, nt).cpu() if use_trick else model(xt_grid).reshape(nx, nt).cpu()
            U_true = exact_u(xt_grid[:, 0:1], xt_grid[:, 1:2]).reshape(nx, nt).cpu()
            
            U_pred_ext = model(xt_ext_grid)[0].reshape(nx, nt).cpu() if use_trick else model(xt_ext_grid).reshape(nx, nt).cpu()
            U_true_ext = exact_u(xt_ext_grid[:, 0:1], xt_ext_grid[:, 1:2]).reshape(nx, nt).cpu()
            
        # Calculate ALL metrics for BOTH domains
        rel_l2_i, rmse_i, mae_i, linf_i, nmse_i = calc_metrics(U_pred, U_true)
        rel_l2_e, rmse_e, mae_e, linf_e, nmse_e = calc_metrics(U_pred_ext, U_true_ext)
        
        results[name] = {"history": history, "train_time": train_time, "rmse": rmse_i}
        
        with open(metrics_file_path, "a") as f:
            f.write(f"{seed},{name},{count_parameters(model)},{train_time:.2f},{rel_l2_i:.4e},{rmse_i:.4e},{mae_i:.4e},{linf_i:.4e},{nmse_i:.4e},{rel_l2_e:.4e},{rmse_e:.4e},{mae_e:.4e},{linf_e:.4e},{nmse_e:.4e}\n")
            
        # Plot Heatmaps (Interpolation)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].contourf(T.numpy(), X.numpy(), U_true.numpy(), levels=50, cmap='viridis'); axes[0].set_title("Exact Solution")
        im1 = axes[1].contourf(T.numpy(), X.numpy(), U_pred.numpy(), levels=50, cmap='viridis'); axes[1].set_title(f"Prediction ({name})")
        im2 = axes[2].contourf(T.numpy(), X.numpy(), torch.abs(U_pred - U_true).numpy(), levels=50, cmap='magma'); axes[2].set_title(f"Error (RMSE: {rmse_i:.4e})")
        plt.tight_layout(); plt.savefig(os.path.join(DATA_DIR, f"{name}_seed{seed}_heatmaps.png"), dpi=300); plt.close()

    # Convergence
    plt.figure(figsize=(8, 5))
    for name in results:
        plt.semilogy(results[name]["history"][:, 0], label=f"{name.replace('_', ' ')} (RMSE: {results[name]['rmse']:.2e})")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title(f"Convergence Comparison (Seed {seed})")
    plt.legend(); plt.grid(alpha=0.5); plt.savefig(os.path.join(DATA_DIR, f"convergence_seed{seed}.png"), dpi=300); plt.close()

# Average Time Plot
names = list(aggregate_times.keys())
avg_times = [np.mean(aggregate_times[n]) for n in names]
std_times = [np.std(aggregate_times[n]) for n in names]

plt.figure(figsize=(10, 6))
bars = plt.bar([n.replace('_', '\n') for n in names], avg_times, yerr=std_times, capsize=5, color=['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c'], alpha=0.8)
plt.ylabel("Average Training Time (seconds)")
plt.title("Computational Cost Comparison (Averaged across 3 seeds)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}s', va='bottom', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(DATA_DIR, "average_training_time_comparison.png"), dpi=300); plt.close()