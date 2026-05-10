# %%
import os
import math
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
    def __init__(self, feature_size=10, quantum_output_size=10, hidden=10):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, feature_size),
        )

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

        u = x * (1.0 - x) * q_u
        return u, ux_hat

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

# Loop configurations from the provided table
# (feature_size, hidden_layer_size)
configs = [
    (10, 8),
    (8, 13),
    (6, 19),
    (4, 28),
    (2, 42)
]

# Define seeds for reproducibility and statistical averaging
seeds = [1234, 42, 100]

# Dictionary to track RMSEs across seeds. Key: config tuple, Value: list of RMSEs
seed_rmses = {config: [] for config in configs}

for seed in seeds:
    print(f"\n{'*'*50}")
    print(f"Starting SEED: {seed}")
    print(f"{'*'*50}")
    
    # Set the seed for this iteration
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a dynamic folder name specific to the frozen quantum runs
    base_folder = f"data_total_params_const_frozen_test_seed_{seed}"
    os.makedirs(base_folder, exist_ok=True)
    
    # Output file for metrics
    metrics_file = open(f"{base_folder}/metrics_summary_frozen.txt", "w")
    metrics_file.write("Feature\tHL\tTotal_Params\tTrainable_Params\tQuantum_Params\tRatio\tRMSE\tMAE\tL_inf\tNMSE\n")

    # Lists to track aggregated metrics for the final plot within this seed
    all_ratios = []
    all_rmses = []

    for feature, hl in configs:
        print(f"\n{'='*50}")
        print(f"Running Config: Feature Size={feature}, Quantum Output={feature}, HL={hl} (Seed: {seed} - Frozen Q)")
        print(f"{'='*50}")
        
        # Initialize Model
        model = MerlinHeatQPINN(feature_size=feature, quantum_output_size=feature, hidden=hl).to(device=device, dtype=dtype)

        for p in model.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype)

        # Freeze the quantum parameters
        for p in model.quantum.parameters():
            p.requires_grad = False

        # Parameter Counting
        total_params = sum(p.numel() for p in model.parameters())
        quantum_params = sum(p.numel() for p in model.quantum.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        classical_params = total_params - quantum_params
        ratio = quantum_params / total_params if total_params > 0 else 0
        
        all_ratios.append(ratio)

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Quantum Parameters (Frozen): {quantum_params}")
        print(f"Classical Parameters: {classical_params}")
        print(f"Ratio (Quantum/Total): {ratio:.4f}")

        # Loss weights and training settings
        n_f = 64      
        n_i = 64      
        n_b = 64      

        epochs = 300
        lr = 1e-2
        lambda_f = 1.0
        lambda_c = 0.1
        lambda_i = 10.0
        lambda_b = 1.0

        # Pass only trainable parameters to the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        mse = nn.MSELoss()
        history = []

        # Training Loop
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            xt_f = sample_interior(n_f)
            xt_i = sample_initial(n_i)
            xt_b = sample_boundary(n_b)

            r_f, r_c = pde_and_consistency_residuals(model, xt_f)
            loss_f = mse(r_f, torch.zeros_like(r_f))
            loss_c = mse(r_c, torch.zeros_like(r_c))

            u_i, _ = model(xt_i)
            x_i = xt_i[:, 0:1]
            t_i = xt_i[:, 1:2]
            loss_i = mse(u_i, exact_u(x_i, t_i))

            u_b, _ = model(xt_b)
            loss_b = mse(u_b, torch.zeros_like(u_b))

            loss = lambda_f * loss_f + lambda_c * loss_c + lambda_i * loss_i + lambda_b * loss_b
            loss.backward()
            optimizer.step()

            history.append([loss.item(), loss_f.item(), loss_c.item(), loss_i.item(), loss_b.item()])

            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"epoch {epoch:04d} | loss={loss.item():.4e} | "
                    f"pde={loss_f.item():.2e} | cons={loss_c.item():.2e} | "
                    f"ic={loss_i.item():.2e} | bc={loss_b.item():.2e}"
                )

        # Save History Plot
        hist = np.array(history)
        np.savetxt(f"{base_folder}/training_history_frozen_F{feature}_HL{hl}.csv", hist, delimiter=",", header="Total,PDE,Consistency,Initial,Boundary", comments="")
        
        plt.figure(figsize=(7, 4))
        plt.semilogy(hist[:, 0], label="total")
        plt.semilogy(hist[:, 1], label="PDE")
        plt.semilogy(hist[:, 2], label="consistency")
        plt.semilogy(hist[:, 3], label="initial")
        plt.semilogy(hist[:, 4], label="boundary")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"Training Surrogate (F={feature}, HL={hl} - Frozen Q)")
        plt.savefig(f"{base_folder}/training_loss_frozen_F{feature}_HL{hl}.png", bbox_inches='tight')
        plt.close()

        # Evaluation
        nx, nt = 60, 60
        x = torch.linspace(0, 1, nx, device=device, dtype=dtype).reshape(-1, 1)
        t = torch.linspace(0, 1, nt, device=device, dtype=dtype).reshape(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

        with torch.no_grad():
            U_pred, UX_hat = model(xt_grid)
            U_pred = U_pred.reshape(nx, nt).cpu()
            U_true = exact_u(xt_grid[:, 0:1], xt_grid[:, 1:2]).reshape(nx, nt).cpu()

        # Save Solution Plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im0 = axes[0].imshow(U_true.numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        axes[0].set_title("Exact Solution")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("x")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(U_pred.numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        axes[1].set_title("QPINN Prediction (Frozen Q)")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("x")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow((U_pred - U_true).numpy(), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        axes[2].set_title("Prediction Error")
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("x")
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"{base_folder}/solutions_frozen_F{feature}_HL{hl}.png", bbox_inches='tight')
        plt.close()

        # Metrics
        rmse = torch.sqrt(torch.mean((U_pred - U_true)**2))
        mae = torch.mean(torch.abs(U_pred - U_true))
        l_infinity = torch.max(torch.abs(U_pred - U_true))
        nmse = torch.sum((U_true - U_pred)**2) / torch.sum(U_true**2)

        # Append RMSE to the tracking dictionaries
        seed_rmses[(feature, hl)].append(float(rmse))
        all_rmses.append(float(rmse))
        
        print(f"RMSE: {float(rmse):.4e}")
        
        # Save to metrics file
        metrics_file.write(f"{feature}\t{hl}\t{total_params}\t{trainable_params}\t{quantum_params}\t{ratio:.4f}\t{float(rmse):.4e}\t{float(mae):.4e}\t{float(l_infinity):.4e}\t{float(nmse):.4e}\n")

    metrics_file.close()

    # Generate plot of Ratio vs RMSE for the current seed
    plt.figure(figsize=(7, 5))
    plt.plot(all_ratios, all_rmses, marker='o', linestyle='-', color='b')
    plt.xlabel("Ratio (Quantum Gates / Total Parameters)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Quantum Parameter Ratio (Seed {seed} - Frozen Q)")
    plt.grid(True)

    # Annotate points with Feature size
    for i, (feature, hl) in enumerate(configs):
        plt.annotate(f"F={feature}", (all_ratios[i], all_rmses[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(f"{base_folder}/rmse_vs_ratio_frozen.png", bbox_inches='tight')
    plt.close()

# %%

# ==========================================
# FINAL AGGREGATION ACROSS ALL SEEDS
# ==========================================
print(f"\n{'='*50}")
print("All seeds completed. Generating aggregated metrics plot.")
print(f"{'='*50}")

# Prepare data for the plot
labels = [f"F={f}\nHL={hl}" for f, hl in configs]
means = [np.mean(seed_rmses[config]) for config in configs]
stds = [np.std(seed_rmses[config]) for config in configs]

x_pos = np.arange(len(labels))

plt.figure(figsize=(10, 6))
# Create a bar plot (histogram) with yerr for standard deviation
plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color='skyblue')

plt.ylabel('RMSE')
plt.xlabel('Configuration (Feature, Hidden Layer)')
plt.title('Mean RMSE across Seeds with Standard Deviation\n(Total Params Const - Frozen Quantum)')
plt.xticks(x_pos, labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the final aggregated plot in the current working directory
plt.tight_layout()
plt.savefig("aggregated_rmse_total_params_const_ratio_frozen.png", dpi=300)
plt.show()

print("\nAggregated plot saved as 'aggregated_rmse_total_params_const_ratio_frozen.png'.")