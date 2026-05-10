
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
file_path = "/local/sugale/hackathons/ETH_QuantumHack2026/submission/phase1_ablation_results/ablation_metrics_summary.csv"
df = pd.read_csv(file_path)

# 2. Calculate MSE from RMSE
# The dataset contains RMSE, so we square it to get MSE
df['MSE_Int'] = df['RMSE_Int'] ** 2
df['MSE_Ext'] = df['RMSE_Ext'] ** 2

# 3. Group by Model_Name and calculate mean and std across seeds
grouped = df.groupby('Model_Name').agg(
    MSE_Int_mean=('MSE_Int', 'mean'),
    MSE_Int_std=('MSE_Int', 'std'),
    MSE_Ext_mean=('MSE_Ext', 'mean'),
    MSE_Ext_std=('MSE_Ext', 'std')
).reset_index()

# 4. Plotting
x = np.arange(len(grouped['Model_Name']))
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Interior MSE
bars1 = ax.bar(
    x - width/2, 
    grouped['MSE_Int_mean'], 
    width, 
    yerr=grouped['MSE_Int_std'], 
    capsize=5, 
    label='Interior MSE', 
    color='skyblue',
    edgecolor='black',
    alpha=0.8
)

# Plot Exterior MSE
bars2 = ax.bar(
    x + width/2, 
    grouped['MSE_Ext_mean'], 
    width, 
    yerr=grouped['MSE_Ext_std'], 
    capsize=5, 
    label='Exterior MSE', 
    color='lightcoral',
    edgecolor='black',
    alpha=0.8
)

# Formatting the plot
ax.set_ylabel('Mean Square Error (MSE)')
ax.set_xlabel('Model')
ax.set_title('Mean MSE across Seeds by Model')
ax.set_xticks(x)
ax.set_xticklabels(grouped['Model_Name'], rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save/show
plt.tight_layout()
plt.savefig("/local/sugale/hackathons/ETH_QuantumHack2026/submission/phase1_ablation_results/mean_mse_histogram.png", dpi=300)
plt.show()

print("Plot saved successfully to 'mean_mse_histogram.png'")
# %%
