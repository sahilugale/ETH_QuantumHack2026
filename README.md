# ETH Quantum Hackathon 2026 — Quandela Challenge

## Are Quantum PINNs Actually Doing Anything?

This repository contains our implementation, benchmarking, and analysis for the Quandela Challenge at the ETH Quantum Hackathon 2026.

The challenge investigates whether Quantum Physics-Informed Neural Networks (QPINNs) provide meaningful advantages over classical PINNs when solving partial differential equations (PDEs).

Rather than assuming a quantum advantage, the goal is to rigorously reproduce, benchmark, analyze, and question the role of the quantum component itself.

---

# Team

- Chayma Faraji
- Iva Sanwald
- Sahil Ugale

---

# Challenge Overview

Physics-Informed Neural Networks (PINNs) solve PDEs by embedding physical laws directly into the training objective. 

Recent work proposes hybrid quantum-classical versions of PINNs using photonic quantum circuits.

This project investigates:
- Whether QPINNs outperform classical PINNs under fair, parameter-matched comparisons.
- Whether the proposed quantum circuits are genuinely quantum or efficiently classically simulable.
- The impact of circuit depth, architecture, and auxiliary derivative formulations.
- Generalization behavior outside training regimes.
- Tradeoffs between performance (RMSE) and computational cost (Training Time).

The central research question is:
> Does the quantum component help, and if so, why?

---

# Objectives & Implemented Methodology

## Phase 1 — Reproduction & Classical Baselines
We reproduced the baseline QPINN implementation described in the challenge resources and established rigorous classical baselines (e.g., Classical Direct PINN, Classical Auxiliary Trick PINN). 

## Phase 2 — Rigorous Statistical Benchmarking
To ensure results are not artifacts of lucky initialization, all major experiments are evaluated across multiple random seeds (e.g., `1234`, `42`, `100`, `999`, `2026`), reporting the **Mean** and **Standard Deviation** for all metrics.

## Executed Experiments

1. **Ablation Studies (Quantum vs. Classical Architectures)**
   - **Model A (QPINN Active):** Full hybrid quantum-classical network.
   - **Model Frozen Quantum:** QPINN where the quantum parameters are frozen at initialization to test if the quantum layer acts merely as a randomized feature projector.
   - **Model B (Classical Trick):** Purely classical PINN using the auxiliary derivative trick.
   - **Model C (Classical Direct):** Purely classical PINN using PyTorch's direct higher-order automatic differentiation.

2. **Parameter Ratio Tests**
   Investigating how the distribution of parameters between the classical and quantum layers affects performance:
   - **Classical Params Constant Ratio:** Varying quantum size while keeping classical parameter count fixed.
   - **Hidden Layer Constant Ratio:** Varying quantum size while keeping the classical hidden layer dimensions fixed.
   - **Total Params Constant Ratio:** Varying the internal classical/quantum ratio while strictly holding the total network parameter count constant.
   *(Note: All ratio tests were run in both active and "frozen quantum" modes for complete ablation).*

3. **Performance & Time Scaling (QPINN vs. Classical Baseline)**
   - Scaling the quantum `feature_size` (depth/parameters) of the QPINN and plotting the resulting Mean Squared Error (MSE/RMSE) and Training Time against the optimal Classical baseline. This answers whether investing parameter budget into the quantum circuit yields diminishing or accelerating returns compared to classical depth.

---

# Metrics

We evaluate models using:

- **Interpolation Error (Interior MSE/RMSE):** Performance within the training domain.
- **Extrapolation Error (Exterior MSE/RMSE):** Generalization performance outside the training domain.
- **Relative L2 Error & L-infinity Norms**
- **Training Time (seconds):** Computational cost of the architecture.
- **Total vs. Trainable Parameters:** To ensure budget-matched fairness.

---

# Tools & Frameworks

- **Python**
- **PyTorch**
- **MerLin** (merlinquantum)
- **Quandela Photonic Framework**
- **NumPy & Pandas**
- **Matplotlib** (for aggregated statistical visualizations)
- **Jupyter**

---

# Deliverables

According to the challenge requirements, this repository provides:

- A working QPINN implementation.
- Mathematically parameter-matched classical baselines.
- Quantitative statistical benchmarking (multi-seed aggregations).
- Visualizations and analysis (Ratio curves, Scaling plots, Convergence histories).
- Final presentation and conclusions.

---

# Evaluation Philosophy

This project focuses on rigorous investigation rather than forcing a quantum advantage claim.

A meaningful outcome may include:
- No observable quantum advantage.
- Improved stability without accuracy gains.
- Better inductive biases in specific regimes.
- Identification of classically simulable quantum models.

The emphasis is on evidence, reproducibility, and careful interpretation.

---

# References

- Quandela ETH Quantum Hackathon 2026 Challenge Description
- *Quantum physics informed neural networks for multi-variable PDEs*
- MerLin Documentation
- Quandela Photonic Cloud Platform