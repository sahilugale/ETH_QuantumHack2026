````markdown
# ETH Quantum Hackathon 2026 — Quandela Challenge

## Are Quantum PINNs Actually Doing Anything?

This repository contains our implementation, benchmarking, and analysis for the Quandela Challenge at the ETH Quantum Hackathon 2026.

The challenge investigates whether Quantum Physics-Informed Neural Networks (QPINNs) provide meaningful advantages over classical PINNs when solving partial differential equations (PDEs). Rather than assuming a quantum advantage, the goal is to rigorously reproduce, benchmark, analyze, and question the role of the quantum component itself. :contentReference[oaicite:0]{index=0}

---

# Team

- Chayma Faraji
- Iva Sanwald
- Sahil Ugale

---

# Challenge Overview

Physics-Informed Neural Networks (PINNs) solve PDEs by embedding physical laws directly into the training objective. Recent work proposes hybrid quantum-classical versions of PINNs using photonic quantum circuits.

This project investigates:

- Whether QPINNs outperform classical PINNs under fair comparisons
- Whether the proposed quantum circuits are genuinely quantum or efficiently classically simulable
- The impact of circuit depth, architecture, and auxiliary derivative formulations
- Generalization behavior outside training regimes
- Tradeoffs between performance and computational/energetic cost

The core research question is:

> Does the quantum component help, and if so, why? :contentReference[oaicite:1]{index=1}

---

# Objectives

## Phase 1 — Reproduction

We reproduce the baseline QPINN implementation described in the challenge resources and compare it against a classical PINN baseline under comparable parameter budgets.

Key investigations:
- Accuracy comparison
- PDE residual analysis
- Training stability
- Scaling with circuit depth
- Classical simulability of the quantum model

## Phase 2 — Going Beyond

We extend the baseline by:
- Designing physically motivated photonic circuits
- Testing deeper and more expressive quantum architectures
- Removing auxiliary derivative formulations
- Computing second derivatives directly
- Evaluating out-of-distribution generalization
- Analyzing computational and energetic costs

---

# Repository Structure

```text
ETH_QuantumHack2026/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── classical/
│   ├── quantum/
│   ├── pinn/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── notebooks/
├── outputs/
├── assets/
├── figures/
└── presentations/
````

---

# Setup

## Clone Repository

```bash
git clone git@github.com:sahilugale/ETH_QuantumHack2026.git
cd ETH_QuantumHack2026
```

## Create Environment

```bash
python3 -m venv quandela_hack
source quandela_hack/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Experiments

## Classical PINN

```bash
python src/classical/train_classical_pinn.py
```

## Quantum PINN

```bash
python src/quantum/train_quantum_pinn.py
```

---

# Metrics

We evaluate models using:

* Relative L2 Error
* PDE Residuals
* Training Stability
* Parameter Efficiency
* Generalization Performance
* Resource/Energy Cost

---

# Planned Experiments

* Classical PINN baseline
* Reproduction of QPINN architecture
* Circuit depth scaling
* Gaussian vs non-Gaussian photonic circuits
* Direct second derivative computation
* Ablation studies
* Out-of-distribution testing
* Hardware inference experiments (if available)

---

# Tools & Frameworks

* Python
* PyTorch
* MerLin
* Quandela Photonic Framework
* NumPy
* Matplotlib
* Jupyter

---

# Deliverables

According to the challenge requirements, this repository aims to provide: 

* Working QPINN implementation
* Classical baseline comparison
* Quantitative benchmarking
* Visualizations and analysis
* Final presentation and conclusions

---

# Evaluation Philosophy

This project focuses on rigorous investigation rather than forcing a quantum advantage claim.

A meaningful outcome may include:

* No observable quantum advantage
* Improved stability without accuracy gains
* Better inductive biases in specific regimes
* Identification of classically simulable quantum models

The emphasis is on evidence, reproducibility, and careful interpretation. 

---

# References

* Quandela ETH Quantum Hackathon 2026 Challenge Description 
* Quantum physics informed neural networks for multi-variable PDEs
* MerLin Documentation
* Quandela Photonic Cloud Platform

```
```
