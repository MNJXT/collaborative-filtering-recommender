# 1-Bit Matrix Completion via Proximal Gradient Methods (FISTA + SVT)
Recovery of a low-rank latent preference matrix from sparse binary 
observations, using a convex relaxation with logistic surrogate loss 
and Singular Value Thresholding (SVT).

## Problem
Given a partially observed binary matrix Y ∈ {-1, +1} and a mask R,
recover the unknown low-rank latent matrix X* that generated the signs.
The original problem is NP-hard (ℓ0 loss). This project solves a 
convex relaxation by replacing ℓ0 with logistic surrogate loss:
min  Σ R_ij · log(1 + exp(-Y_ij · X_ij))  +  λ ||X||*

## Algorithm: FISTA with SVT
Each iteration has two steps:
**Forward step** — gradient descent on the smooth logistic loss:
Z_k = W_k − η · ∇g(W_k)
**Backward step** — proximal operator = Singular Value Thresholding:
X_{k+1} = U · diag(max(σ_i − τ, 0)) · Vᵀ
**Momentum** — Nesterov acceleration (FISTA) for O(1/k²) convergence:
W_{k+1} = X_{k+1} + ((t_k − 1) / t_{k+1}) · (X_{k+1} − X_k)

## Results
| Metric                     | Value        |
|----------------------------|--------------|
| Convergence (iterations) | **37 / 300** |
| Train Sign Accuracy      | **95.60%**   |
| Test Sign Accuracy       | **90.11%**   |
| Recovered Rank           | **5** (exact match to ground truth rank-5)|
| Effective Rank           | **4.81**     |
| Runtime                  | **0.73 sec** |
| Relative Frobenius Error   | 3.70 (expected — binary obs. only encode sign, not scale) |
| Final Objective F(X)       | 4902.67      |

> FISTA converged **8x faster** than the max iteration budget (37 vs 300),
> consistent with the theoretical O(1/k²) convergence guarantee.

## Experimental Setup
- Matrix size: 100 × 100, ground-truth rank = 5
- Observation density: 80% train / 20% held-out test
- λ = 8.5 (enforces rank ≤ 5 via SVT threshold τ = ηλ = 8.5)
- η = 1.0, max iterations = 300, tolerance ε = 1e-4
- Random seed = 42 (reproducible)

## How to Run
```bash
pip install numpy scipy
python 2023309_code.py
```

No dataset download needed — synthetic data is generated automatically.

## Key Hyperparameters
| Parameter     | Value  | Effect                              |
|---------------|--------|-------------------------------------|
| λ (lambda)    | 8.5    | Controls sparsity of singular values|
| η (step size) | 1.0    | Gradient descent step               |
| Max Iter      | 300    | Converged early at 37               |
| Tolerance ε   | 1e-4   | Relative change stopping criterion  |

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-grey?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-blue?logo=scipy)
