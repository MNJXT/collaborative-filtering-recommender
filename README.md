# Collaborative Filtering — Movie Recommender Systems

Two assignments exploring classical and deep learning approaches to 
collaborative filtering on the **MovieLens 100K** dataset.

## Projects:

### 1. [`Hybrid-cf/`](./Hybrid-cf)
**Hybrid Collaborative Filtering (UCF + ICF Fusion)**  
A memory-based approach that fuses User-Based and Item-Based CF 
using cosine similarity and a weighted gamma parameter.  
> Hybrid NMAE: **0.1785** — outperforms both UCF (0.1860) and ICF (0.2242)

### 2. [`Jncf-deep/`](./Jncf-deep)
**J-NCF: Joint Neural Collaborative Filtering**  
A deep learning model with dual-tower architecture trained using 
CrossEntropyLoss over 5-fold cross-validation.  
> Average 5-Fold Test Accuracy: **46.11%**

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 
100,000 ratings from 943 users on 1,682 movies, with 5 pre-defined 
80/20 train/test splits.

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-grey?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-green?logo=pandas)
