# Hybrid Collaborative Filtering (UCF + ICF Fusion)
A classical memory-based recommender that combines User-Based CF 
and Item-Based CF via a weighted fusion parameter (gamma).

## Approach
Instead of choosing between user similarity or item similarity, 
This model blends both predictions:
```
final_prediction = γ × UCF_prediction + (1 − γ) × ICF_prediction
```

- **UCF Tower:** Pearson correlation-based user similarity (top-k=30 neighbors)
- **ICF Tower:** Cosine similarity-based item similarity (top-k=10 neighbors)
- **Fusion:** Weighted blend with γ = 0.5

## Results (5-Fold Cross Validation)

| Model          | Avg MAE | Avg NMAE |
|----------------|---------|----------|
|  Hybrid CF   | 0.7139  | **0.1785** |
| User-Based CF  | 0.7439  | 0.1860   |
| Item-Based CF  | 0.8871  | 0.2242   |

### Per-Fold Breakdown (Hybrid Model)

| Fold | MAE    | NMAE   |
|------|--------|--------|
| u1   | 0.7228 | 0.1807 |
| u2   | 0.7112 | 0.1778 |
| u3   | 0.7110 | 0.1778 |
| u4   | 0.7094 | 0.1774 |
| u5   | 0.7153 | 0.1788 |
| **Avg** | **0.7139** | **0.1785** |

> The Hybrid model achieves **4.0% lower NMAE** than UCF and 
> **20.4% lower NMAE** than ICF alone.

## How to Run
```bash
pip install numpy pandas scikit-learn
jupyter notebook final.ipynb
```
Update the dataset path inside the notebook to point to your local `ml-100k/` folder.
