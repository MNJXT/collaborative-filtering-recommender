# J-NCF: Joint Neural Collaborative Filtering
A deep learning recommender using a two-tower neural network 
architecture, trained on MovieLens 100K with 5-fold cross-validation.

## Architecture
```
Rating Matrix R (943 × 1682)
         │
  ┌──────┴──────────┐
User Row (v_u)   Item Column (v_i)
  │                  │
DF Network        DF Network
1682→128→64      943→128→64
  │                  │
  └──── Concat ───── ┘
          │ (128-dim)
     DI Network
   128 → 64 → 32 → 5
          │
  Rating Class (1–5 stars)
```
- **Loss:** CrossEntropyLoss (5-class classification)
- **Optimizer:** Adam (lr = 0.0005)
- **Epochs:** 30 per fold

## Results (5-Fold Cross Validation)

| Fold | Best Test Accuracy |
|------|--------------------|
| u1   | 46.64%             |
| u2   | 46.48%             |
| u3   | 46.39%             |
| u4   | 45.93%             |
| u5   | 45.10%             |
| **Average** | **46.11%** |

---

## How to Run
```bash
pip install -r requirements.txt
```
Update `BASE_PATH` in `prediction.py` to point to your `ml-100k/` folder, then:
```bash
python prediction.py
```
## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Learning Rate | 0.0005 |
| Epochs | 30 |
| Embedding Dims | [128, 64] |
| Interaction Dims | [64, 32] |
