
import numpy as np
from scipy.sparse.linalg import svds
import time

def generate_synthetic_data(m=100, n=100, rank=5, p_obs=0.8, seed=42):
    """Generates synthetic low-rank binary matrix completion dataset."""
    np.random.seed(seed)
    
    # 1. Generate underlying low-rank continuous matrix X*
    U_true = np.random.randn(m, rank)
    V_true = np.random.randn(n, rank)
    X_star = U_true @ V_true.T
    
    # Scale X_star to prevent overflow in exponentials and bound infinity norm
    X_star = X_star / np.max(np.abs(X_star))
    
    # 2. Generate observed binary labels Y = sign(X*)
    Y = np.sign(X_star)
    Y = np.where(Y == 0, 1, Y) # Force strict {-1, 1}
    
    # 3. Generate binary mask R for missing entries (Train-Test Split)
    R = np.random.binomial(1, p_obs, size=(m, n))
    
    return X_star, Y, R

def objective_function(X, Y, R, lambda_reg):
    """Computes the smooth logistic objective + nuclear norm penalty."""
    # Logistic loss on observed entries
    loss = np.sum(R * np.log(1 + np.exp(-Y * X)))
    # Nuclear norm computation via SVD
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    nuclear_norm = np.sum(S)
    
    return loss + lambda_reg * nuclear_norm

def binary_matrix_completion(Y, R, lambda_reg=8.5, eta=0.1, max_iter=300, tol=1e-4):
    """
    Implements Proximal Gradient Descent with Singular Value Thresholding.
    Uses Accelerated Nesterov momentum (FISTA formulation) for speed.
    """
    m, n = Y.shape
    X = np.zeros((m, n))
    W = np.zeros((m, n)) # Extrapolated search point for momentum
    t = 1.0 # Momentum parameter
    
    history = {'objective': list(), 'time': list()}
    start_time = time.time()
    
    for k in range(max_iter):
        # --- FORWARD STEP (Gradient of Logistic Loss at extrapolated point W) ---
        exp_term = np.exp(Y * W)
        grad = R * (-Y / (1 + exp_term))
        
        # Gradient Descent update
        Z = W - eta * grad
        
        # --- BACKWARD STEP (Proximal Operator via SVT) ---
        # To optimize, we compute partial SVD assuming rank is relatively small.
        # We compute top k_svd singular values. If rank exceeds this, it falls back to full SVD.
        k_svd = min(m-1, 20) 
        try:
            U, S, Vt = svds(Z, k=k_svd)
        except:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            
        # Apply Soft-Thresholding to singular values
        tau = eta * lambda_reg
        S_thresh = np.maximum(S - tau, 0)
        
        # Reconstruct thresholded matrix
        X_next = (U * S_thresh) @ Vt
        
        # --- MOMENTUM UPDATE (APG/FISTA) ---
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        W = X_next + ((t - 1) / t_next) * (X_next - X)
        
        # --- LOGGING & CONVERGENCE CHECK ---
        obj_val = objective_function(X_next, Y, R, lambda_reg)
        history['objective'].append(obj_val)
        history['time'].append(time.time() - start_time)
        
        # Relative change tolerance
        if np.linalg.norm(X_next - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-8) < tol:
            print(f"Converged at iteration {k+1}")
            X = X_next
            break
            
        X = X_next
        t = t_next
        
    return X, history

def calculate_metrics(X_pred, Y_true, X_star, R):
    """Calculates extended performance metrics."""
    # Predict binary labels by taking the sign of the latent continuous prediction
    Y_pred = np.sign(X_pred)
    Y_pred = np.where(Y_pred == 0, 1, Y_pred)
    
    # 1. Accuracy
    R_test = 1 - R 
    train_accuracy = np.sum(R * (Y_pred == Y_true)) / np.sum(R)
    test_accuracy = np.sum(R_test * (Y_pred == Y_true)) / np.sum(R_test)
    
    # 2. Relative Frobenius Error
    rel_frob_error = np.linalg.norm(X_pred - X_star, 'fro') / np.linalg.norm(X_star, 'fro')
    
    # 3. Recovered Rank and Effective Rank
    _, S, _ = np.linalg.svd(X_pred, full_matrices=False)
    tol = max(X_pred.shape) * np.spacing(max(S)) if len(S) > 0 else 1e-14
    recovered_rank = np.sum(S > tol)
    
    # Effective Rank based on Shannon entropy
    p = S / np.sum(S)
    p = p[p > 0] # Filter out exact zeros for log
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)
    
    return train_accuracy, test_accuracy, recovered_rank, rel_frob_error, effective_rank

# --- Execution Script ---
if __name__ == "__main__":
    m, n, rank = 100, 100, 5
    p_obs = 0.8  # 80-20 Train-Test split
    lambda_reg = 8.5  # Heavy regularization to force recovered rank < 10
    
    # Generate Environment
    X_star, Y, R = generate_synthetic_data(m, n, rank, p_obs)
    
    print(f"Matrix Size: {m}x{n}, Target Rank: {rank}, Observed: {p_obs*100}%")
    print("Running Proximal Gradient Descent with SVT...")
    
    # Execute Algorithm
    X_pred, history = binary_matrix_completion(Y, R, lambda_reg=lambda_reg, eta=1.0, max_iter=300)
    
    # Evaluate
    train_acc, test_acc, rec_rank, rel_frob, eff_rank = calculate_metrics(X_pred, Y, X_star, R)
    
    print(f"\n--- Final Report ---")
    print(f"Final Objective Value: {history['objective'][-1]:.4f}")
    print(f"Total Iterations: {len(history['objective'])}")
    print(f"Total Runtime: {history['time'][-1]:.4f} seconds")
    print(f"Train Sign Accuracy: {train_acc*100:.2f}%")
    print(f"Test Sign Accuracy (Held-Out): {test_acc*100:.2f}%")
    print(f"Relative Frobenius Error: {rel_frob:.4f}")
    print(f"Recovered Exact Rank: {rec_rank}")
    print(f"Effective Rank: {eff_rank:.2f}")