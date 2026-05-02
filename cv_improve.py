import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Load p-value selected features (first step in the paper)
pval_mask = np.load('../Data/selected_coefs_pvals.npy', allow_pickle=True)

# Fine-grained search for achieving exactly 78 features
ALPHAS = [0.00101, 0.00102, 0.00103, 0.00104, 0.00105]
total = len(ALPHAS)

if __name__ == "__main__":
    print("============================================================")
    print("  LASSO Feature Selection | Paper Reproduction | p-val + Lasso → 78")
    print("  WITH ADJUSTED R² SEARCH | Fast Version")
    print("============================================================\n")

    X_train = pd.read_csv('../Data/data_cleaned_train_comments_X.csv')
    y_train = pd.read_csv('../Data/data_cleaned_train_y.csv').values.ravel()
    X_val = pd.read_csv('../Data/data_cleaned_val_comments_X.csv')
    y_val = pd.read_csv('../Data/data_cleaned_val_y.csv').values.ravel()

    # Apply p-value feature mask first
    X_train = X_train.loc[:, pval_mask]
    X_val = X_val.loc[:, pval_mask]

    score_best = -np.inf
    alpha_best = ALPHAS[0]
    target_features = 78
    best_diff = float('inf')

    n_train, n_feats = X_train.shape

    for idx, alpha in enumerate(ALPHAS):
        print(f"[{idx+1}/{total}] Processing alpha = {alpha:.6f}")

        reg = Lasso(
            alpha=alpha,
            max_iter=2000,
            tol=0.001,
            random_state=42
        )

        reg.fit(X_train, y_train)
        n_features = np.sum(reg.coef_ != 0)

        # Calculate validation R²
        y_pred_val = reg.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        # Calculate adjusted R² for model selection
        y_pred_train = reg.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        adjusted_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - n_features - 1)

        diff = abs(n_features - target_features)

        print(f"    -> Features: {n_features:3d} | Val R²: {val_r2:.4f} | Adj R²: {adjusted_r2:.4f} | Diff: {diff}")

        # Selection criteria: closest to 78 features, then highest adjusted R²
        if diff < best_diff or (diff == best_diff and adjusted_r2 > score_best):
            best_diff = diff
            score_best = adjusted_r2
            alpha_best = alpha

    print(f"\nBest alpha = {alpha_best:.6f}")
    print(f"Best adjusted R² = {score_best:.4f}\n")

    # Final Lasso model training
    reg_final = Lasso(alpha=alpha_best, max_iter=2000, tol=0.001, random_state=42)
    reg_final.fit(X_train, y_train)
    lasso_mask = reg_final.coef_ != 0

    # Generate final feature mask for 78 features
    final_mask = np.zeros(len(pval_mask), dtype=bool)
    final_mask[np.where(pval_mask)[0]] = lasso_mask

    np.save('../Data/selected_coefs_final.npy', final_mask)
    print(f"\n Final features = {np.sum(final_mask)} (saved as selected_coefs_final.npy)")
    print(f" Adjusted R² search completed ")
