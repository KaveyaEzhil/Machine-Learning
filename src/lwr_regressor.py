"""
Locally Weighted Regression (LWR) from First Principles (CO6)
Implements LWR / LOESS regression using pure NumPy with a Gaussian kernel solver,
L2 regularization stability, bandwidth tuning, and bias-variance analysis utilities.
"""

import numpy as np
import pandas as pd

class FirstPrinciplesLWR:
    def __init__(self, tau=0.5, l2_reg=1e-4):
        """
        Parameters:
        - tau: Bandwidth hyperparameter controlling local neighborhood decay.
        - l2_reg: Regularization constant for numerical stability of matrix inverse.
        """
        self.tau = tau
        self.l2_reg = l2_reg
        self.X_train = None
        self.y_train = None
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y):
        """Stores training data and computes standardization parameters."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        self.feature_means = np.mean(X_arr, axis=0)
        self.feature_stds = np.std(X_arr, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0
        
        # Standardize features and add bias column (intercept)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        self.X_train = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
        self.y_train = y_arr
        return self

    def _predict_single_point(self, x_query_single):
        """
        Predicts target for a single query point x_query (with bias term already prepended).
        Solves: theta = (X^T W X + lambda I)^-1 X^T W y
        """
        # Compute squared Euclidean distances to all training points
        diff = self.X_train[:, 1:] - x_query_single[1:]  # Exclude bias column for distance
        dist_sq = np.sum(diff**2, axis=1)  # (N_train,)
        
        # Gaussian weighting kernel: w_i = exp(- dist_sq / (2 * tau^2))
        weights = np.exp(-dist_sq / (2.0 * (self.tau ** 2)))
        
        # Form W * X: multiply each row of X by weight_i
        # W is diagonal, so W @ X is weights[:, None] * X_train
        WX = self.X_train * weights[:, np.newaxis]
        
        # Matrix equation: X^T W X
        XtWX = np.dot(self.X_train.T, WX)
        
        # Add L2 regularization for non-singular matrix inverse
        n_features = XtWX.shape[0]
        XtWX_reg = XtWX + self.l2_reg * np.eye(n_features)
        
        # X^T W y
        XtWy = np.dot(WX.T, self.y_train)
        
        # Solve for local theta
        try:
            theta = np.linalg.solve(XtWX_reg, XtWy)
        except np.linalg.LinAlgError:
            theta = np.dot(np.linalg.pinv(XtWX_reg), XtWy)
            
        # Prediction y_hat = x_query . theta
        return np.dot(x_query_single, theta)

    def predict(self, X):
        """Predicts targets for query matrix X."""
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        X_query = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
        
        preds = np.zeros(X_query.shape[0], dtype=np.float64)
        for i in range(X_query.shape[0]):
            preds[i] = self._predict_single_point(X_query[i])
            
        return preds


def evaluate_lwr_bandwidth_tradeoff(X_train, y_train, X_val, y_val, tau_list=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]):
    """Evaluates LWR performance across bandwidth list tau_list."""
    results = []
    for tau in tau_list:
        lwr = FirstPrinciplesLWR(tau=tau)
        lwr.fit(X_train, y_train)
        
        # Sample for validation runtime efficiency
        n_eval = min(500, len(X_val))
        val_pred = lwr.predict(X_val[:n_eval])
        
        val_mse = np.mean((y_val[:n_eval] - val_pred) ** 2)
        val_rmse = np.sqrt(val_mse)
        val_mae = np.mean(np.abs(y_val[:n_eval] - val_pred))
        
        results.append({
            'tau': tau,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })
        print(f"[LWR Bandwidth Evaluation] tau={tau:.2f} | Val RMSE: {val_rmse:.2f} | Val MAE: {val_mae:.2f}")
        
    return pd.DataFrame(results)

if __name__ == '__main__':
    # Unit test harness for LWR
    X_dummy = np.random.randn(200, 3)
    y_dummy = np.sin(X_dummy[:, 0]) * 5.0 + X_dummy[:, 1] ** 2 + np.random.randn(200) * 0.2
    
    lwr_model = FirstPrinciplesLWR(tau=0.3)
    lwr_model.fit(X_dummy[:150], y_dummy[:150])
    lwr_preds = lwr_model.predict(X_dummy[150:])
    
    print("LWR Preds sample:", lwr_preds[:5])
    print("LWR RMSE:", np.sqrt(np.mean((y_dummy[150:] - lwr_preds) ** 2)))
