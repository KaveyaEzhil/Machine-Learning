"""
k-Nearest Neighbours Regressor from First Principles (CO6)
Implements KNN regression using pure NumPy with Euclidean and Mahalanobis distance metrics,
distance-weighted averaging, and manual validation curve evaluation.
"""

import numpy as np
import pandas as pd

class FirstPrinciplesKNNRegressor:
    def __init__(self, k=5, metric='euclidean', weights='uniform', epsilon=1e-6):
        """
        Parameters:
        - k: Number of nearest neighbours.
        - metric: 'euclidean' or 'mahalanobis'.
        - weights: 'uniform' or 'distance'.
        - epsilon: Small regularization constant for matrix inverse or distance weighting.
        """
        self.k = k
        self.metric = metric.lower()
        self.weights = weights.lower()
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        self.cov_inv = None
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y):
        """Fits the KNN regressor by storing training data and computing standardization parameters."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        # Standardize features for scale invariance
        self.feature_means = np.mean(X_arr, axis=0)
        self.feature_stds = np.std(X_arr, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # Prevent div by zero
        
        self.X_train = (X_arr - self.feature_means) / self.feature_stds
        self.y_train = y_arr
        
        if self.metric == 'mahalanobis':
            # Compute covariance matrix of standardized training features
            cov_matrix = np.cov(self.X_train, rowvar=False)
            # Add L2 regularization for stable matrix inverse
            cov_matrix_reg = cov_matrix + self.epsilon * np.eye(self.X_train.shape[1])
            self.cov_inv = np.linalg.pinv(cov_matrix_reg)
            
        return self

    def _compute_distances(self, X_query):
        """
        Computes pairwise distance matrix between query samples and training set.
        Returns distance matrix of shape (n_query, n_train).
        """
        n_query = X_query.shape[0]
        n_train = self.X_train.shape[0]
        
        if self.metric == 'euclidean':
            # Vectorized Euclidean distance computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            q_sq = np.sum(X_query**2, axis=1, keepdims=True)  # (n_query, 1)
            t_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T  # (1, n_train)
            dot_prod = np.dot(X_query, self.X_train.T)  # (n_query, n_train)
            dist_sq = q_sq + t_sq - 2.0 * dot_prod
            dist_sq = np.maximum(0.0, dist_sq)  # Clamp numerical floating precision artifacts
            return np.sqrt(dist_sq)
            
        elif self.metric == 'mahalanobis':
            # Vectorized Mahalanobis distance computation
            # d_M(x, y) = sqrt((x - y) C^-1 (x - y)^T)
            distances = np.zeros((n_query, n_train), dtype=np.float64)
            for i in range(n_query):
                diff = self.X_train - X_query[i]  # (n_train, n_features)
                # Compute diff @ cov_inv @ diff.T row by row efficiently
                left = np.dot(diff, self.cov_inv)  # (n_train, n_features)
                d_sq = np.sum(left * diff, axis=1)  # dot product per row
                distances[i] = np.sqrt(np.maximum(0.0, d_sq))
            return distances
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Choose 'euclidean' or 'mahalanobis'.")

    def predict(self, X):
        """Predicts continuous target values for X."""
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        
        distances = self._compute_distances(X_scaled)  # (n_query, n_train)
        
        # Find indices of top K nearest neighbours
        # np.argpartition is O(N) compared to full sort O(N log N)
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        
        n_query = X_scaled.shape[0]
        y_pred = np.zeros(n_query, dtype=np.float64)
        
        for i in range(n_query):
            idx = k_indices[i]
            dists = distances[i, idx]
            
            # Sort exact top k
            sort_order = np.argsort(dists)
            idx = idx[sort_order]
            dists = dists[sort_order]
            
            neighbor_y = self.y_train[idx]
            
            if self.weights == 'uniform':
                y_pred[i] = np.mean(neighbor_y)
            elif self.weights == 'distance':
                # Inverse distance weighting: w_i = 1 / (d_i + epsilon)
                w = 1.0 / (dists + self.epsilon)
                w_sum = np.sum(w)
                y_pred[i] = np.sum(w * neighbor_y) / w_sum
            else:
                raise ValueError(f"Unsupported weighting scheme: {self.weights}")
                
        return y_pred


def compute_manual_validation_curve(X_train, y_train, X_val, y_val, k_list=[1, 3, 5, 7, 9, 15, 21, 31, 51], metric='euclidean'):
    """
    Computes manual validation curves across k values returning train/val MSE, RMSE, and MAE.
    """
    results = []
    for k in k_list:
        model = FirstPrinciplesKNNRegressor(k=k, metric=metric, weights='uniform')
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train[:1000])  # Sample train for computational speed
        val_pred = model.predict(X_val)
        
        train_mse = np.mean((y_train[:1000] - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)
        
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        
        train_mae = np.mean(np.abs(y_train[:1000] - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))
        
        results.append({
            'k': k,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae
        })
        print(f"[Validation Curve] k={k:2d} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f} | Val MAE: {val_mae:.2f}")
        
    return pd.DataFrame(results)

if __name__ == '__main__':
    # Unit test harness
    X_dummy = np.random.randn(200, 4)
    y_dummy = X_dummy[:, 0] * 3.0 + X_dummy[:, 1] * -2.0 + np.random.randn(200) * 0.5
    
    knn_euc = FirstPrinciplesKNNRegressor(k=5, metric='euclidean')
    knn_euc.fit(X_dummy[:150], y_dummy[:150])
    preds_euc = knn_euc.predict(X_dummy[150:])
    
    knn_mah = FirstPrinciplesKNNRegressor(k=5, metric='mahalanobis')
    knn_mah.fit(X_dummy[:150], y_dummy[:150])
    preds_mah = knn_mah.predict(X_dummy[150:])
    
    print("Euclidean Preds sample:", preds_euc[:5])
    print("Mahalanobis Preds sample:", preds_mah[:5])
