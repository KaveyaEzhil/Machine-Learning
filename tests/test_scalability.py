import numpy as np
from src.scalability_tree import FirstPrinciplesKDTree

def test_kd_tree_nearest_neighbour():
    np.random.seed(42)
    X = np.random.randn(500, 4)
    q = X[25] + 0.01  # Query near index 25
    
    kd = FirstPrinciplesKDTree(X)
    dists_kd, indices_kd = kd.query_knn(q, k=3)
    
    # Brute force comparison
    dists_bf = np.sqrt(np.sum((X - q)**2, axis=1))
    idx_bf = np.argsort(dists_bf)[:3]
    dists_bf_top = dists_bf[idx_bf]
    
    assert np.allclose(dists_kd, dists_bf_top)
    assert indices_kd[0] == 25  # The closest point should be index 25
