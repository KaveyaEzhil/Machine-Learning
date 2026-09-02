import numpy as np
from src.knn_regressor import FirstPrinciplesKNNRegressor

def test_knn_euclidean_and_mahalanobis():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100) * 0.1
    
    knn_euc = FirstPrinciplesKNNRegressor(k=3, metric='euclidean', weights='uniform')
    knn_euc.fit(X[:80], y[:80])
    preds_euc = knn_euc.predict(X[80:])
    
    assert len(preds_euc) == 20
    assert not np.isnan(preds_euc).any()
    
    knn_mah = FirstPrinciplesKNNRegressor(k=3, metric='mahalanobis', weights='distance')
    knn_mah.fit(X[:80], y[:80])
    preds_mah = knn_mah.predict(X[80:])
    
    assert len(preds_mah) == 20
    assert not np.isnan(preds_mah).any()
    
    # Assert correlation with true target is high
    corr_euc = np.corrcoef(y[80:], preds_euc)[0, 1]
    assert corr_euc > 0.85
