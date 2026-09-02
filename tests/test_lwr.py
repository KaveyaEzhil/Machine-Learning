import numpy as np
from src.lwr_regressor import FirstPrinciplesLWR

def test_lwr_gaussian_kernel():
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (120, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    
    lwr = FirstPrinciplesLWR(tau=0.5)
    lwr.fit(X[:100], y[:100])
    preds = lwr.predict(X[100:])
    
    assert len(preds) == 20
    assert not np.isnan(preds).any()
    rmse = np.sqrt(np.mean((y[100:] - preds)**2))
    assert rmse < 1.0
