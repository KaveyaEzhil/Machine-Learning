"""
Custom Unit Test Runner for ITA0613 Machine Learning Pipeline.
Executes all test cases in tests/ directory and reports results.
"""

import sys
from tests.test_data_pipeline import test_pipeline_imputation_and_feature_engineering
from tests.test_knn import test_knn_euclidean_and_mahalanobis
from tests.test_lwr import test_lwr_gaussian_kernel
from tests.test_candidate_elimination import test_candidate_elimination_boundaries
from tests.test_scalability import test_kd_tree_nearest_neighbour

def run_all_tests():
    tests = [
        ("Data Pipeline Imputation & Feature Engineering", test_pipeline_imputation_and_feature_engineering),
        ("KNN Regressor (Euclidean & Mahalanobis)", test_knn_euclidean_and_mahalanobis),
        ("Locally Weighted Regression (LWR)", test_lwr_gaussian_kernel),
        ("Candidate Elimination Boundary Updates", test_candidate_elimination_boundaries),
        ("k-d Tree Nearest Neighbour Search", test_kd_tree_nearest_neighbour)
    ]
    
    print("==================================================")
    print(" RUNNING AUTOMATED UNIT TEST SUITE")
    print("==================================================")
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name} -> Error: {e}")
            failed += 1
            
    print("-" * 50)
    print(f"TEST SUMMARY: Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    print("==================================================")
    
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()
