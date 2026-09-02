"""
Main End-to-End Execution Pipeline (ITA0613 Machine Learning Assignment)
Executes the full instance-based and statistical learning workflow from raw data ingestion
to model evaluation, candidate elimination, scalability benchmarking, and publication graphics.
"""

import os
import time
import json
import numpy as np
import pandas as pd

from src.generate_dataset import generate_raw_agro_climatic_data
from src.data_pipeline import AgroClimaticDataPipeline
from src.knn_regressor import FirstPrinciplesKNNRegressor, compute_manual_validation_curve
from src.lwr_regressor import FirstPrinciplesLWR, evaluate_lwr_bandwidth_tradeoff
from src.candidate_elimination import CandidateElimination, discretize_agro_climatic_data
from src.scalability_tree import run_scalability_benchmark, FirstPrinciplesKDTree
from src.visualization import AgroClimaticVisualizer

def main():
    print("=" * 90)
    print(" ITA0613 - MACHINE LEARNING ASSIGNMENT PIPELINE")
    print(" Title: Large-Scale Instance-Based and Statistical Learning Pipeline for Climate-Resilient Crop Yield")
    print(" Student: Kaveya E (IET Assignment kaveya E)")
    print("=" * 90)
    
    start_time = time.time()
    
    # ---------------------------------------------------------
    # Step 1: Raw Data Sourcing & Generation (>10,000 Records)
    # ---------------------------------------------------------
    print("\n--- STEP 1: Sourcing & Generating Multi-Region Agro-Climatic Dataset ---")
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) < 5:
        generate_raw_agro_climatic_data(output_dir=raw_dir, n_total_records=12500)
    
    # ---------------------------------------------------------
    # Step 2: Data Engineering & EDA Pipeline (NumPy & Pandas)
    # ---------------------------------------------------------
    print("\n--- STEP 2: Executing Data Engineering & Feature Pipeline ---")
    pipeline = AgroClimaticDataPipeline(raw_dir=raw_dir)
    df_processed = pipeline.run_pipeline(output_path="data/processed/agro_climatic_processed.csv")
    
    # ---------------------------------------------------------
    # Train / Validation / Test Split (Manual NumPy indices)
    # ---------------------------------------------------------
    feature_cols = ['GDD_Degree_Days', 'RAI_Index', 'THI_Index', 'SNI_Index', 
                    'Soil_pH', 'Organic_Carbon_Pct', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha']
    target_col = 'Crop_Yield_kg_ha'
    
    X = df_processed[feature_cols].values
    y = df_processed[target_col].values
    
    np.random.seed(42)
    n_samples = len(X)
    shuffled_indices = np.random.permutation(n_samples)
    
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:train_size + val_size]
    test_idx = shuffled_indices[train_size + val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # ---------------------------------------------------------
    # Step 3: k-NN Regressor & Manual Validation Curve (CO6)
    # ---------------------------------------------------------
    print("\n--- STEP 3: k-NN Regressor & Distance Metrics Evaluation ---")
    k_search_list = [1, 3, 5, 7, 9, 15, 21, 31, 51]
    val_curve_df = compute_manual_validation_curve(X_train, y_train, X_val, y_val, k_list=k_search_list, metric='euclidean')
    
    # Save validation curve table
    tables_dir = "results/tables"
    os.makedirs(tables_dir, exist_ok=True)
    val_curve_df.to_csv(os.path.join(tables_dir, "knn_validation_curve_metrics.csv"), index=False)
    
    opt_k_idx = np.argmin(val_curve_df['val_rmse'].values)
    opt_k = int(val_curve_df.iloc[opt_k_idx]['k'])
    print(f"\n[Optimal k Selected]: k = {opt_k} with Validation RMSE = {val_curve_df.iloc[opt_k_idx]['val_rmse']:.2f} kg/ha")
    
    # Compare Euclidean vs Mahalanobis Distance Metrics
    print("\n[Distance Metric Comparison on Test Set (k=opt_k)]:")
    knn_euc = FirstPrinciplesKNNRegressor(k=opt_k, metric='euclidean', weights='distance')
    knn_euc.fit(X_train, y_train)
    t0_euc = time.time()
    preds_euc = knn_euc.predict(X_test[:500])
    t_euc = time.time() - t0_euc
    rmse_euc = np.sqrt(np.mean((y_test[:500] - preds_euc)**2))
    mae_euc = np.mean(np.abs(y_test[:500] - preds_euc))
    
    knn_mah = FirstPrinciplesKNNRegressor(k=opt_k, metric='mahalanobis', weights='distance')
    knn_mah.fit(X_train, y_train)
    t0_mah = time.time()
    preds_mah = knn_mah.predict(X_test[:500])
    t_mah = time.time() - t0_mah
    rmse_mah = np.sqrt(np.mean((y_test[:500] - preds_mah)**2))
    mae_mah = np.mean(np.abs(y_test[:500] - preds_mah))
    
    print(f"Euclidean   -> RMSE: {rmse_euc:.2f} kg/ha | MAE: {mae_euc:.2f} kg/ha | Time: {t_euc:.4f}s")
    print(f"Mahalanobis -> RMSE: {rmse_mah:.2f} kg/ha | MAE: {mae_mah:.2f} kg/ha | Time: {t_mah:.4f}s")
    
    # ---------------------------------------------------------
    # Step 4: Locally Weighted Regression (LWR) (CO6)
    # ---------------------------------------------------------
    print("\n--- STEP 4: Locally Weighted Regression (LWR) & Bandwidth Tuning ---")
    tau_list = [0.1, 0.25, 0.5, 1.0, 2.0]
    lwr_results_df = evaluate_lwr_bandwidth_tradeoff(X_train, y_train, X_val, y_val, tau_list=tau_list)
    lwr_results_df.to_csv(os.path.join(tables_dir, "lwr_bandwidth_tradeoff_metrics.csv"), index=False)
    
    opt_tau = 0.5
    lwr_opt = FirstPrinciplesLWR(tau=opt_tau)
    lwr_opt.fit(X_train, y_train)
    lwr_preds_test = lwr_opt.predict(X_test[:500])
    rmse_lwr = np.sqrt(np.mean((y_test[:500] - lwr_preds_test)**2))
    mae_lwr = np.mean(np.abs(y_test[:500] - lwr_preds_test))
    print(f"\n[LWR Test Evaluation (tau={opt_tau})]: RMSE = {rmse_lwr:.2f} kg/ha | MAE = {mae_lwr:.2f} kg/ha")
    
    # ---------------------------------------------------------
    # Step 5: Candidate-Elimination & Version Space (CO1/CO2)
    # ---------------------------------------------------------
    print("\n--- STEP 5: Candidate-Elimination & Version Space Analysis ---")
    df_disc, y_risk, domain_vals = discretize_agro_climatic_data(df_processed)
    attr_names = list(df_disc.columns)
    
    # Filter unique consistent instances to avoid noise-induced version space collapse
    combined_df = df_disc.copy()
    combined_df['Target_Risk'] = y_risk
    unique_instances = combined_df.drop_duplicates(subset=attr_names, keep='first')
    
    X_ce = unique_instances[attr_names].values[:20]
    y_ce = unique_instances['Target_Risk'].values[:20]
    
    ce_engine = CandidateElimination(attr_names, domain_vals)
    S_final, G_final = ce_engine.fit(X_ce, y_ce)
    
    print("\n[Candidate-Elimination Final Version Space Boundaries]:")
    print(f"Specific Boundary S ({len(S_final)} hypotheses): {S_final}")
    print(f"General Boundary G  ({len(G_final)} hypotheses): {G_final}")
    
    with open(os.path.join(tables_dir, "candidate_elimination_hypotheses.json"), "w") as f:
        json.dump({'S_Boundary': S_final, 'G_Boundary': G_final}, f, indent=2)
        
    # ---------------------------------------------------------
    # Step 6: Scalability Analysis & k-d Tree Prototype
    # ---------------------------------------------------------
    print("\n--- STEP 6: Scalability Complexity Benchmark (10^3 to 10^6 Records) ---")
    scale_df = run_scalability_benchmark(scales=[1000, 10000, 100000, 1000000], n_queries=50, k=opt_k)
    scale_df.to_csv(os.path.join(tables_dir, "scalability_benchmark_metrics.csv"), index=False)
    
    # ---------------------------------------------------------
    # Step 7: Render 5 Publication-Quality Visualizations
    # ---------------------------------------------------------
    print("\n--- STEP 7: Rendering 5 Publication-Quality Visualizations ---")
    vis = AgroClimaticVisualizer(output_dir="results/plots")
    
    vis.plot_figure_1_correlation_heatmap(df_processed)
    vis.plot_figure_2_knn_validation_curve(val_curve_df)
    vis.plot_figure_3_lwr_vs_knn_bias_variance(X_test[:500], y_test[:500], preds_euc, lwr_preds_test)
    vis.plot_figure_4_scalability_benchmark(scale_df)
    vis.plot_figure_5_climate_yield_risk_distribution(df_processed)
    
    elapsed = time.time() - start_time
    print(f"\n==========================================================================")
    print(f" PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f" All reproducible artifacts saved to 'results/plots/' and 'results/tables/'.")
    print(f"==========================================================================")

if __name__ == '__main__':
    main()
