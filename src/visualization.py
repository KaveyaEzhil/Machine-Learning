"""
Publication-Quality Visualization Suite for ITA0613 ML Assignment (CO7, SDG 2/13)
Renders 5 distinct publication-quality figures illustrating agro-climatic relationships,
k-NN validation curves, LWR bias-variance behavior, scalability benchmarks, and climate risk maps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style styling parameters
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

class AgroClimaticVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_figure_1_correlation_heatmap(self, df):
        """Figure 1: Agro-Climatic Feature Correlation Matrix & Target Relationships."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        feature_cols = ['Crop_Yield_kg_ha', 'GDD_Degree_Days', 'RAI_Index', 'THI_Index', 'SNI_Index', 
                        'Temp_Max_C', 'Rainfall_mm', 'Humidity_Pct', 'Organic_Carbon_Pct']
        
        corr = df[feature_cols].corr()
        
        cax = ax.matshow(corr, cmap='YlGnBu', vmin=-1.0, vmax=1.0)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        
        labels = ['Yield', 'GDD', 'RAI', 'THI', 'SNI', 'T_Max', 'Rainfall', 'Humidity', 'Organic C']
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='left')
        ax.set_yticklabels(labels)
        
        # Annotate values inside matrix cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr.iloc[i, j]
                text_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontweight='bold')
                
        ax.set_title("Figure 1: Agro-Climatic Feature Correlation Heatmap", pad=25)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "fig1_agro_climatic_correlation_heatmap.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Visualizer] Saved Figure 1 to {out_path}")

    def plot_figure_2_knn_validation_curve(self, val_df):
        """Figure 2: Manual k-NN Validation Curve (k vs Train/Val RMSE & MAE)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
        
        k_vals = val_df['k'].values
        
        # Subplot 1: RMSE
        ax1.plot(k_vals, val_df['train_rmse'], 'o-', label='Train RMSE', color='#1f77b4', linewidth=2, markersize=6)
        ax1.plot(k_vals, val_df['val_rmse'], 's--', label='Validation RMSE', color='#d62728', linewidth=2, markersize=6)
        
        # Highlight optimal k (min validation RMSE)
        opt_idx = np.argmin(val_df['val_rmse'].values)
        opt_k = k_vals[opt_idx]
        opt_rmse = val_df['val_rmse'].values[opt_idx]
        
        ax1.axvline(opt_k, color='green', linestyle=':', label=f'Optimal k = {opt_k}')
        ax1.set_xlabel("Number of Neighbours (k)")
        ax1.set_ylabel("Root Mean Squared Error (kg/ha)")
        ax1.set_title("k-NN Validation Curve: RMSE vs. k")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: MAE
        ax2.plot(k_vals, val_df['train_mae'], 'o-', label='Train MAE', color='#2ca02c', linewidth=2, markersize=6)
        ax2.plot(k_vals, val_df['val_mae'], 's--', label='Validation MAE', color='#ff7f0e', linewidth=2, markersize=6)
        ax2.axvline(opt_k, color='green', linestyle=':', label=f'Optimal k = {opt_k}')
        ax2.set_xlabel("Number of Neighbours (k)")
        ax2.set_ylabel("Mean Absolute Error (kg/ha)")
        ax2.set_title("k-NN Validation Curve: MAE vs. k")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Figure 2: Manual k-NN Hyperparameter Optimization & Model Selection", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "fig2_knn_validation_curve.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Visualizer] Saved Figure 2 to {out_path}")

    def plot_figure_3_lwr_vs_knn_bias_variance(self, X_eval, y_eval, knn_preds, lwr_preds):
        """Figure 3: Empirical Bias-Variance Tradeoff (LWR vs k-NN Prediction Curves)."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # Sort along GDD feature for smooth line rendering
        gdd_vals = X_eval[:, 0]
        sort_idx = np.argsort(gdd_vals)
        
        ax.scatter(gdd_vals[sort_idx], y_eval[sort_idx], color='gray', alpha=0.3, label='Historical Crop Yield Data', s=25)
        ax.plot(gdd_vals[sort_idx], knn_preds[sort_idx], 'r-', label='k-NN Regressor (k=7, High Local Variance)', linewidth=2)
        ax.plot(gdd_vals[sort_idx], lwr_preds[sort_idx], 'b-', label='Locally Weighted Regression (tau=0.5, Smooth Bias-Variance)', linewidth=2.5)
        
        ax.set_xlabel("Growing Degree Days (GDD)")
        ax.set_ylabel("Crop Yield (kg/ha)")
        ax.set_title("Figure 3: Empirical Bias-Variance Comparison: LWR Smoothness vs. k-NN Piecewise Averaging")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "fig3_lwr_vs_knn_bias_variance.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Visualizer] Saved Figure 3 to {out_path}")

    def plot_figure_4_scalability_benchmark(self, scale_df):
        """Figure 4: Scalability Complexity Benchmark (10^3 to 10^6 Records)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
        
        N_vals = scale_df['N_records'].values
        
        # Log-Log scale query latency comparison
        ax1.loglog(N_vals, scale_df['Brute_Force_KNN_Time_Sec'], 'o-', label='Brute-Force O(N) Scan', color='#d62728', linewidth=2.5)
        ax1.loglog(N_vals, scale_df['KD_Tree_Query_Time_Sec'], 's--', label='First-Principles k-d Tree O(log N)', color='#1f77b4', linewidth=2.5)
        ax1.set_xlabel("Dataset Scale (N records)")
        ax1.set_ylabel("Query Latency (seconds, log scale)")
        ax1.set_title("Query Execution Time Scaling")
        ax1.legend(loc='upper left')
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        
        # Speedup Factor
        ax2.plot(N_vals, scale_df['Speedup_Factor'], 'g^--', linewidth=2.5, markersize=8, label='k-d Tree Speedup Factor')
        ax2.set_xscale('log')
        ax2.set_xlabel("Dataset Scale (N records)")
        ax2.set_ylabel("Speedup Multiplier (x)")
        ax2.set_title("k-d Tree Optimization Speedup")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Figure 4: Algorithmic Scalability Analysis from 1,000 to 1,000,000 Records", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "fig4_scalability_benchmark.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Visualizer] Saved Figure 4 to {out_path}")

    def plot_figure_5_climate_yield_risk_distribution(self, df):
        """Figure 5: Agro-Climatic Yield Risk Map & GDD/RAI Climate Impact (SDG 2 / SDG 13)."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        
        scatter = ax.scatter(df['GDD_Degree_Days'], df['RAI_Index'], c=df['Crop_Yield_kg_ha'], 
                             cmap='RdYlGn', alpha=0.7, s=30)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Crop Yield (kg/ha)", rotation=270, labelpad=15)
        
        # Overlay climate risk zones
        ax.axvline(1800, color='red', linestyle='--', alpha=0.6, label='Suboptimal GDD Boundary (<1800)')
        ax.axhline(-1.0, color='blue', linestyle='--', alpha=0.6, label='Drought Risk Boundary (RAI < -1.0)')
        
        ax.set_xlabel("Growing Degree Days (GDD)")
        ax.set_ylabel("Rainfall Anomaly Index (RAI)")
        ax.set_title("Figure 5: Climate-Yield Risk Map: Heat Accumulation vs. Precipitation Anomaly (SDG 2/13)")
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "fig5_climate_yield_risk_distribution.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Visualizer] Saved Figure 5 to {out_path}")

if __name__ == '__main__':
    # Test renderer
    df_sample = pd.DataFrame({
        'Crop_Yield_kg_ha': np.random.normal(3200, 800, 500),
        'GDD_Degree_Days': np.random.normal(2100, 300, 500),
        'RAI_Index': np.random.normal(0, 1, 500),
        'THI_Index': np.random.normal(76, 5, 500),
        'SNI_Index': np.random.normal(60, 10, 500),
        'Temp_Max_C': np.random.normal(34, 3, 500),
        'Rainfall_mm': np.random.normal(800, 200, 500),
        'Humidity_Pct': np.random.normal(65, 10, 500),
        'Organic_Carbon_Pct': np.random.normal(0.6, 0.1, 500),
        'Season': np.random.choice(['Kharif', 'Rabi'], 500)
    })
    vis = AgroClimaticVisualizer()
    vis.plot_figure_1_correlation_heatmap(df_sample)
    vis.plot_figure_5_climate_yield_risk_distribution(df_sample)
