"""
Data Engineering and Exploratory Pipeline Module (CO7)
Pure NumPy & Pandas implementation for dataset merging, cleaning, missing weather imputation,
and derived feature engineering (GDD, RAI, THI, SNI).
"""

import os
import glob
import numpy as np
import pandas as pd

class AgroClimaticDataPipeline:
    def __init__(self, raw_dir="data/raw", base_temp=10.0):
        self.raw_dir = raw_dir
        self.base_temp = base_temp
        self.imputation_stats = {}
        
    def load_and_merge_files(self):
        """Loads and merges all regional/yearly CSV files from raw_dir."""
        csv_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV raw files found in {self.raw_dir}")
            
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"[Pipeline] Merged {len(csv_files)} files totaling {len(merged_df)} raw records.")
        return merged_df

    def clean_and_impute(self, df):
        """Cleans duplicates and imputes missing weather variables using pure NumPy/Pandas."""
        df = df.copy()
        initial_len = len(df)
        df.drop_duplicates(subset=['Record_ID'], inplace=True)
        print(f"[Pipeline] Removed {initial_len - len(df)} duplicate records.")
        
        weather_cols = ['Temp_Min_C', 'Temp_Max_C', 'Rainfall_mm', 'Humidity_Pct']
        
        # Track missingness statistics
        for col in weather_cols:
            missing_cnt = df[col].isna().sum()
            self.imputation_stats[col] = {
                'missing_count': int(missing_cnt),
                'missing_pct': float((missing_cnt / len(df)) * 100)
            }
            print(f"[Pipeline] Variable '{col}' has {missing_cnt} missing entries ({self.imputation_stats[col]['missing_pct']:.2f}%).")
        
        # Pure Pandas/NumPy District-Season Median Imputation
        for col in weather_cols:
            # Group-level median by District and Season
            district_season_medians = df.groupby(['District', 'Season'])[col].transform('median')
            df[col] = df[col].fillna(district_season_medians)
            
            # Fallback to District median
            district_medians = df.groupby('District')[col].transform('median')
            df[col] = df[col].fillna(district_medians)
            
            # Fallback to overall median
            overall_median = df[col].median()
            df[col] = df[col].fillna(overall_median)
            
            assert df[col].isna().sum() == 0, f"Imputation incomplete for column {col}"

        print("[Pipeline] Missing weather imputation complete across all records.")
        return df

    def engineer_derived_features(self, df):
        """
        Engineers at least 3 domain-specific agro-climatic derived features:
        1. Growing Degree Days (GDD)
        2. Rainfall Anomaly Index (RAI)
        3. Temperature-Humidity Index (THI)
        4. Soil Nutrient Index (SNI)
        """
        df = df.copy()
        
        # 1. Growing Degree Days (GDD)
        # GDD = max(0, ((T_max + T_min)/2) - T_base) * 120 (accumulated over ~120 day crop cycle)
        t_avg = (df['Temp_Max_C'] + df['Temp_Min_C']) / 2.0
        gdd = np.maximum(0.0, t_avg - self.base_temp) * 120.0
        df['GDD_Degree_Days'] = np.round(gdd, 2)
        
        # 2. Rainfall Anomaly Index (RAI)
        # RAI = (R - mean(R_district)) / std(R_district)
        district_rain_mean = df.groupby('District')['Rainfall_mm'].transform('mean')
        district_rain_std = df.groupby('District')['Rainfall_mm'].transform('std').replace(0, 1.0)
        rai = (df['Rainfall_mm'] - district_rain_mean) / district_rain_std
        df['RAI_Index'] = np.round(rai, 3)
        
        # 3. Temperature-Humidity Index (THI) - Heat Stress Indicator
        # THI = 0.8 * T_avg + (RH / 100) * (T_avg - 14.3) + 46.4
        thi = 0.8 * t_avg + (df['Humidity_Pct'] / 100.0) * (t_avg - 14.3) + 46.4
        df['THI_Index'] = np.round(thi, 2)
        
        # 4. Soil Nutrient Ratio Index (SNI)
        sni = (df['Nitrogen_kg_ha'] + df['Phosphorus_kg_ha'] + df['Potassium_kg_ha']) / df['Soil_pH']
        df['SNI_Index'] = np.round(sni, 2)
        
        print("[Pipeline] Successfully engineered 4 derived features: GDD_Degree_Days, RAI_Index, THI_Index, SNI_Index.")
        return df

    def run_pipeline(self, output_path="data/processed/agro_climatic_processed.csv"):
        """Executes end-to-end data processing workflow."""
        raw_df = self.load_and_merge_files()
        cleaned_df = self.clean_and_impute(raw_df)
        processed_df = self.engineer_derived_features(cleaned_df)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(f"[Pipeline] Saved processed dataset ({len(processed_df)} rows, {processed_df.shape[1]} columns) to {output_path}")
        return processed_df

if __name__ == '__main__':
    pipeline = AgroClimaticDataPipeline()
    df = pipeline.run_pipeline()
    print("Summary of Processed Dataset:")
    print(df[['Crop_Yield_kg_ha', 'GDD_Degree_Days', 'RAI_Index', 'THI_Index', 'SNI_Index']].describe())
