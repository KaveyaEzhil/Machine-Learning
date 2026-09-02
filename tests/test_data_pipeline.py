import numpy as np
import pandas as pd
from src.data_pipeline import AgroClimaticDataPipeline

def test_pipeline_imputation_and_feature_engineering():
    # Construct synthetic raw dataframe with missing values
    df_raw = pd.DataFrame({
        'Record_ID': ['AGR_1', 'AGR_2', 'AGR_3', 'AGR_4'],
        'Region': ['North', 'North', 'South', 'South'],
        'District': ['D1', 'D1', 'D2', 'D2'],
        'Season': ['Kharif', 'Kharif', 'Rabi', 'Rabi'],
        'Temp_Min_C': [20.0, np.nan, 15.0, 18.0],
        'Temp_Max_C': [32.0, 34.0, np.nan, 28.0],
        'Rainfall_mm': [600.0, 700.0, 300.0, np.nan],
        'Humidity_Pct': [70.0, 65.0, 50.0, 55.0],
        'Soil_pH': [6.5, 6.8, 7.0, 6.2],
        'Organic_Carbon_Pct': [0.6, 0.7, 0.5, 0.4],
        'Nitrogen_kg_ha': [180, 200, 150, 160],
        'Phosphorus_kg_ha': [20, 25, 18, 22],
        'Potassium_kg_ha': [200, 210, 180, 190],
        'Crop_Yield_kg_ha': [3200, 3400, 2800, 2600]
    })
    
    pipeline = AgroClimaticDataPipeline()
    df_clean = pipeline.clean_and_impute(df_raw)
    
    # Assert missing values are imputed
    assert df_clean['Temp_Min_C'].isna().sum() == 0
    assert df_clean['Temp_Max_C'].isna().sum() == 0
    assert df_clean['Rainfall_mm'].isna().sum() == 0
    
    df_feat = pipeline.engineer_derived_features(df_clean)
    
    # Assert derived features exist
    assert 'GDD_Degree_Days' in df_feat.columns
    assert 'RAI_Index' in df_feat.columns
    assert 'THI_Index' in df_feat.columns
    assert 'SNI_Index' in df_feat.columns
    
    # Verify mathematical non-negativity of GDD
    assert np.all(df_feat['GDD_Degree_Days'] >= 0)
