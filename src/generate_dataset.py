"""
Dataset Generation Script for ITA0613 ML Assignment.
Generates multi-region, multi-year agro-climatic raw dataset files (>12,000 records)
with missing weather records and realistic agronomic-climatic dynamics.
"""

import os
import numpy as np
import pandas as pd

def generate_raw_agro_climatic_data(output_dir="data/raw", n_total_records=12500, random_seed=42):
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    regions = {
        'North_Zone': {'districts': ['Punjab_Central', 'Haryana_East', 'UP_Western', 'Rajasthan_North'], 'base_temp': 24.0, 'temp_var': 8.0, 'base_rain': 650.0, 'rain_var': 200.0},
        'South_Zone': {'districts': ['TN_Delta', 'AP_Coastal', 'Karnataka_Deccan', 'Kerala_Highland'], 'base_temp': 29.0, 'temp_var': 4.0, 'base_rain': 1100.0, 'rain_var': 350.0},
        'East_Zone': {'districts': ['WB_Gangetic', 'Odisha_Coastal', 'Bihar_Plains', 'Assam_Valley'], 'base_temp': 26.0, 'temp_var': 5.0, 'base_rain': 1400.0, 'rain_var': 400.0},
        'West_Zone': {'districts': ['MH_Vidarbha', 'GJ_Saurashtra', 'MP_Malwa', 'MH_Marathwada'], 'base_temp': 28.0, 'temp_var': 6.0, 'base_rain': 750.0, 'rain_var': 250.0},
        'Central_Zone': {'districts': ['MP_Narmada', 'CG_Plains', 'UP_Bundelkhand', 'JH_Plateau'], 'base_temp': 27.0, 'temp_var': 6.5, 'base_rain': 900.0, 'rain_var': 300.0}
    }
    
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    seasons = ['Kharif', 'Rabi', 'Zaid']
    crops = ['Rice', 'Wheat', 'Maize', 'Soybean', 'Cotton', 'Pulses']
    
    records_per_region = n_total_records // len(regions)
    
    for r_idx, (region_name, r_info) in enumerate(regions.items()):
        data = []
        for i in range(records_per_region):
            year = np.random.choice(years)
            season = np.random.choice(seasons, p=[0.5, 0.4, 0.1])
            district = np.random.choice(r_info['districts'])
            crop = np.random.choice(crops)
            
            # Climate physics simulation
            season_temp_offset = -4.0 if season == 'Rabi' else (2.0 if season == 'Zaid' else 0.0)
            season_rain_mult = 0.2 if season == 'Rabi' else (0.1 if season == 'Zaid' else 1.0)
            
            t_min = r_info['base_temp'] + season_temp_offset + np.random.normal(-3.0, 1.5)
            t_max = t_min + np.random.uniform(7.0, 14.0)
            rainfall = max(10.0, (r_info['base_rain'] * season_rain_mult) + np.random.normal(0, r_info['rain_var'] * season_rain_mult))
            humidity = np.clip(50.0 + (rainfall / 30.0) + np.random.normal(0, 10.0), 30.0, 95.0)
            solar_rad = np.clip(18.0 - (humidity / 10.0) + np.random.normal(0, 2.0), 10.0, 26.0)
            
            # Soil parameters
            soil_ph = np.clip(np.random.normal(6.8, 0.6), 5.2, 8.5)
            organic_carbon = np.clip(np.random.normal(0.6, 0.18), 0.2, 1.2)
            nitrogen = np.clip(np.random.normal(180, 35), 90, 280)
            phosphorus = np.clip(np.random.normal(22, 6), 8, 45)
            potassium = np.clip(np.random.normal(210, 45), 110, 350)
            
            # Realistic non-linear yield generation formula (kg/ha)
            # Base crop yields
            crop_base_yield = {'Rice': 3600, 'Wheat': 3200, 'Maize': 2800, 'Soybean': 2000, 'Cotton': 1800, 'Pulses': 1200}[crop]
            
            # Derived physics factors for realistic modeling
            t_avg = (t_max + t_min) / 2.0
            gdd = max(0.0, t_avg - 10.0) * 120.0  # Approx cumulative GDD for season
            heat_stress_penalty = max(0.0, (t_max - 35.0) * 80.0)
            water_stress_penalty = max(0.0, (400.0 - rainfall) * 2.5) if rainfall < 400 else max(0.0, (rainfall - 1500.0) * 1.2)
            soil_quality_factor = 0.7 + (organic_carbon * 0.3) + (nitrogen / 600.0)
            
            yield_val = (crop_base_yield * soil_quality_factor) + (gdd * 0.4) - heat_stress_penalty - water_stress_penalty + np.random.normal(0, 150)
            yield_val = max(300.0, yield_val)
            
            # Introduce controlled missing values in weather variables (~6% NaN rate)
            if np.random.rand() < 0.06:
                t_min = np.nan
            if np.random.rand() < 0.06:
                t_max = np.nan
            if np.random.rand() < 0.06:
                rainfall = np.nan
            if np.random.rand() < 0.05:
                humidity = np.nan
                
            data.append({
                'Record_ID': f"AGR_{region_name[:3]}_{year}_{i+1:05d}",
                'Region': region_name,
                'District': district,
                'Year': year,
                'Season': season,
                'Crop': crop,
                'Temp_Min_C': round(t_min, 2) if not np.isnan(t_min) else np.nan,
                'Temp_Max_C': round(t_max, 2) if not np.isnan(t_max) else np.nan,
                'Rainfall_mm': round(rainfall, 2) if not np.isnan(rainfall) else np.nan,
                'Humidity_Pct': round(humidity, 2) if not np.isnan(humidity) else np.nan,
                'Solar_Rad_MJm2': round(solar_rad, 2),
                'Soil_pH': round(soil_ph, 2),
                'Organic_Carbon_Pct': round(organic_carbon, 3),
                'Nitrogen_kg_ha': round(nitrogen, 1),
                'Phosphorus_kg_ha': round(phosphorus, 1),
                'Potassium_kg_ha': round(potassium, 1),
                'Crop_Yield_kg_ha': round(yield_val, 2)
            })
            
        df_region = pd.DataFrame(data)
        file_path = os.path.join(output_dir, f"agro_climatic_{region_name.lower()}.csv")
        df_region.to_csv(file_path, index=False)
        print(f"Generated {len(df_region)} records for {region_name} at {file_path}")

if __name__ == '__main__':
    generate_raw_agro_climatic_data()
