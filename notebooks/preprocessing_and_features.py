# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.6
# ---

# %% [markdown]
# Data from: https://www.ncei.noaa.gov/cdo-web/search?datasetid=GHCND
#
# Collected Daily Summaries from 3 different stations to find average weather across regions of Minnesota: 
# - INTERNATIONAL FALLS INTERNATIONAL AIRPORT, MN US (N Region)
# - MINNEAPOLIS ST. PAUL INTERNATIONAL AIRPORT, MN US (SE Region)
# - ROCHESTER INTERNATIONAL AIRPORT, MN US (S/SE Region)

# %%
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('default', category=DeprecationWarning)

# %%
df = pd.read_csv('../data/raw/weather_data.csv')
df.tail(1)

# %% [markdown]
# Weather Type Codes
# - WT01	Fog, ice fog, or freezing fog (or haze): Reduced visibility, increased slickness.
# - WT02	Heavy fog or thick fog: Severely reduced visibility (major crash factor).
# - WT03	Thunder: Often correlated with heavy rain and sudden visibility changes.
# - WT04	Ice pellets, sleet, snow pellets, or small hail: Immediate increase in road slickness and difficulty controlling vehicles.
# - WT05	Hail (larger): Can cause property damage and sudden driver maneuvers.
# - WT06	Glaze or rime (freezing rain): The most dangerous condition for black ice formation.
# - WT08	Smoke or ash: Reduced air quality and significant visibility reduction.
# - WT09	Blowing or drifting snow: Reduced visibility and accumulation creating slick, uneven conditions.

# %% [markdown]
# ## 1. Initial Weather Data 

# %%
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df = df.sort_values(by='DATE', ascending=True)

# Checking NaN values in WT columns - filling NaN with 0
wt_cols = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09']
df[wt_cols] = df[wt_cols].fillna(0)

# Count NaN in TAVG column 
cols_to_check = ['TAVG', 'TMAX', 'TMIN', 'PRCP', 'SNOW']
for col in cols_to_check: 
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Initial NaN counts
print(f'Initial NaN values in TAVG: {df["TAVG"].isna().sum()}, NaN values in PRCP: {df["PRCP"].isna().sum()}, NaN values in SNOW: {df["SNOW"].isna().sum()}')

# %% [markdown]
# ## 2. Dealing with NaNs for Numerical Features
#
#

# %%
numerical_features = ['TAVG', 'PRCP', 'SNOW'] 

# First check if TMAX and TMIN are available to find TAVG
df['TAVG_ESTIMATE'] = (df['TMAX'] + df['TMIN']) / 2
df['TAVG'] = df['TAVG'].fillna(df['TAVG_ESTIMATE'])
print(f"NaNs remaining in TAVG after TMAX/TMIN imputation: {df['TAVG'].isnull().sum()}")

# If NaN still present, fill with daily median across stations
for col in numerical_features:
    daily_median_across_stations = df.groupby('DATE')[col].transform('median')
    df[col] = df[col].fillna(daily_median_across_stations)
    print(f"NaNs remaining in {col} after peer median fill: {df[col].isnull().sum()}")

# %% [markdown]
# ## 3. Aggregate and Merge

# %%
daily_median_weather = df.groupby('DATE')[numerical_features].median().reset_index()
daily_median_weather.columns = ['DATE'] + [f'{col}_MEDIAN' for col in numerical_features]

daily_max_wt = df.groupby('DATE')[wt_cols].max().reset_index()
daily_max_wt.columns = ['DATE'] + [f'{col}_MAX' for col in wt_cols]

final_weather_df = pd.merge(daily_median_weather, daily_max_wt, on='DATE', how='inner') 
print(final_weather_df.sample(10))
print(final_weather_df.info())


# %% [markdown]
# ## 4. Save Output for Weather (Optional)

# %%
""""
# Create processed data directory if it doesn't exist
import os
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Save the cleaned dataset
cleaned_file_path = os.path.join(processed_dir, 'cleaned_weather.csv')
final_weather_df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")
"""

# %% [markdown]
# ## 5. Initial Traffic Data

# %%
df = pd.read_csv('../data/raw/all_crashes.csv')

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample of first few rows:")
display(df.head())

print("\nDataset Info:")
df.info()

# %%
# Convert the 'geom' column from WKT to geometry objects
df['geometry'] = df['geom'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

gdf['DateOfIncident'] = pd.to_datetime(gdf['DateOfIncident'])
gdf['DATE'] = gdf['DateOfIncident'].dt.date
gdf['day_of_week'] = gdf['DateOfIncident'].dt.day_name()
gdf['month'] = gdf['DateOfIncident'].dt.month_name()
display(gdf[['DateOfIncident', 'DATE','day_of_week', 'month']].head())

# %% [markdown]
# Keep info about location, environment, weather/surface conditions, temporal for feature consideration

# %%
selected_cols = ['geometry', 'CountyNameTxt', 'Region', 
                'WeatherCde_txtCARTO', 'SurfaceConditionCde_txtCARTO', 'RdwyTypeCde_txtCARTO', 'DATE']
new_df = gdf[selected_cols].copy()

new_df.head()

# %% [markdown]
# ## 6. Handle Missing Values for Traffic

# %%
missing_values = new_df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# %% [markdown]
# Find values that are null, missing, or unknown and mark them as 'unknown' for uniformity
#

# %%
for col in new_df.columns:
    print(f"{col}: {new_df[col].unique()}")

# %%
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)
from src.preprocessing import standardize_unknown_values
    
# Standardize unknown/missing values
new_df = standardize_unknown_values(new_df)

# Calculate statistics for unknown values
total_records = len(new_df)
unknown_stats = {}

for col in new_df.columns:
    if col != 'geom' and col != 'date':  # Skip non-categorical columns
        unknown_count = (new_df[col] == 'unknown').sum()
        unknown_percentage = (unknown_count / total_records) * 100
        unknown_stats[col] = {
            'unknown_count': unknown_count,
            'unknown_percentage': unknown_percentage
        }

# Display the results
print(f"Total records in dataset: {total_records:,}\n")
print("Unknown Values Statistics:")
print("-" * 60)
print(f"{'Column':<30} {'Count':>10} {'Percentage':>12}")
print("-" * 60)
for col, stats in unknown_stats.items():
    print(f"{col:<30} {stats['unknown_count']:>10,} {stats['unknown_percentage']:>11.2f}%")

# %%
display(new_df.describe(include='all'))
display(pd.DataFrame(new_df.columns.tolist(), columns=["Column Name"]))

# %% [markdown]
# ## 7. Save Cleaned Dataset for Traffic (Optional)

# %%
""""
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Save the cleaned dataset
cleaned_file_path = os.path.join(processed_dir, 'cleaned_crashes.csv')
new_df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")
"""

# %% [markdown]
# ## 8. Combine Datasets

# %%
from src.preprocessing import build_master_dataset

# Testing on small sample first (2016-2017)
master_df_final2 = build_master_dataset('2016-01-01', '2017-12-31', new_df, final_weather_df, gdf)
master_df_final2.tail()

# %%
# Full dataset (2016-01-01 to 2025-09-29)
master_df = build_master_dataset('2016-01-01', '2025-09-28', new_df, final_weather_df, gdf)

pd.set_option('display.max_columns', None)
master_df = master_df.sort_values(by='DATE', ascending=True)



# %%
# Add date features BEFORE saving
master_df['day_of_week'] = master_df['DATE'].dt.dayofweek  # 0=Monday, 6=Sunday
master_df['day_of_year'] = master_df['DATE'].dt.dayofyear  # 1-365
master_df['month'] = master_df['DATE'].dt.month  # 1-12

# Reorder columns so crash_tomorrow is last
cols = [c for c in master_df.columns if c != 'crash_tomorrow']
cols.append('crash_tomorrow')
master_df = master_df[cols]

master_df.describe()

# SAVE (Optional) 
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)
final_df_path = os.path.join(processed_dir, 'master_df.csv')
master_df.to_csv(final_df_path, index=False)

# %%
master_df.tail(2)

# %%
cutoff = pd.Timestamp("2024-01-01")
df_training = master_df[master_df["DATE"] < cutoff]
df_testing = master_df[master_df["DATE"] >= cutoff]

df_training.to_csv(os.path.join(processed_dir, 'master_training.csv'), index=False)
df_testing.to_csv(os.path.join(processed_dir, 'master_testing.csv'), index=False)

# %%
