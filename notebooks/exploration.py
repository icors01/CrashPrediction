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

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)

# %%
df = pd.read_csv('../data/raw/all_crashes.csv')
df.head()

# %%
from shapely import wkt
from src.preprocessing import generate_grid_tc_metro_area

df['geometry'] = df['geom'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

grid = generate_grid_tc_metro_area(gdf)
grid.head()


# %%
from src.preprocessing import create_cell_day_df

cell_day_df = create_cell_day_df(gdf, grid, '2017-01-01', '2025-01-01')
cell_day_df.head()

# %%
cell_day_df.info()
cell_day_df.describe()
print(cell_day_df['num_crashes'].value_counts())
print(cell_day_df['crash_tomorrow'].mean())

# %%
pd.crosstab(cell_day_df['cell_id'], cell_day_df['crash_tomorrow'])



# %%
cell_day_df['weekday'] = cell_day_df['date'].dt.day_name()

weekly = cell_day_df.groupby('weekday')['num_crashes'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

weekly.plot(kind='bar', figsize=(7,4), title='Total Crashes by Weekday')

# %%
cell_day_df['month'] = cell_day_df['date'].dt.month
monthly = cell_day_df.groupby('month')['num_crashes'].sum()

monthly.plot(kind='line', marker='o', title='Monthly Crash Totals')

# %%
import matplotlib.pyplot as plt

gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='EPSG:4326')

gdf.plot(markersize=1, alpha=0.3)
plt.title("Crash Locations in Minnesota")
plt.show()

# %%
cell_crashes = cell_day_df.groupby('cell_id')['num_crashes'].sum().reset_index()

grid = grid.merge(cell_crashes, on='cell_id', how='left').fillna(0)

grid.plot(column='num_crashes', cmap='Reds', legend=True, figsize=(8,6))
plt.title("Total Crashes per Grid Cell")
plt.show()

# %%

# %%
