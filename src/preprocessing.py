import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def generate_grid(gdf, cell_size_m=5000, crs="EPSG:5070"):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    cell_size_m: grid cell size in meters
    crs: metric CRS to build grid in
    returns: GeoDataFrame of grid cells in EPSG:4326 (lat/lon)
    """

    # Set crs if GeoDataFrame doesn't already contain it
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Get gdf in metric crs
    gdf_m = gdf.to_crs(crs)

    # Set minimum and maximum bounds
    minx, miny, maxx, maxy = gdf_m.total_bounds
    minx -= cell_size_m
    miny -= cell_size_m
    maxx += cell_size_m
    maxy += cell_size_m

    # Get gridlines
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    # Generate cells
    cells = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
                square = Polygon([
                    (x, y),
                    (x + cell_size_m, y),
                    (x + cell_size_m, y + cell_size_m),
                    (x, y + cell_size_m)
                ])
                cells.append({"cell_id": f"{i}_{j}", "geometry": square})
            
    # Generate the GeoDataFrame
    grid_m = gpd.GeoDataFrame(cells, crs=crs)

    # Converte back to EPSG:4326 (lat/long)
    grid = grid_m.to_crs("EPSG:4326")
    
    return grid

def reduce_gdf_area(gdf, min_lat, max_lat, min_long, max_long):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    min_lat: minimum latitude bounds
    max_lat: maximum latitude bounds
    min_long: minimum longitude bounds
    max_long: maximum longitude bounds
    returns: GeoDataFrame only containing points within the bounds
    """
    return gdf[(gdf.geometry.y >= min_lat) & (gdf.geometry.y <= max_lat) & (gdf.geometry.x >= min_long) & (gdf.geometry.x <= max_long)]

def generate_grid_tc_metro_area(gdf, cell_size_m=500):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    cell_size_m: grid cell size in meters
    returns: GeoDataFrame of grid cells in EPSG:4326, bounded around the Twin Cities metro area
    """

    # Remove all points outside the Twin Cities metro area (rough approximation of metro area as a rectangle and not including Wisconsin portions)
    gdf = reduce_gdf_area(gdf, 44.545707, 45.681479, -94.390340, -92.759918)

    # Call generate_grid with adjusted GDF
    return generate_grid(gdf, cell_size_m)

def create_cell_day_df(crash_gdf, grid_gdf, start_date, end_date):
    """
    crash_gdf: GeoDataFrame with crash information
    grid_gdf: GeoDataFrame with 'cell_id' and 'geometry' columns
    start_date: string or pd.Timestamp, date of earliest crash in return DataFrame (inclusive)
    end_date: strings or pd.Timestamp, dates of latest crashes in return DataFrame (exclusive - last day of crashes used for 'crash_tomorrow' label)
    returns: DataFrame with columns ['cell_id', 'date', 'num_crashes', 'crash_tomorrow']
    """

    # Convert DateOfIncident to datetime, truncate time (only worrying about date)
    crash_gdf['date'] = pd.to_datetime(crash_gdf['DateOfIncident']).dt.floor('D')

    # Filter crash_gdf to crashes occuring between start_date and end_date
    mask = (crash_gdf['date'] >= pd.to_datetime(start_date)) & (crash_gdf['date'] <= pd.to_datetime(end_date))
    crash_gdf = crash_gdf[mask].copy()

    # Make sure the CRS matches
    if crash_gdf.crs != grid_gdf.crs:
        crash_gdf = crash_gdf.to_crs(grid_gdf.crs)

    # Assign each crash to a cell
    crash_with_cells = gpd.sjoin(crash_gdf, grid_gdf[['cell_id', 'geometry']], how='inner', predicate='within')

    # Aggregate crashes by cell/day combination
    crash_cells = (
        crash_with_cells
        .groupby(['cell_id', 'date'])
        .size()
        .reset_index(name='num_crashes')
    )

    # All possible cells and days
    all_cells = grid_gdf['cell_id'].unique()
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with each unique combination of cell and date
    full_index = pd.MultiIndex.from_product([all_cells, all_days], names=['cell_id', 'date'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Add crash data to DataFrame with all cell and date combinations
    cell_day_df = full_df.merge(crash_cells, on=['cell_id', 'date'], how='left')
    cell_day_df['num_crashes'] = cell_day_df['num_crashes'].fillna(0).astype(int)

    # Sort by cell_id and date
    cell_day_df = cell_day_df.sort_values(['cell_id', 'date'])

    # Add crash_tomorrow classifier
    cell_day_df['crash_tomorrow'] = cell_day_df.groupby('cell_id')['num_crashes'].shift(-1)
    cell_day_df['crash_tomorrow'] = (cell_day_df['crash_tomorrow'] > 0).astype(int)

    # Remove data from the final day (as they all have crash_tommorow = 0 no matter what)
    cell_day_df = cell_day_df[cell_day_df['date'] < pd.to_datetime(end_date)]

    return cell_day_df
    
def standardize_unknown_values(df):
    # Define unknown variations once
    unknown_variations = {
        'unknown', 'did not describe', 'not described', 'missing',
        'other', 'unspecified', 'n/a', 'na', 'none', 'no input',
        'not known at time of the crash'
    }
    
    # Process each column in the dataframe
    for col in df.columns:
        if col in ['geometry', 'geom', 'date']:
            continue
        
        # Handle numeric columns separately to avoid string operations
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna('unknown')
            continue
            
        # For string columns, create a mask for all variations of unknown
        mask = df[col].isna()  # Get null values
        
        # Convert to string and lowercase for comparison
        str_series = df[col].astype(str).str.lower().str.strip()
        mask |= str_series.isin(unknown_variations)
        
        # Apply the replacement where mask is True
        df.loc[mask, col] = 'unknown'
    
    return df

def build_master_dataset(start_date, end_feature_date, crash_df, weather_df, gdf):
    """
    Build the final master_df with crash + weather features and crash_tomorrow target.
    """
    #Parse dates & compute target end date 
    start = pd.Timestamp(start_date)
    feature_end = pd.Timestamp(end_feature_date)
    target_end = feature_end + pd.Timedelta(days=1)

    # Make sure DATE is a proper date column 
    crash_df['DATE'] = pd.to_datetime(crash_df['DATE']).dt.normalize()

    max_crash_date = crash_df['DATE'].max()
    #print(f'Max crash DATE in crash_df: {max_crash_date.date()}')
    if max_crash_date < target_end:
        raise ValueError(
            f"Not enough crash data to compute crash_tomorrow up to {feature_end.date()}.\n"
            f"Max crash DATE in crash_df is {max_crash_date.date()}, "
            f"but we need at least {target_end.date()}."
        )

    # Do the same DATE normalization for weather
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE']).dt.normalize()

    # Crash + weather merge on DATE 
    print('Merging crash and weather data...')
    master_df = (
        crash_df
        .merge(weather_df, on='DATE', how='left')
        .sort_values('DATE')
    )

    # Grid + cell_day_df (target) 
    print('Generating grid...')
    grid = generate_grid_tc_metro_area(gdf)  

    print(f'Generating cell_day_df from {start.date()} to {target_end.date()}...')
    # pass Timestamps directly
    cell_day_df = create_cell_day_df(gdf, grid, start, target_end)

    # Filter by feature window and attach cell_id via spatial join 
    print('Filtering master_df to feature window and attaching cell_id...')
    master_gdf = gpd.GeoDataFrame(master_df, geometry='geometry', crs='EPSG:4326')

    feature_mask = master_gdf['DATE'].between(start, feature_end)
    master_gdf_filtered = master_gdf[feature_mask].copy()

    master_with_cell = gpd.sjoin(master_gdf_filtered, grid, how='left', predicate='within')
    master_with_cell = master_with_cell.drop(columns=['index_right'], errors='ignore')

    # Merge features (X) with target (Y) on cell_id + DATE 
    print('Merging with cell_day_df to attach crash_tomorrow...')
    master_df_final = pd.merge(
        master_with_cell,
        cell_day_df[['cell_id', 'date', 'crash_tomorrow']],
        left_on=['cell_id', 'DATE'],
        right_on=['cell_id', 'date'],
        how='left'
    )

    # Drop rows with no target
    master_df_final.dropna(subset=['crash_tomorrow'], inplace=True) 

    # Clean up temp columns
    master_df_final = master_df_final.drop(columns=['date'], errors='ignore') 

    print(f'Final record count: {len(master_df_final)}')
    print('crash_tomorrow value counts:')
    print(master_df_final['crash_tomorrow'].value_counts(dropna=False))

    return master_df_final

def get_count_matrix(df, date_col='DATE', cell_col='cell_id'):
    """
    Transforms a long-form dataframe into a (T, C) matrix of counts.
    Returns: matrix, index (dates), columns (cell_ids)
    """
    # Ensure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate counts if not already aggregated
    if 'num_crashes' in df.columns:
        counts = df.groupby([date_col, cell_col])['num_crashes'].sum().reset_index()
    else:
        counts = df.groupby([date_col, cell_col]).size().reset_index(name='num_crashes')
        
    pivot_df = counts.pivot(index=date_col, columns=cell_col, values='num_crashes').fillna(0)
    return pivot_df.to_numpy(), pivot_df.index, pivot_df.columns

def build_cell_day_master(start_date, end_date, weather_df, grid_gdf, crash_gdf):
    """
    Builds a master dataset with one row per (cell, day), merging in weather data.
    This is useful for general-purpose crash prediction modeling.
    """
    # 1. Create the base grid-day structure with crash counts
    cell_day_df = create_cell_day_df(crash_gdf, grid_gdf, start_date, end_date)
    
    # 2. Prepare weather data (ensure DATE is datetime and normalize)
    weather_df = weather_df.copy()
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE']).dt.normalize()
    
    # 3. Merge weather into the cell-day grid
    # cell_day_df has 'date', weather_df has 'DATE'
    master_cell_day = pd.merge(
        cell_day_df,
        weather_df,
        left_on='date',
        right_on='DATE',
        how='left'
    )
    
    # Clean up and sort
    master_cell_day = master_cell_day.drop(columns=['DATE'], errors='ignore')
    master_cell_day = master_cell_day.sort_values(['cell_id', 'date'])
    
    return master_cell_day
