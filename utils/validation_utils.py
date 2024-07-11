import pandas as pd
import geopandas as gpd
from typing import Union



def validate_predictions(
        predictions: Union[pd.DataFrame, gpd.GeoDataFrame],
        ground_truth: Union[pd.DataFrame, gpd.GeoDataFrame],
        pred_col: str = 'snow_depth',
        gt_col: str = 'observed_snow_depth',
        buffer_distance: Union[int, float] = 3,
) -> pd.DataFrame:
    """
    Validate and compare predictions against ground truth data based on spatial proximity.
    
    Parameters:
    - predictions (Union[pd.DataFrame, gpd.GeoDataFrame]): The predicted data. Expected to have 'easting' and 'northing' columns if it's a DataFrame.
    - ground_truth (Union[pd.DataFrame, gpd.GeoDataFrame]): The ground truth data. Expected to have 'easting' and 'northing' columns if it's a DataFrame.
    - pred_col (str, optional): Column name for predicted snow depth in the predictions dataframe. Defaults to 'snow_depth'.
    - gt_col (str, optional): Column name for observed snow depth in the ground truth dataframe. Defaults to 'observed_snow_depth'.
    - buffer_distance (Union[int, float], optional): Buffer distance to create around each ground truth point. Defaults to 3.
    
    Returns:
    - pd.DataFrame: A DataFrame with averaged predicted snow depth for each ground truth point, along with the observed snow depth.
    """
    
    # Convert DataFrames to GeoDataFrames if they aren't already
    if isinstance(predictions, pd.DataFrame):
        predictions = gpd.GeoDataFrame(
            predictions,
            geometry=gpd.points_from_xy(predictions.easting, predictions.northing),
            crs='EPSG:32613',
        )
    
    if isinstance(ground_truth, pd.DataFrame):
        ground_truth = gpd.GeoDataFrame(
            ground_truth,
            geometry=gpd.points_from_xy(ground_truth.easting, ground_truth.northing),
            crs='EPSG:32613',
        )

    # Create a buffer around the ground truth points
    ground_truth['geometry'] = ground_truth.buffer(buffer_distance)

    # Spatially join the predictions and ground truth
    joined = gpd.sjoin(ground_truth, predictions, how='inner', predicate='intersects')

    # For each in-situ point, calculate the average predicted snow_depth
    averaged = joined.groupby(joined.index)[pred_col].mean().reset_index()

    # Merge back to get observed
    results = pd.merge(averaged, ground_truth[[gt_col]], left_on='index', right_index=True).drop('index', axis=1)

    return results