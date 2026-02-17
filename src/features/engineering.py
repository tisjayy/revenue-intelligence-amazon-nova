"""
Feature engineering for demand and revenue forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
import holidays
from sklearn.cluster import MiniBatchKMeans
from ..utils.config_loader import config


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = 'tpep_pickup_datetime') -> pd.DataFrame:
    """
    Add temporal features from timestamp
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with added temporal features
    """
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract temporal components
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Rush hour indicator
    rush_hours = config.get('features.rush_hours')
    morning_rush = rush_hours.get('morning', [7, 8, 9])
    evening_rush = rush_hours.get('evening', [16, 17, 18, 19])
    
    df['is_rush_hour'] = (
        df['hour_of_day'].isin(morning_rush + evening_rush)
    ).astype(int)
    
    # Time of day category
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_of_day'] = df['hour_of_day'].apply(categorize_time)
    
    return df


def add_holiday_features(df: pd.DataFrame, timestamp_col: str = 'tpep_pickup_datetime') -> pd.DataFrame:
    """
    Add holiday indicators
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with holiday features
    """
    # US holidays
    us_holidays = holidays.US()
    
    # Extract date
    df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
    
    # Check if holiday
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)
    
    # Day before/after holiday
    df['day_before_holiday'] = df['date'].apply(
        lambda x: (x + pd.Timedelta(days=1)) in us_holidays
    ).astype(int)
    
    df['day_after_holiday'] = df['date'].apply(
        lambda x: (x - pd.Timedelta(days=1)) in us_holidays
    ).astype(int)
    
    # Drop temporary date column
    df = df.drop('date', axis=1)
    
    return df


def create_time_bins(df: pd.DataFrame, timestamp_col: str = 'tpep_pickup_datetime') -> pd.DataFrame:
    """
    Create time bins for aggregation (10-minute intervals)
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with pickup_bins column
    """
    bin_size = config.get('timeseries.bin_size_seconds', 600)
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Get start of month
    start_of_month = df[timestamp_col].min().replace(day=1, hour=0, minute=0, second=0)
    
    # Calculate bin number
    df['pickup_bins'] = (
        (df[timestamp_col] - start_of_month).dt.total_seconds() / bin_size
    ).astype(int)
    
    return df


def cluster_locations(
    df: pd.DataFrame, 
    n_clusters: int = None,
    fit: bool = True,
    kmeans_model = None
) -> Tuple[pd.DataFrame, MiniBatchKMeans]:
    """
    Cluster pickup locations using MiniBatchKMeans
    
    Args:
        df: DataFrame with pickup coordinates
        n_clusters: Number of clusters (from config if None)
        fit: Whether to fit new model or use existing
        kmeans_model: Pre-fitted KMeans model (if fit=False)
    
    Returns:
        Tuple of (DataFrame with cluster assignments, KMeans model)
    """
    if n_clusters is None:
        n_clusters = config.n_clusters
    
    batch_size = config.get('clustering.batch_size', 10000)
    random_state = config.get('clustering.random_state', 42)
    
    coords = df[['pickup_latitude', 'pickup_longitude']].values
    
    if fit:
        # Fit new model
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state
        )
        df['pickup_cluster'] = kmeans.fit_predict(coords)
    else:
        # Use existing model
        if kmeans_model is None:
            raise ValueError("kmeans_model must be provided when fit=False")
        kmeans = kmeans_model
        df['pickup_cluster'] = kmeans.predict(coords)
    
    # Add cluster center coordinates
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['cluster_lat', 'cluster_lon']
    )
    cluster_centers['pickup_cluster'] = cluster_centers.index
    
    df = df.merge(cluster_centers, on='pickup_cluster', how='left')
    
    return df, kmeans


def aggregate_to_cluster_timeseries(
    df: pd.DataFrame,
    agg_columns: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate trips to cluster-timeseries level
    
    Args:
        df: DataFrame with pickup_cluster and pickup_bins
        agg_columns: Columns to aggregate (default: count, revenue)
    
    Returns:
        Aggregated DataFrame
    """
    if agg_columns is None:
        agg_columns = ['total_amount', 'trip_distance', 'passenger_count']
    
    # Count trips per cluster-time bin
    agg_dict = {'tpep_pickup_datetime': 'count'}  # Count as demand
    
    # Add sum aggregations for other columns
    for col in agg_columns:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Group by cluster and time bin
    grouped = df.groupby(['pickup_cluster', 'pickup_bins']).agg(agg_dict).reset_index()
    
    # Rename count column to 'demand'
    grouped = grouped.rename(columns={'tpep_pickup_datetime': 'demand'})
    
    # Calculate average fare
    if 'total_amount' in grouped.columns:
        grouped['avg_fare'] = grouped['total_amount'] / grouped['demand']
        grouped['revenue'] = grouped['total_amount']
    
    return grouped


def create_lag_features(
    df: pd.DataFrame, 
    target_col: str = 'demand',
    n_lags: int = None
) -> pd.DataFrame:
    """
    Create lag features for time series
    
    Args:
        df: DataFrame sorted by time with cluster groups
        target_col: Column to create lags for
        n_lags: Number of lag periods (from config if None)
    
    Returns:
        DataFrame with lag features
    """
    if n_lags is None:
        n_lags = config.get('timeseries.lag_features', 5)
    
    # Sort by cluster and time
    df = df.sort_values(['pickup_cluster', 'pickup_bins'])
    
    # Create lag features per cluster
    for lag in range(1, n_lags + 1):
        df[f'ft_{lag}'] = df.groupby('pickup_cluster')[target_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'demand',
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling average features
    
    Args:
        df: DataFrame with time series data
        target_col: Column to calculate rolling averages for
        windows: List of window sizes (from config if None)
    
    Returns:
        DataFrame with rolling features
    """
    if windows is None:
        windows = config.get('timeseries.rolling_windows', [7, 30])
    
    # Sort by cluster and time
    df = df.sort_values(['pickup_cluster', 'pickup_bins'])
    
    # Create rolling features per cluster
    for window in windows:
        # Shift by 1 to avoid look-ahead bias (exclude current observation)
        df[f'rolling_avg_{window}'] = df.groupby('pickup_cluster')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
    
    return df


def create_exponential_moving_average(
    df: pd.DataFrame,
    target_col: str = 'demand',
    alpha: float = None
) -> pd.DataFrame:
    """
    Create exponential moving average feature
    
    Args:
        df: DataFrame with time series data
        target_col: Column to calculate EMA for
        alpha: Smoothing factor (from config if None)
    
    Returns:
        DataFrame with exp_avg column
    """
    if alpha is None:
        alpha = config.get('features.exponential_moving_avg_alpha', 0.3)
    
    # Sort by cluster and time
    df = df.sort_values(['pickup_cluster', 'pickup_bins'])
    
    # Calculate EMA per cluster (shift to avoid look-ahead bias)
    df['exp_avg'] = df.groupby('pickup_cluster')[target_col].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean().shift(1)
    )
    
    return df


def engineer_all_features(
    df: pd.DataFrame,
    kmeans_model = None,
    fit_clusters: bool = True
) -> Tuple[pd.DataFrame, MiniBatchKMeans]:
    """
    Complete feature engineering pipeline
    
    Args:
        df: Raw preprocessed DataFrame
        kmeans_model: Existing KMeans model (if fit_clusters=False)
        fit_clusters: Whether to fit new clustering model
    
    Returns:
        Tuple of (Featured DataFrame, KMeans model)
    """
    print("Starting feature engineering pipeline...")
    print("=" * 60)
    
    # Temporal features
    print("Adding temporal features...")
    df = add_temporal_features(df)
    
    # Holiday features
    print("Adding holiday features...")
    df = add_holiday_features(df)
    
    # Time bins
    print("Creating time bins...")
    df = create_time_bins(df)
    
    # Cluster locations
    print(f"Clustering locations into {config.n_clusters} clusters...")
    df, kmeans = cluster_locations(df, fit=fit_clusters, kmeans_model=kmeans_model)
    
    # Aggregate to cluster-timeseries level
    print("Aggregating to cluster-timeseries level...")
    df_agg = aggregate_to_cluster_timeseries(df)
    
    # Create lag features
    print("Creating lag features...")
    df_agg = create_lag_features(df_agg, target_col='demand')
    
    # Create rolling features
    print("Creating rolling average features...")
    df_agg = create_rolling_features(df_agg, target_col='demand')
    
    # Create exponential moving average
    print("Creating exponential moving average...")
    df_agg = create_exponential_moving_average(df_agg, target_col='demand')
    
    # Drop rows with NaN lag features
    df_agg = df_agg.dropna()
    
    print("=" * 60)
    print(f"Feature engineering complete! Shape: {df_agg.shape}")
    print(f"Features: {df_agg.columns.tolist()}")
    
    return df_agg, kmeans
