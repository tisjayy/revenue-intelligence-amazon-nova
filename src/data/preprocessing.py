"""
Data preprocessing and cleaning for NYC Taxi data
"""
import pandas as pd
import numpy as np
from typing import Dict
from ..utils.config_loader import config


def remove_geographic_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove trips outside NYC geographic boundaries
    
    Args:
        df: DataFrame with pickup/dropoff coordinates
    
    Returns:
        Filtered DataFrame
    """
    bounds = config.nyc_bounds
    
    # Filter pickup coordinates
    df = df[
        (df['pickup_longitude'] >= bounds['lon_min']) & 
        (df['pickup_longitude'] <= bounds['lon_max']) &
        (df['pickup_latitude'] >= bounds['lat_min']) & 
        (df['pickup_latitude'] <= bounds['lat_max'])
    ]
    
    # Filter dropoff coordinates
    df = df[
        (df['dropoff_longitude'] >= bounds['lon_min']) & 
        (df['dropoff_longitude'] <= bounds['lon_max']) &
        (df['dropoff_latitude'] >= bounds['lat_min']) & 
        (df['dropoff_latitude'] <= bounds['lat_max'])
    ]
    
    return df


def calculate_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trip duration and speed
    
    Args:
        df: DataFrame with timestamp and distance info
    
    Returns:
        DataFrame with added trip_times and Speed columns
    """
    # Convert timestamps to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    if not pd.api.types.is_datetime64_any_dtype(df['tpep_dropoff_datetime']):
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Calculate trip time in minutes
    df['trip_times'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    
    # Calculate speed in mph
    df['Speed'] = df['trip_distance'] / (df['trip_times'] / 60)
    df['Speed'] = df['Speed'].replace([np.inf, -np.inf], np.nan)
    
    return df


def filter_by_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filtering thresholds from config
    
    Args:
        df: DataFrame with trip metrics
    
    Returns:
        Filtered DataFrame
    """
    filtering = config.get('filtering')
    
    # Trip distance
    df = df[
        (df['trip_distance'] > filtering['trip_distance']['min']) &
        (df['trip_distance'] < filtering['trip_distance']['max'])
    ]
    
    # Trip time
    df = df[
        (df['trip_times'] > filtering['trip_time']['min']) &
        (df['trip_times'] < filtering['trip_time']['max'])
    ]
    
    # Speed
    df = df[
        (df['Speed'] > filtering['speed']['min']) &
        (df['Speed'] < filtering['speed']['max'])
    ]
    
    # Fare
    df = df[
        (df['total_amount'] > filtering['fare']['min']) &
        (df['total_amount'] < filtering['fare']['max'])
    ]
    
    return df


def remove_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Complete outlier removal pipeline
    
    Args:
        df: Raw taxi DataFrame
        verbose: Whether to print statistics
    
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    
    if verbose:
        print(f"Initial records: {initial_count:,}")
    
    # Remove geographic outliers
    df = remove_geographic_outliers(df)
    if verbose:
        print(f"After geographic filter: {len(df):,} ({len(df)/initial_count:.1%})")
    
    # Calculate trip metrics
    df = calculate_trip_metrics(df)
    
    # Apply threshold filters
    df = filter_by_thresholds(df)
    if verbose:
        print(f"After threshold filters: {len(df):,} ({len(df)/initial_count:.1%})")
    
    # Remove nulls
    df = df.dropna(subset=['trip_times', 'Speed'])
    if verbose:
        print(f"Final records: {len(df):,}")
        print(f"Total removed: {initial_count - len(df):,} ({(initial_count - len(df))/initial_count:.1%})")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame with potential missing values
    
    Returns:
        DataFrame with missing values handled
    """
    # Drop rows with critical missing values
    critical_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                       'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude']
    
    df = df.dropna(subset=critical_columns)
    
    # Impute passenger_count with median
    if 'passenger_count' in df.columns:
        median_passengers = df['passenger_count'].median()
        df['passenger_count'] = df['passenger_count'].fillna(median_passengers)
    
    return df


def preprocess_raw_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for raw taxi data
    
    Args:
        df: Raw taxi DataFrame
        verbose: Whether to print progress
    
    Returns:
        Cleaned and processed DataFrame
    """
    if verbose:
        print("Starting data preprocessing pipeline...")
        print("=" * 60)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers
    df = remove_outliers(df, verbose=verbose)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    if verbose:
        print("=" * 60)
        print("Preprocessing complete!")
    
    return df
