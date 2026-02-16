"""
Metrics calculation utilities for model evaluation
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAPE as decimal (0.134 = 13.4%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    
    if mask.sum() == 0:
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': calculate_mape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'median_ae': np.median(np.abs(y_true - y_pred)),
        'mean_actual': np.mean(y_true),
        'mean_predicted': np.mean(y_pred),
        'std_actual': np.std(y_true),
        'std_predicted': np.std(y_pred)
    }
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for pretty printing
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("MODEL PERFORMANCE METRICS")
    lines.append("=" * 50)
    
    for key, value in metrics.items():
        if 'mape' in key.lower() or 'r2' in key.lower():
            lines.append(f"{key.upper():20s}: {value:.4f} ({value*100:.2f}%)")
        else:
            lines.append(f"{key.upper():20s}: {value:,.2f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def prediction_intervals(
    predictions: np.ndarray, 
    confidence: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals using bootstrap method
    
    Args:
        predictions: Array of predictions (can be 2D with multiple samples)
        confidence: Confidence level (0.9 = 90% interval)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    lower = np.percentile(predictions, lower_percentile, axis=0)
    upper = np.percentile(predictions, upper_percentile, axis=0)
    
    return lower, upper


def calculate_interval_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Calculate width of prediction intervals
    
    Args:
        lower: Lower bound of interval
        upper: Upper bound of interval
    
    Returns:
        Interval widths
    """
    return upper - lower


def identify_high_uncertainty(
    lower: np.ndarray, 
    upper: np.ndarray, 
    threshold: float = 0.3
) -> np.ndarray:
    """
    Identify periods of high uncertainty based on interval width
    
    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions
        threshold: Threshold for relative interval width
    
    Returns:
        Boolean array indicating high uncertainty periods
    """
    midpoint = (lower + upper) / 2
    interval_width = upper - lower
    
    # Avoid division by zero
    mask = midpoint != 0
    relative_width = np.zeros_like(interval_width)
    relative_width[mask] = interval_width[mask] / midpoint[mask]
    
    return relative_width > threshold
