"""
Demand Forecasting Model using XGBoost
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from ..utils.config_loader import config
from ..utils.metrics import calculate_all_metrics, format_metrics


class DemandForecaster:
    """XGBoost-based demand forecasting model"""
    
    def __init__(self, model_params: Dict = None):
        """
        Initialize demand forecaster
        
        Args:
            model_params: XGBoost model parameters (uses config defaults if None)
        """
        if model_params is None:
            model_params = config.get('models.xgboost_demand', {})
        
        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for training"""
        features = [
            'ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5',  # Lag features
            'cluster_lat', 'cluster_lon',  # Spatial features
            'day_of_week',  # Temporal features
            'hour_of_day',
            'is_weekend',
            'is_rush_hour',
            'is_holiday',
            'month',
            'exp_avg',  # Moving average
            'rolling_avg_7',  # Rolling averages
            'rolling_avg_30'
        ]
        
        return features
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame,
        train_ratio: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/validation/test
        
        Args:
            df: Featured DataFrame
            train_ratio: Training data ratio (from config if None)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if train_ratio is None:
            train_ratio = config.get('split.train_ratio', 0.7)
        
        val_ratio = config.get('split.validation_ratio', 0.15)
        
        # Sort by pickup_bins to ensure chronological split
        df = df.sort_values('pickup_bins')
        
        # Split per cluster to maintain temporal order
        train_list, val_list, test_list = [], [], []
        
        for cluster_id in df['pickup_cluster'].unique():
            cluster_data = df[df['pickup_cluster'] == cluster_id].copy()
            n = len(cluster_data)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_list.append(cluster_data.iloc[:train_end])
            val_list.append(cluster_data.iloc[train_end:val_end])
            test_list.append(cluster_data.iloc[val_end:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"Train size: {len(train_df):,} ({len(train_df)/len(df):.1%})")
        print(f"Validation size: {len(val_df):,} ({len(val_df)/len(df):.1%})")
        print(f"Test size: {len(test_df):,} ({len(test_df)/len(df):.1%})")
        
        return train_df, val_df, test_df
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        verbose: bool = True
    ):
        """
        Train XGBoost demand model
        
        Args:
            X_train: Training features
            y_train: Training target (demand)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Whether to print training progress
        """
        self.feature_names = X_train.columns.tolist()
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        eval_list = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, 'val'))
        
        # Train model
        if verbose:
            print("Training XGBoost demand model...")
            print(f"Parameters: {self.model_params}")
        
        self.model = xgb.train(
            params=self.model_params,
            dtrain=dtrain,
            evals=eval_list,
            verbose_eval=verbose
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type='weight')
        
        if verbose:
            print("\nTop 10 Important Features:")
            sorted_importance = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            for feat, importance in sorted_importance:
                print(f"  {feat:20s}: {importance}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features DataFrame
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Ensure columns match training
        X = X[self.feature_names]
        
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        
        return predictions
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True values
            dataset_name: Name of dataset for printing
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        metrics = calculate_all_metrics(y, predictions)
        
        print(f"\n{dataset_name} Set Performance:")
        print(format_metrics(metrics))
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = filepath.replace('.pkl', '_xgb.json')
        self.model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params
        }
        joblib.dump(metadata, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        # Load XGBoost model
        model_path = filepath.replace('.pkl', '_xgb.json')
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load metadata
        metadata = joblib.load(filepath)
        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']
        self.model_params = metadata['model_params']
        
        print(f"Model loaded from {filepath}")


def train_demand_model(
    df: pd.DataFrame,
    save_path: str = None
) -> Tuple[DemandForecaster, Dict[str, float]]:
    """
    Complete training pipeline for demand forecasting
    
    Args:
        df: Featured DataFrame with demand column
        save_path: Path to save trained model
    
    Returns:
        Tuple of (trained model, test metrics)
    """
    print("=" * 70)
    print("DEMAND FORECASTING MODEL TRAINING")
    print("=" * 70)
    
    # Initialize model
    model = DemandForecaster()
    
    # Prepare train/test split
    train_df, val_df, test_df = model.prepare_train_test_split(df)
    
    # Get feature columns
    feature_cols = model.get_feature_columns()
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['demand']
    
    X_val = val_df[feature_cols]
    y_val = val_df['demand']
    
    X_test = test_df[feature_cols]
    y_test = test_df['demand']
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train, "Train")
    val_metrics = model.evaluate(X_val, y_val, "Validation")
    test_metrics = model.evaluate(X_test, y_test, "Test")
    
    # Save model
    if save_path:
        model.save(save_path)
    
    print("=" * 70)
    print(f"✅ Demand Model Training Complete!")
    print(f"   Test MAPE: {test_metrics['mape']:.4f} ({test_metrics['mape']*100:.2f}%)")
    print(f"   Test R²: {test_metrics['r2']:.4f}")
    print("=" * 70)
    
    return model, test_metrics
