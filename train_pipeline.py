"""
Complete Training Pipeline Script for Revenue Intelligence Platform

This script demonstrates the full end-to-end pipeline from raw data to trained models.
Run this after setting up the project structure.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_raw_data
from src.features.engineering import engineer_all_features
from src.models.demand_model import train_demand_model
from src.utils.config_loader import config

def main():
    print("=" * 80)
    print("REVENUE INTELLIGENCE PLATFORM - TRAINING PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Load Data
    print("📁 Step 1: Loading NYC Taxi Data...")
    print("-" * 80)
    
    # Update this path to your data location
    data_path = "C:/Users/2594j/Downloads/New folder/cleaned_data.parquet"
    
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Date range: {df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Please update the data_path in this script to your data location")
        return
    
    print()
    
    # Step 2: Preprocess Data
    print("🧹 Step 2: Preprocessing Data...")
    print("-" * 80)
    
    df_clean = preprocess_raw_data(df, verbose=True)
    
    print()
    
    # Step 3: Feature Engineering
    print("⚙️  Step 3: Engineering Features...")
    print("-" * 80)
    
    df_featured, kmeans_model = engineer_all_features(df_clean, fit_clusters=True)
    
    print()
    print(f"✅ Feature engineering complete!")
    print(f"   Final shape: {df_featured.shape}")
    print(f"   Features created: {df_featured.columns.tolist()}")
    
    print()
    
    # Step 4: Train Demand Model
    print("🤖 Step 4: Training Demand Forecasting Model...")
    print("-" * 80)
    
    model_save_path = Path(__file__).parent.parent / "models" / "demand_model.pkl"
    demand_model, test_metrics = train_demand_model(
        df_featured,
        save_path=str(model_save_path)
    )
    
    print()
    
    # Step 5: Summary
    print("=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print("📊 Final Results:")
    print(f"   ✅ Demand Model Test MAPE: {test_metrics['mape']:.4f} ({test_metrics['mape']*100:.2f}%)")
    print(f"   ✅ Demand Model Test R²: {test_metrics['r2']:.4f}")
    print(f"   ✅ Demand Model Test RMSE: {test_metrics['rmse']:.2f}")
    print()
    print(f"📁 Models saved to: {model_save_path.parent}")
    print()
    print("🚀 Next Steps:")
    print("   1. Review model performance metrics")
    print("   2. Train revenue forecasting model")
    print("   3. Build what-if simulator")
    print("   4. Integrate AWS Nova for explanations")
    print("   5. Deploy to AWS Lambda")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
