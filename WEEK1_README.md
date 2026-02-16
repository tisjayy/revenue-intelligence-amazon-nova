# Revenue Intelligence Platform - Week 1 Quick Start

## 🚀 Get Started in 3 Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Week 1 implementation
python week1_quick_start.py

# 3. Check results
ls data/
ls models/
```

## ✅ What This Does

**Week 1 Complete Implementation** (Days 1-7):

1. **Loads & Cleans Data** - NYC taxi parquet files with proper filtering
2. **Feature Engineering** - Temporal features (hour, day, rush_hour, weekend)
3. **Geographic Clustering** - 40-zone segmentation with K-means
4. **Demand Forecasting** - XGBoost model with lag features & temporal patterns
5. **Revenue Forecasting** - XGBoost predicting total revenue per zone/time
6. **Profit Calculations** - Platform margin, driver costs, processing fees
7. **Price Elasticity** - Estimates demand response to price changes
8. **What-If Simulator** - Test pricing scenarios before implementation

## 📊 Expected Results

**Target Accuracy**:
- Demand MAPE: < 12%
- Revenue MAPE: < 18%  
- Profit Margin: ~20-25%

**Outputs**:
- `models/demand_model.pkl` - Trained demand forecaster
- `models/revenue_model.pkl` - Trained revenue predictor
- `models/cluster_model.pkl` - Geographic clustering model
- `data/test_predictions.csv` - All predictions with profit calculations
- `data/week1_summary.yaml` - Performance metrics

## 🎮 Using the What-If Simulator

After running the script, use the simulator:

```python
import pickle
import pandas as pd
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load predictions
predictions = pd.read_csv('data/test_predictions.csv')

# Load simulator function from week1_quick_start.py
from week1_quick_start import simulate_price_change

# Test scenario: +15% price increase
sample = predictions.iloc[100]
result = simulate_price_change(sample, price_change_pct=0.15)

print(f"Revenue Impact: ${result['delta']['revenue_delta_$']:,.2f}")
print(f"Profit Impact: ${result['delta']['profit_delta_$']:,.2f}")
```

##  Next Steps

**Week 2: Nova Integration** (THE winning differentiator!)

1. Natural language business queries
2. AI-powered revenue explanations
3. Recommendation engine
4. Autonomous monitoring agent

Run: `python week2_nova_integration.py` (coming next!)

## 📁 Project Structure

```
revenue-intelligence/
├── config/
│   └── config.yaml          # All configuration
├── data/
│   ├── test_predictions.csv # Model outputs
│   └── week1_summary.yaml   # Performance metrics
├── models/
│   ├── demand_model.pkl
│   ├── revenue_model.pkl
│   └── cluster_model.pkl
├── notebooks/
│   └── week1_revenue_intelligence_implementation.ipynb
├── week1_quick_start.py     # ← Run this!
└── README.md
```

## 🐛 Troubleshooting

**ImportError: No module named 'xgboost'**
```bash
pip install xgboost scikit-learn pandas numpy pyyaml
```

**FileNotFoundError: parquet file not found**
- Check paths in `config/config.yaml`
- Ensure yellow_tripdata_2015-01.parquet is in C:/Users/2594j/Downloads/

**Model accuracy lower than expected**
- Normal! First run uses default hyperparameters
- Fine-tune in config.yaml after seeing baseline results

## 💡 Tips

1. **First run**: Just execute `python week1_quick_start.py` and review results
2. **Customize**: Edit `config/config.yaml` to adjust filters, hyperparameters
3. **Iterate**: Retrain models with different features/parameters
4. **Visualize**: Load predictions in Jupyter for charts/analysis

---

**Status**: Week 1 ✅ Complete | Week 2 🚧 Next | Week 3-4 📅 Planned
