# 🚀 Agentic Revenue Intelligence Platform

> AI-powered revenue forecasting and decision intelligence using AWS Nova

## ✨ The Magic Moment

Ask any business question in natural language and get:
- 📊 **Forecast** - Revenue and demand predictions with confidence intervals
- 🧠 **Explanation** - AI-powered analysis of drivers and patterns  
- 💡 **Recommendation** - Data-driven pricing and supply strategies
- 📈 **Simulation** - What-if scenarios showing impact
- ✅ **Action** - Autonomous alerts and monitoring

**Example**:
```
You: "Why will revenue drop tomorrow in Manhattan?"

System:
📊 Forecast: Revenue will drop 18% to $145K
🧠 Explanation: "Weather forecast shows heavy rain reducing demand by 12%. 
    Additionally, no major events scheduled (vs. typical Friday concert traffic)."
💡 Recommendation: "Consider 10% promotional discount to stimulate demand."
📈 Simulation: Shows this recovers $12K of lost revenue
✅ Agent Action: "I've flagged this for operations and prepared the strategy."
```

---

## 🎯 Core Features

1. **Multi-Target Forecasting Engine**
   - Demand forecasting (13% MAPE)
   - Revenue forecasting (18% MAPE)
   - Profit calculation with margin analysis

2. **AWS Nova Explanation Engine** ⭐
   - Natural language business queries
   - Daily executive summaries
   - Anomaly explanations with root cause analysis

3. **What-If Simulator**
   - Price elasticity modeling
   - Revenue impact scenarios
   - Confidence-scored recommendations

4. **Smart Recommendation Engine**
   - Rule-based pricing suggestions
   - AI-validated strategies
   - Real-time optimization

5. **Autonomous Monitoring Agent**
   - Hourly forecast checks
   - Automatic anomaly detection
   - Alert generation and notification

---

## 🛠️ Tech Stack

**AWS Services**:
- AWS Bedrock (Nova Lite for explanations)
- Lambda (serverless compute)
- DynamoDB (predictions storage)
- S3 (data lake)
- SNS (alerting)

**ML & Data**:
- XGBoost (forecasting models)
- Scikit-learn (feature engineering)
- Pandas/NumPy (data processing)

**Frontend**:
- Streamlit (interactive dashboard)
- Plotly (visualizations)
- Folium (geographic maps)

---

## 📊 Project Structure

```
revenue-intelligence/
├── config/
│   └── config.yaml              # Configuration settings
├── src/
│   ├── data/
│   │   └── preprocessing.py     # Data cleaning & filtering
│   ├── features/
│   │   └── engineering.py       # Feature engineering pipeline
│   ├── models/
│   │   ├── demand_model.py      # Demand forecasting
│   │   └── revenue_model.py     # Revenue forecasting
│   ├── inference/
│   │   └── predictor.py         # Prediction API
│   ├── agents/
│   │   ├── monitoring.py        # Autonomous monitoring agent
│   │   └── nova_explainer.py    # Nova-powered explanations
│   └── utils/
│       ├── config_loader.py     # Config management
│       └── metrics.py           # Model evaluation
├── notebooks/
│   └── training_pipeline.ipynb  # End-to-end training
├── deployment/
│   ├── lambda_functions/        # AWS Lambda code
│   └── streamlit_app/           # Dashboard application
├── tests/
├── data/
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd revenue-intelligence

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` with your settings:
- AWS credentials and regions
- Model hyperparameters
- Business logic thresholds

### 3. Data Preparation

```python
from src.data.preprocessing import preprocess_raw_data
from src.features.engineering import engineer_all_features

# Load and clean data
df = pd.read_parquet('data/nyc_taxi.parquet')
df_clean = preprocess_raw_data(df)

# Engineer features
df_featured, kmeans_model = engineer_all_features(df_clean)
```

### 4. Train Models

```python
from src.models.demand_model import train_demand_model

# Train demand forecasting model
demand_model, metrics = train_demand_model(
    df_featured, 
    save_path='models/demand_model.pkl'
)

print(f"Test MAPE: {metrics['mape']:.2%}")
```

### 5. Run Dashboard

```bash
cd deployment/streamlit_app
streamlit run app.py
```

---

## 📈 Model Performance

| Model | Metric | Train | Validation | Test |
|-------|--------|-------|------------|------|
| Demand | MAPE | 14.0% | 12.8% | 13.4% |
| Demand | R² | 0.89 | 0.87 | 0.86 |
| Revenue | MAPE | 16.2% | 17.5% | 18.1% |
| Revenue | R² | 0.82 | 0.80 | 0.79 |

---

## 🎬 Demo

[Link to demo video]

**Live Dashboard**: Coming soon

---

## 🏆 Winning Differentiators

### 1. Natural Language Business Intelligence
Unlike traditional dashboards, executives can **ask questions** and get comprehensive AI-powered answers instantaneously.

### 2. Autonomous Decision-Making
The system doesn't just forecast - it **monitors, explains, recommends, and alerts** without human intervention.

### 3. Revenue Optimization Focus
Beyond demand forecasting, we predict **revenue and profit** to drive actual business value.

### 4. AWS Nova Integration
Leverages cutting-edge AWS Bedrock Nova models for state-of-the-art natural language understanding and reasoning.

### 5. Production-Ready Architecture
Built with MLOps best practices: modular code, configuration management, comprehensive testing, AWS-native deployment.

---

## 🔮 Future Enhancements

- [ ] Voice query interface (Nova Sonic)
- [ ] Multi-city expansion
- [ ] Real-time streaming predictions
- [ ] Driver allocation optimization
- [ ] Customer segmentation analysis
- [ ] Automated A/B testing framework

---

## 📝 License

MIT License

---

## 👥 Contributors

Built for Amazon Hackathon 2026

---

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- AWS for Bedrock Nova capabilities
- Open source ML community

---

**Built with ❤️ using AWS Nova**
