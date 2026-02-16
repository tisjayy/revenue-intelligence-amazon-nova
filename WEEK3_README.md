# Week 3: Interactive Dashboard

Complete Streamlit dashboard with Nova-powered AI insights.

## Features

### 1. Executive Dashboard
- Key performance metrics (demand, revenue, profit, margin)
- Top 10 zones by revenue visualization
- Demand distribution across all zones
- AI-generated executive summary from Nova

### 2. Nova Chat
- Natural language query interface
- Ask questions about revenue predictions in plain English
- Chat history tracking
- Sample questions provided
- Real-time AI responses

### 3. What-If Simulator
- Zone-level pricing simulations
- Adjustable price changes (-30% to +30%)
- Price elasticity modeling
- Revenue/demand/profit impact calculations
- Visual comparison charts
- Nova AI analysis of simulation results

### 4. Zone Explorer
- Deep dive into individual zone performance
- AI-powered zone insights
- Time series revenue visualization
- Peak hour analysis
- Average metrics display

### 5. Recommendations
- AI-powered recommendation generation
- 5 recommendation types:
  - Surge pricing strategies
  - Promotional discounts
  - Supply reallocation
  - Cost optimization
  - Event-based strategies
- Nova validation for each recommendation
- Executive summary report
- Expected impact calculations

## Installation

```bash
pip install streamlit plotly
```

## Usage

### Launch Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Prerequisites

1. **Week 1 predictions must exist:**
   ```bash
   python week1_quick_start.py
   ```
   This creates `data/test_predictions.csv`

2. **AWS Bedrock credentials configured:**
   ```powershell
   $env:AWS_BEARER_TOKEN_BEDROCK="ABSKQmVkcm9ja0FQSUtleS1oNXR0..."
   ```

## Navigation

Use the sidebar to navigate between pages:
- **Executive Dashboard** - High-level overview and AI summary
- **Nova Chat** - Ask questions in natural language
- **What-If Simulator** - Test pricing strategies
- **Zone Explorer** - Analyze specific zones
- **Recommendations** - Get AI-powered action items

## Sample Questions for Nova Chat

1. "Which zones have the highest revenue potential?"
2. "What's the average profit margin?"
3. "Which zones should we focus on for improvement?"
4. "Compare zones 237 and 161"
5. "What zones have lowest demand?"

## Architecture

```
dashboard.py
├── Executive Dashboard (Page 1)
│   ├── Metrics display
│   ├── Revenue charts
│   └── AI executive summary
├── Nova Chat (Page 2)
│   ├── Query input
│   ├── Chat history
│   └── Sample questions
├── What-If Simulator (Page 3)
│   ├── Zone selection
│   ├── Price adjustment controls
│   ├── Simulation results
│   └── Nova analysis
├── Zone Explorer (Page 4)
│   ├── Zone insights
│   ├── Time series charts
│   └── AI summary
└── Recommendations (Page 5)
    ├── Recommendation generation
    ├── Nova validation
    └── Executive report
```

## Session State

The dashboard maintains state for:
- `predictions`: Week 1 prediction data
- `nova`: NovaExplainer instance
- `query_handler`: QueryHandler instance
- `rec_engine`: RecommendationEngine instance
- `chat_history`: Nova conversation history

## Performance

- Data caching via session state
- Lazy loading of AI models
- Efficient DataFrame operations
- ~15 Nova API calls per full demo (~$0.10)

## Troubleshooting

### "Predictions not found"
Run Week 1 pipeline first:
```bash
python week1_quick_start.py
```

### "Authentication error"
Set bearer token:
```powershell
$env:AWS_BEARER_TOKEN_BEDROCK="your_token_here"
```

### Slow performance
- Reduce prediction dataset size
- Use data sampling for visualizations
- Cache expensive operations

## Demo Flow

1. **Start with Executive Dashboard** - Show high-level metrics
2. **Click "Generate Executive Summary"** - Demonstrate Nova AI
3. **Navigate to Nova Chat** - Ask sample questions
4. **Try What-If Simulator** - Adjust pricing, get AI analysis
5. **Generate Recommendations** - Show actionable insights

## Next Steps (Week 4)

- Polish UI/UX with animations
- Add demo recording
- Deploy to AWS
- Create presentation slides
- Rehearse demo script
