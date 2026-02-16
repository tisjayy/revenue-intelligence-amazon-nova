# Week 2: AWS Nova Integration

## Quick Setup

### 1. Install AWS SDK
```powershell
pip install boto3
```

### 2. Configure AWS Credentials

**Option A: Environment Variables (Recommended for development)**
```powershell
$env:AWS_ACCESS_KEY_ID="your-access-key"
$env:AWS_SECRET_ACCESS_KEY="your-secret-key"
$env:AWS_DEFAULT_REGION="us-east-1"
```

**Option B: AWS CLI Configuration**
```powershell
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Region: us-east-1
# Output format: json
```

**Option C: Credentials File**
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-1
```

### 3. Enable Bedrock Nova Access

1. Go to AWS Console → Bedrock
2. Select "Model access" in left sidebar
3. Click "Modify model access"
4. Enable "Amazon Nova Lite"
5. Submit request (usually instant approval)

### 4. Run Week 2 Demo

```powershell
python week2_nova_demo.py
```

## What Week 2 Does

1. **Nova Explanation Engine** - Generates AI explanations for:
   - Revenue changes ("Why is Zone 237 revenue dropping 15%?")
   - Anomaly detection explanations
   - Executive summaries
   - Recommendation validation

2. **Natural Language Queries** - Answer questions like:
   - "Which zones have highest revenue potential?"
   - "What's the average profit margin?"
   - "Which zones need improvement?"

3. **AI-Powered Recommendations** - Generate 5 types:
   - **Surge Pricing**: High-demand, underpriced zones
   - **Promotional Discounts**: Low-demand zones with margin buffer
   - **Supply Reallocation**: Move drivers from low to high demand
   - **Cost Optimization**: Reduce ops costs in low-margin zones
   - **Event Strategies**: Weekend/rush hour incentives

4. **Nova Validation** - AI reviews and enhances each recommendation

## Expected Output

- Executive summary of all predictions
- Answers to 3 natural language queries
- Top 5 actionable recommendations with AI validation
- Revenue change explanations for high-variance zones
- Comprehensive recommendation report
- Saved insights in `data/nova_insights.yaml`

## Estimated Costs

- Nova Lite: $0.06 per 1M input tokens, $0.24 per 1M output tokens
- Demo usage: ~10-15 API calls
- **Total cost: < $0.10**

## Troubleshooting

**Error: "NoCredentialsError"**
- Set environment variables (see Option A above)

**Error: "AccessDeniedException"**
- Enable Bedrock Nova access (see step 3 above)
- Wait 2-3 minutes after enabling

**Error: "ModelNotFoundException"**
- Verify region is `us-east-1`
- Ensure Nova Lite is enabled in Bedrock console

**Error: "test_predictions.csv not found"**
- Run `python week1_quick_start.py` first

## Demo Queries to Try

After running the demo, test the query handler with:
- "Why will revenue increase in zone 161?"
- "Compare zones 237 and 161"
- "What's the profit outlook for tomorrow?"
- "Which zones are underperforming?"

## Next Steps (Week 3)

- Streamlit dashboard with Nova chat interface
- Autonomous monitoring agent (Lambda)
- Real-time anomaly detection + explanations
- DynamoDB for storing AI insights
