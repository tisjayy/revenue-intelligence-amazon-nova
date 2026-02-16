"""
Week 2: Nova Integration Demo
Demonstrates AI-powered revenue intelligence with AWS Bedrock Nova

Prerequisites:
- Run week1_quick_start.py first
- AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Bedrock Nova access enabled in us-east-1
"""

import pandas as pd
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ai.nova_explainer import NovaExplainer, test_nova_connection
from ai.query_handler import QueryHandler
from ai.recommendations import RecommendationEngine

print("=" * 80)
print("WEEK 2: NOVA INTEGRATION DEMO")
print("=" * 80)

# ============================================================================
# 1. TEST NOVA CONNECTION
# ============================================================================
print("\n[1/7] Testing Nova connection...")
if not test_nova_connection():
    print("\n❌ CRITICAL: Nova connection failed!")
    print("\nPlease ensure:")
    print("1. AWS credentials are set (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("2. Bedrock Nova access is enabled in your AWS account")
    print("3. You're using region us-east-1")
    sys.exit(1)

# ============================================================================
# 2. LOAD WEEK 1 PREDICTIONS
# ============================================================================
print("\n[2/7] Loading Week 1 predictions...")
try:
    predictions = pd.read_csv('data/test_predictions.csv')
    print(f"Loaded {len(predictions):,} predictions")
    print(f"Zones: {predictions['cluster_id'].nunique()}")
    print(f"Avg Revenue: ${predictions['revenue_pred'].mean():.2f}")
except FileNotFoundError:
    print("\n❌ ERROR: data/test_predictions.csv not found!")
    print("Please run week1_quick_start.py first.")
    sys.exit(1)

# ============================================================================
# 3. INITIALIZE AI COMPONENTS
# ============================================================================
print("\n[3/7] Initializing AI components...")
nova = NovaExplainer(region='us-east-1')
query_handler = QueryHandler(predictions, nova)
rec_engine = RecommendationEngine(predictions, nova)
print("✅ All AI components ready")

# ============================================================================
# 4. GENERATE EXECUTIVE SUMMARY
# ============================================================================
print("\n[4/7] Generating executive summary with Nova...")
print("\n" + "-" * 80)
summary = nova.generate_executive_summary(predictions)
print("EXECUTIVE SUMMARY:")
print(summary)
print("-" * 80)

# ============================================================================
# 5. DEMONSTRATE NATURAL LANGUAGE QUERIES
# ============================================================================
print("\n[5/7] Testing natural language queries...")

queries = [
    "Which zones have the highest revenue potential?",
    "What's the average profit margin across all zones?",
    "Which zones should we focus on for improvement?"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*80}")
    print(f"QUERY {i}: {query}")
    print(f"{'='*80}")
    answer = query_handler.answer_query(query)
    print(f"\nANSWER:\n{answer}")

# ============================================================================
# 6. GENERATE AND VALIDATE RECOMMENDATIONS
# ============================================================================
print(f"\n[6/7] Generating recommendations...")
recommendations = rec_engine.generate_recommendations(top_n=5)

print(f"\nGenerated {len(recommendations)} recommendations")
print("\n" + "=" * 80)
print("TOP RECOMMENDATIONS:")
print("=" * 80)

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['type'].upper().replace('_', ' ')}")
    print(f"   Zone: {rec.get('zone_id', 'Multiple')}")
    print(f"   Action: {rec['action']}")
    print(f"   Expected Impact: ${rec.get('expected_impact_$', 0):,.2f}")
    print(f"   Confidence: {rec['confidence']}")
    print(f"   Rationale: {rec['rationale']}")

# Validate top recommendation with Nova
print(f"\n{'='*80}")
print("NOVA VALIDATION OF TOP RECOMMENDATION:")
print(f"{'='*80}")
validated = nova.validate_recommendation(recommendations[0])
print(f"\n{validated['ai_validation']}")

# ============================================================================
# 7. DEMONSTRATE REVENUE CHANGE EXPLANATION
# ============================================================================
print(f"\n[7/7] Demonstrating revenue change explanations...")

# Find a zone with significant variance
zone_variance = predictions.groupby('cluster_id')['revenue_pred'].std()
high_variance_zone = zone_variance.idxmax()
zone_data = predictions[predictions['cluster_id'] == high_variance_zone]

if len(zone_data) >= 2:
    row1 = zone_data.iloc[0]
    row2 = zone_data.iloc[-1]
    
    context = {
        'hour': int(row2.get('hour', 12)),
        'day_of_week_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(row2.get('day_of_week', 0))],
        'is_weekend': bool(row2.get('is_weekend', 0)),
        'is_rush_hour': bool(row2.get('is_rush_hour', 0)),
        'demand_trend': 'increasing' if row2['demand_pred'] > row1['demand_pred'] else 'decreasing'
    }
    
    print(f"\n{'='*80}")
    print(f"REVENUE CHANGE EXPLANATION - ZONE {high_variance_zone}")
    print(f"{'='*80}")
    print(f"\nCurrent Revenue: ${row1['revenue_pred']:.2f}")
    print(f"Predicted Revenue: ${row2['revenue_pred']:.2f}")
    print(f"Change: {((row2['revenue_pred'] - row1['revenue_pred']) / row1['revenue_pred'] * 100):+.1f}%")
    
    explanation = nova.explain_revenue_change(
        zone_id=high_variance_zone,
        current_revenue=row1['revenue_pred'],
        predicted_revenue=row2['revenue_pred'],
        context=context
    )
    
    print(f"\nNOVA EXPLANATION:\n{explanation}")

# ============================================================================
# 8. GENERATE RECOMMENDATION REPORT
# ============================================================================
print(f"\n{'='*80}")
print("COMPREHENSIVE RECOMMENDATION REPORT:")
print(f"{'='*80}")

report = rec_engine.generate_recommendation_report(recommendations)
print(f"\n{report}")

# ============================================================================
# 9. SAVE AI-GENERATED INSIGHTS
# ============================================================================
print(f"\n[Saving] Storing AI insights...")

insights = {
    'generated_at': pd.Timestamp.now().isoformat(),
    'executive_summary': summary,
    'top_recommendations': recommendations,
    'recommendation_report': report
}

with open('data/nova_insights.yaml', 'w') as f:
    yaml.dump(insights, f)

print("✅ Saved to data/nova_insights.yaml")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 WEEK 2 COMPLETE!")
print("=" * 80)

print(f"\n✅ Deliverables:")
print(f"   [X] Nova explanation engine (revenue changes, anomalies)")
print(f"   [X] Natural language query handler ({len(queries)} demo queries)")
print(f"   [X] Recommendation engine ({len(recommendations)} recommendations)")
print(f"   [X] AI validation and enhancement")
print(f"   [X] Executive reporting")

print(f"\n📁 Saved Files:")
print(f"   - data/nova_insights.yaml (AI-generated insights)")

print(f"\n📊 Nova API Usage:")
print(f"   - ~{7 + len(queries) + len(recommendations)} API calls")
print(f"   - Estimated cost: <$0.10 (Nova Lite pricing)")

print(f"\n🚀 NEXT: Week 3 - Dashboard + Autonomous Monitoring")
print("=" * 80)
