"""
Nova Explanation Engine
Generates AI-powered explanations for revenue insights using AWS Bedrock Nova
"""

import boto3
import requests
import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class NovaExplainer:
    """Generate business explanations using AWS Bedrock Nova"""
    
    def __init__(self, region: str = 'us-east-1', model_id: str = 'us.amazon.nova-lite-v1:0'):
        """
        Initialize Nova client
        
        Args:
            region: AWS region
            model_id: Bedrock Nova model identifier
        """
        self.region = region
        self.model_id = model_id
        
        # Check for bearer token (Bedrock API Key)
        self.bearer_token = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
        
        if self.bearer_token:
            # Use bearer token authentication (API Key)
            self.endpoint = f'https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke'
            self.use_bearer = True
        else:
            # Use standard AWS credentials
            self.bedrock = boto3.client('bedrock-runtime', region_name=region)
            self.use_bearer = False
    
    def generate_explanation(self, prompt: str, max_tokens: int = 2048, 
                           temperature: float = 0.7) -> str:
        """
        Call Nova to generate explanation
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            temperature: Creativity (0.0-1.0)
            
        Returns:
            Generated explanation text
        """
        try:
            # Nova API request format
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
            
            if self.use_bearer:
                # Use bearer token (API Key) via HTTPS request
                headers = {
                    'Authorization': f'Bearer {self.bearer_token}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                response = requests.post(self.endpoint, headers=headers, json=request_body)
                
                if response.status_code != 200:
                    return f"Error: HTTP {response.status_code} - {response.text}"
                
                response_body = response.json()
            else:
                # Use standard AWS credentials
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType='application/json',
                    accept='application/json'
                )
                
                response_body = json.loads(response['body'].read())
            
            # Extract text from Nova response
            if 'output' in response_body and 'message' in response_body['output']:
                message = response_body['output']['message']
                if 'content' in message and len(message['content']) > 0:
                    return message['content'][0]['text']
            
            return "No explanation generated."
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def explain_revenue_change(self, zone_id: int, current_revenue: float, 
                              predicted_revenue: float, context: Dict) -> str:
        """
        Explain why revenue is changing for a zone
        
        Args:
            zone_id: Taxi zone ID
            current_revenue: Current revenue
            predicted_revenue: Predicted revenue
            context: Additional context (hour, day, weather, etc.)
            
        Returns:
            AI-generated explanation
        """
        change_pct = ((predicted_revenue - current_revenue) / current_revenue) * 100
        direction = "increase" if change_pct > 0 else "decrease"
        
        prompt = f"""You are a revenue intelligence analyst for a ride-sharing platform.

SITUATION:
- Taxi Zone {zone_id}
- Current revenue: ${current_revenue:,.2f}
- Predicted revenue: ${predicted_revenue:,.2f}
- Change: {change_pct:+.1f}% {direction}

CONTEXT:
- Time: {context.get('hour', 'N/A')}:00
- Day: {context.get('day_of_week_name', 'N/A')}
- Is Weekend: {context.get('is_weekend', False)}
- Is Rush Hour: {context.get('is_rush_hour', False)}
- Demand Trend: {context.get('demand_trend', 'stable')}

Provide a concise, business-focused explanation (2-3 sentences) for this revenue change. Focus on:
1. Primary driver of the change
2. Actionable insight for operators

Keep it executive-friendly and data-driven."""

        return self.generate_explanation(prompt, max_tokens=300, temperature=0.6)
    
    def explain_anomaly(self, zone_id: int, metric: str, actual: float, 
                       expected: float, severity: str = "medium") -> str:
        """
        Explain an anomaly detection
        
        Args:
            zone_id: Taxi zone ID
            metric: Metric name (demand, revenue, etc.)
            actual: Actual value
            expected: Expected value
            severity: low, medium, high
            
        Returns:
            AI-generated anomaly explanation
        """
        deviation = ((actual - expected) / expected) * 100
        
        prompt = f"""You are monitoring a ride-sharing platform for operational issues.

ANOMALY DETECTED:
- Zone {zone_id}
- Metric: {metric}
- Expected: {expected:.2f}
- Actual: {actual:.2f}
- Deviation: {deviation:+.1f}%
- Severity: {severity.upper()}

Provide a brief explanation (2-3 sentences) covering:
1. Likely cause of this anomaly
2. Recommended immediate action

Be specific and actionable."""

        return self.generate_explanation(prompt, max_tokens=250, temperature=0.5)
    
    def generate_executive_summary(self, predictions_df) -> str:
        """
        Generate executive summary of daily predictions
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Executive summary text
        """
        # Calculate summary stats
        total_predicted_demand = predictions_df['demand_pred'].sum()
        total_predicted_revenue = predictions_df['revenue_pred'].sum()
        total_predicted_profit = predictions_df['profit_pred'].sum()
        avg_margin = predictions_df['margin_pred'].mean()
        
        # Top/bottom performers
        top_zones = predictions_df.nlargest(5, 'revenue_pred')[['cluster_id', 'revenue_pred']]
        bottom_zones = predictions_df.nsmallest(5, 'revenue_pred')[['cluster_id', 'revenue_pred']]
        
        prompt = f"""You are presenting revenue forecasts to executives.

DAILY FORECAST SUMMARY:
- Total Predicted Demand: {total_predicted_demand:,.0f} trips
- Total Predicted Revenue: ${total_predicted_revenue:,.2f}
- Total Predicted Profit: ${total_predicted_profit:,.2f}
- Average Margin: {avg_margin*100:.1f}%

TOP 5 REVENUE ZONES:
{top_zones.to_string(index=False)}

BOTTOM 5 REVENUE ZONES:
{bottom_zones.to_string(index=False)}

Create an executive summary (4-5 sentences) with:
1. Overall performance outlook
2. Key opportunities highlighted
3. Areas of concern (if any)
4. One strategic recommendation

Keep it high-level and business-focused."""

        return self.generate_explanation(prompt, max_tokens=400, temperature=0.7)
    
    def validate_recommendation(self, recommendation: Dict) -> Dict:
        """
        Use Nova to validate and enhance a recommendation
        
        Args:
            recommendation: Dict with recommendation details
            
        Returns:
            Enhanced recommendation with AI validation
        """
        prompt = f"""You are validating a pricing strategy recommendation.

RECOMMENDATION:
- Type: {recommendation.get('type', 'N/A')}
- Zone: {recommendation.get('zone_id', 'N/A')}
- Action: {recommendation.get('action', 'N/A')}
- Expected Impact: {recommendation.get('expected_impact', 'N/A')}
- Confidence: {recommendation.get('confidence', 'N/A')}

Provide:
1. Assessment: Is this recommendation sound? (1-2 sentences)
2. Risk: Any potential downsides? (1 sentence)
3. Enhancement: One way to improve this recommendation

Keep response concise and analytical."""

        validation = self.generate_explanation(prompt, max_tokens=300, temperature=0.6)
        
        return {
            **recommendation,
            'ai_validation': validation,
            'validated_at': datetime.now().isoformat()
        }


def test_nova_connection(region: str = 'us-east-1'):
    """Quick test to verify Nova connectivity"""
    try:
        explainer = NovaExplainer(region=region)
        test_prompt = "Say 'Nova connection successful' if you can read this."
        response = explainer.generate_explanation(test_prompt, max_tokens=50)
        print(f"✅ Nova Test: {response}")
        return True
    except Exception as e:
        print(f"❌ Nova Test Failed: {e}")
        return False


if __name__ == "__main__":
    # Test connectivity
    test_nova_connection()
