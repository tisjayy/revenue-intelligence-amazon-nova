"""
Natural Language Query Handler
Answers business questions using predictions + Nova explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .nova_explainer import NovaExplainer


class QueryHandler:
    """Handle natural language queries about revenue predictions"""
    
    def __init__(self, predictions_df: pd.DataFrame, nova_explainer: NovaExplainer):
        """
        Initialize query handler
        
        Args:
            predictions_df: DataFrame with predictions
            nova_explainer: NovaExplainer instance
        """
        self.predictions = predictions_df
        self.nova = nova_explainer
    
    def answer_query(self, question: str) -> str:
        """
        Answer a natural language question
        
        Args:
            question: User's question
            
        Returns:
            AI-generated answer with data context
        """
        # Extract data context based on question keywords
        context = self._extract_context(question)
        
        # Build data-driven prompt
        prompt = f"""You are a revenue intelligence assistant answering questions about ride-sharing forecasts.

USER QUESTION: {question}

RELEVANT DATA:
{context}

Provide a clear, concise answer (2-4 sentences) that:
1. Directly answers the question using the data
2. Highlights key numbers
3. Provides actionable insight

Be specific and data-driven."""

        return self.nova.generate_explanation(prompt, max_tokens=400, temperature=0.6)
    
    def _extract_context(self, question: str) -> str:
        """Extract relevant data context based on question"""
        question_lower = question.lower()
        context_parts = []
        
        # Overall summary
        context_parts.append(f"Total Predictions: {len(self.predictions)} zone-time combinations")
        context_parts.append(f"Avg Demand: {self.predictions['demand_pred'].mean():.1f} trips")
        context_parts.append(f"Avg Revenue: ${self.predictions['revenue_pred'].mean():.2f}")
        context_parts.append(f"Avg Margin: {self.predictions['margin_pred'].mean()*100:.1f}%")
        
        # Zone-specific
        if 'zone' in question_lower:
            # Try to extract zone number
            import re
            zone_match = re.search(r'\d+', question)
            if zone_match:
                zone_id = int(zone_match.group())
                zone_data = self.predictions[self.predictions['cluster_id'] == zone_id]
                if not zone_data.empty:
                    context_parts.append(f"\nZone {zone_id} Details:")
                    context_parts.append(f"- Demand: {zone_data['demand_pred'].mean():.1f} trips")
                    context_parts.append(f"- Revenue: ${zone_data['revenue_pred'].mean():.2f}")
                    context_parts.append(f"- Profit: ${zone_data['profit_pred'].mean():.2f}")
        
        # Top/bottom performers
        if 'best' in question_lower or 'top' in question_lower or 'highest' in question_lower:
            top5 = self.predictions.nlargest(5, 'revenue_pred')[['cluster_id', 'revenue_pred', 'demand_pred']]
            context_parts.append(f"\nTop 5 Revenue Zones:\n{top5.to_string(index=False)}")
        
        if 'worst' in question_lower or 'bottom' in question_lower or 'lowest' in question_lower:
            bottom5 = self.predictions.nsmallest(5, 'revenue_pred')[['cluster_id', 'revenue_pred', 'demand_pred']]
            context_parts.append(f"\nBottom 5 Revenue Zones:\n{bottom5.to_string(index=False)}")
        
        # Time-based
        if 'hour' in question_lower or 'time' in question_lower:
            if 'hour' in self.predictions.columns:
                hourly_avg = self.predictions.groupby('hour')['revenue_pred'].mean()
                context_parts.append(f"\nHourly Revenue Patterns:\n{hourly_avg.to_string()}")
        
        # Profitability
        if 'profit' in question_lower or 'margin' in question_lower:
            high_margin = self.predictions.nlargest(5, 'margin_pred')[['cluster_id', 'margin_pred', 'profit_pred']]
            context_parts.append(f"\nHighest Margin Zones:\n{high_margin.to_string(index=False)}")
        
        return '\n'.join(context_parts)
    
    def get_zone_insights(self, zone_id: int) -> Dict:
        """
        Get comprehensive insights for a specific zone
        
        Args:
            zone_id: Taxi zone ID
            
        Returns:
            Dict with insights
        """
        zone_data = self.predictions[self.predictions['cluster_id'] == zone_id]
        
        if zone_data.empty:
            return {'error': f'No data found for zone {zone_id}'}
        
        insights = {
            'zone_id': zone_id,
            'avg_demand': float(zone_data['demand_pred'].mean()),
            'avg_revenue': float(zone_data['revenue_pred'].mean()),
            'avg_profit': float(zone_data['profit_pred'].mean()),
            'avg_margin': float(zone_data['margin_pred'].mean()),
            'total_predictions': len(zone_data)
        }
        
        # Peak performance time
        if 'hour' in zone_data.columns:
            peak_hour = zone_data.loc[zone_data['revenue_pred'].idxmax(), 'hour']
            insights['peak_hour'] = int(peak_hour)
        
        # Generate AI summary
        prompt = f"""Summarize this taxi zone's performance in 2-3 sentences:

Zone {zone_id} Metrics:
- Average Demand: {insights['avg_demand']:.1f} trips
- Average Revenue: ${insights['avg_revenue']:.2f}
- Average Profit: ${insights['avg_profit']:.2f}
- Profit Margin: {insights['avg_margin']*100:.1f}%
- Peak Hour: {insights.get('peak_hour', 'N/A')}:00

Focus on: performance level, profitability, and one operational recommendation."""

        insights['ai_summary'] = self.nova.generate_explanation(prompt, max_tokens=250)
        
        return insights
    
    def compare_zones(self, zone_ids: List[int]) -> str:
        """
        Compare performance of multiple zones
        
        Args:
            zone_ids: List of zone IDs to compare
            
        Returns:
            AI-generated comparison
        """
        comparison_data = []
        
        for zone_id in zone_ids:
            zone_data = self.predictions[self.predictions['cluster_id'] == zone_id]
            if not zone_data.empty:
                comparison_data.append({
                    'zone_id': zone_id,
                    'avg_demand': zone_data['demand_pred'].mean(),
                    'avg_revenue': zone_data['revenue_pred'].mean(),
                    'avg_margin': zone_data['margin_pred'].mean()
                })
        
        if not comparison_data:
            return "No data available for the specified zones."
        
        comparison_df = pd.DataFrame(comparison_data)
        
        prompt = f"""Compare these taxi zones and identify strategic insights:

ZONE COMPARISON:
{comparison_df.to_string(index=False)}

Provide:
1. Which zone is performing best and why (1 sentence)
2. Key differences between zones (1-2 sentences)
3. One strategic recommendation for the underperforming zone(s) (1 sentence)

Be specific and actionable."""

        return self.nova.generate_explanation(prompt, max_tokens=350)


if __name__ == "__main__":
    # Example usage
    print("Query Handler initialized. Use with predictions DataFrame and NovaExplainer.")
