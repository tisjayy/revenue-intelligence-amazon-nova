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
        import re
        question_lower = question.lower()
        context_parts = []
        filtered_data = self.predictions.copy()
        
        # Parse time filter (4pm, 16:00, etc.)
        hour_filter = None
        time_patterns = [
            (r'(\d{1,2})\s*pm', lambda h: int(h) + 12 if int(h) != 12 else 12),
            (r'(\d{1,2})\s*am', lambda h: int(h) if int(h) != 12 else 0),
            (r'(\d{1,2}):00', lambda h: int(h)),
            (r'hour\s+(\d{1,2})', lambda h: int(h))
        ]
        
        for pattern, converter in time_patterns:
            match = re.search(pattern, question_lower)
            if match:
                hour_filter = converter(match.group(1))
                break
        
        if hour_filter is not None and 'hour' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['hour'] == hour_filter]
            context_parts.append(f"FILTERED BY TIME: {hour_filter}:00 ({len(filtered_data)} records)")
        
        # Parse day filter (monday, tuesday, etc.)
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        day_filter = None
        for day_name, day_num in day_map.items():
            if day_name in question_lower:
                day_filter = day_num
                break
        
        if day_filter is not None and 'day_of_week' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['day_of_week'] == day_filter]
            day_name = [k for k, v in day_map.items() if v == day_filter][0].capitalize()
            context_parts.append(f"FILTERED BY DAY: {day_name} ({len(filtered_data)} records)")
        
        # Handle "maximum rides" or "most rides" queries
        time_filtered = hour_filter is not None or day_filter is not None
        if any(keyword in question_lower for keyword in ['maximum', 'max', 'most', 'highest', 'where should i', 'should i be']):
            if 'ride' in question_lower or 'demand' in question_lower or 'trip' in question_lower:
                if len(filtered_data) > 0:
                    # Find zone with maximum demand in filtered data
                    max_row = filtered_data.loc[filtered_data['demand_pred'].idxmax()]
                    
                    # Build clear, direct answer
                    filter_desc = []
                    if hour_filter is not None:
                        filter_desc.append(f"{hour_filter}:00")
                    if day_filter is not None:
                        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        filter_desc.append(day_names[day_filter])
                    
                    time_context = " at " + " ".join(filter_desc) if filter_desc else ""
                    
                    context_parts.append(f"\n**ANSWER: Zone {int(max_row['cluster_id'])}**")
                    context_parts.append(f"This zone has the HIGHEST demand{time_context}:")
                    context_parts.append(f"Demand: {max_row['demand_pred']:.1f} trips")
                    context_parts.append(f"Revenue: ${max_row['revenue_pred']:.2f}")
                    
                    # Add context about other high-demand zones
                    if len(filtered_data) >= 3:
                        other_zones = filtered_data.nlargest(3, 'demand_pred')[['cluster_id', 'demand_pred']]
                        other_zones = other_zones[other_zones['cluster_id'] != max_row['cluster_id']].head(2)
                        if len(other_zones) > 0:
                            context_parts.append(f"\nAlternative options{time_context}:")
                            for idx, row in other_zones.iterrows():
                                context_parts.append(f"- Zone {int(row['cluster_id'])}: {row['demand_pred']:.1f} trips")
                    
                    # Do NOT add global statistics for time-specific queries
                    return '\n'.join(context_parts)
                else:
                    context_parts.append("No data matches the specified filters.")
                    return '\n'.join(context_parts)
        
        # Overall summary (only for non-time-filtered general questions)
        if not time_filtered and (len(context_parts) == 0 or 'average' in question_lower or 'overall' in question_lower):
            context_parts.append(f"Total Predictions: {len(self.predictions)} zone-time combinations")
            context_parts.append(f"Avg Demand: {self.predictions['demand_pred'].mean():.1f} trips")
            context_parts.append(f"Avg Revenue: ${self.predictions['revenue_pred'].mean():.2f}")
            context_parts.append(f"Avg Margin: {self.predictions['margin_pred'].mean()*100:.1f}%")
        
        # Zone-specific lookup (when zone number is explicitly mentioned)
        if 'zone' in question_lower and not any(keyword in question_lower for keyword in ['which zone', 'what zone']):
            zone_match = re.search(r'zone\s+(\d+)', question_lower)
            if zone_match:
                zone_id = int(zone_match.group(1))
                zone_data = self.predictions[self.predictions['cluster_id'] == zone_id]
                if not zone_data.empty:
                    context_parts.append(f"\nZone {zone_id} Details:")
                    context_parts.append(f"- Demand: {zone_data['demand_pred'].mean():.1f} trips")
                    context_parts.append(f"- Revenue: ${zone_data['revenue_pred'].mean():.2f}")
                    context_parts.append(f"- Profit: ${zone_data['profit_pred'].mean():.2f}")
        
        # Profitability queries
        if 'profit' in question_lower or 'margin' in question_lower:
            high_margin = filtered_data.nlargest(5, 'margin_pred')[['cluster_id', 'margin_pred', 'profit_pred']]
            context_parts.append(f"\nHighest Margin Zones:")
            for idx, row in high_margin.iterrows():
                context_parts.append(f"  Zone {int(row['cluster_id'])}: {row['margin_pred']*100:.1f}% margin, ${row['profit_pred']:.2f} profit")
        
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
