"""
AI-Powered Recommendation Engine
Generates pricing and operational recommendations with Nova validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from .nova_explainer import NovaExplainer


class RecommendationEngine:
    """Generate and validate business recommendations"""
    
    def __init__(self, predictions_df: pd.DataFrame, nova_explainer: NovaExplainer, 
                 elasticity: float = -0.5):
        """
        Initialize recommendation engine
        
        Args:
            predictions_df: DataFrame with predictions
            nova_explainer: NovaExplainer instance
            elasticity: Price elasticity of demand
        """
        self.predictions = predictions_df
        self.nova = nova_explainer
        self.elasticity = elasticity
    
    def generate_recommendations(self, top_n: int = 10) -> List[Dict]:
        """
        Generate top N recommendations across all types
        
        Args:
            top_n: Number of recommendations to generate
            
        Returns:
            List of recommendation dicts
        """
        all_recs = []
        
        # 1. Surge pricing opportunities
        all_recs.extend(self._recommend_surge_pricing())
        
        # 2. Promotional discounts
        all_recs.extend(self._recommend_discounts())
        
        # 3. Supply reallocation
        all_recs.extend(self._recommend_reallocation())
        
        # 4. Cost optimization
        all_recs.extend(self._recommend_cost_optimization())
        
        # 5. Event-based strategies (if applicable)
        all_recs.extend(self._recommend_event_strategies())
        
        # Sort by expected impact and return top N
        all_recs = sorted(all_recs, key=lambda x: x.get('expected_impact_$', 0), reverse=True)
        
        return all_recs[:top_n]
    
    def _recommend_surge_pricing(self) -> List[Dict]:
        """Identify zones where surge pricing would increase profit"""
        recommendations = []
        
        # Find high-demand, underpriced zones
        median_fare = self.predictions['avg_fare_pred'].median()
        
        high_demand_zones = self.predictions[
            (self.predictions['demand_pred'] > self.predictions['demand_pred'].quantile(0.75)) &
            (self.predictions['avg_fare_pred'] < median_fare)
        ]
        
        for idx, row in high_demand_zones.head(3).iterrows():
            # Calculate potential impact of 15% surge
            surge_pct = 0.15
            new_demand = row['demand_pred'] * (1 + self.elasticity * surge_pct)
            new_revenue = new_demand * row['avg_fare_pred'] * (1 + surge_pct)
            revenue_increase = new_revenue - row['revenue_pred']
            
            rec = {
                'type': 'surge_pricing',
                'zone_id': int(row['cluster_id']),
                'action': f'Implement 15% surge pricing',
                'current_demand': float(row['demand_pred']),
                'current_revenue': float(row['revenue_pred']),
                'expected_revenue': float(new_revenue),
                'expected_impact_$': float(revenue_increase),
                'confidence': 'high' if row['demand_pred'] > 10 else 'medium',
                'rationale': f'High demand ({row["demand_pred"]:.1f} trips) with below-median pricing'
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _recommend_discounts(self) -> List[Dict]:
        """Identify zones where discounts could boost utilization"""
        recommendations = []
        
        # Find low-demand zones with potential
        low_demand_zones = self.predictions[
            (self.predictions['demand_pred'] < self.predictions['demand_pred'].quantile(0.25)) &
            (self.predictions['margin_pred'] > 0.20)  # Maintain healthy margin
        ]
        
        for idx, row in low_demand_zones.head(2).iterrows():
            # Calculate 10% discount impact
            discount_pct = -0.10
            new_demand = row['demand_pred'] * (1 + self.elasticity * discount_pct)
            new_revenue = new_demand * row['avg_fare_pred'] * (1 + discount_pct)
            revenue_increase = new_revenue - row['revenue_pred']
            
            if revenue_increase > 0:  # Only recommend if revenue increases
                rec = {
                    'type': 'promotional_discount',
                    'zone_id': int(row['cluster_id']),
                    'action': '10% promotional discount',
                    'current_demand': float(row['demand_pred']),
                    'expected_demand': float(new_demand),
                    'expected_impact_$': float(revenue_increase),
                    'confidence': 'medium',
                    'rationale': f'Low utilization ({row["demand_pred"]:.1f} trips) with margin buffer'
                }
                
                recommendations.append(rec)
        
        return recommendations
    
    def _recommend_reallocation(self) -> List[Dict]:
        """Recommend driver reallocation between zones"""
        recommendations = []
        
        # Find mismatches: high demand zones vs low demand zones
        high_demand = self.predictions.nlargest(5, 'demand_pred')
        low_demand = self.predictions.nsmallest(5, 'demand_pred')
        
        if not high_demand.empty and not low_demand.empty:
            best_source = low_demand.iloc[0]
            best_dest = high_demand.iloc[0]
            
            if best_dest['demand_pred'] > 2 * best_source['demand_pred']:
                rec = {
                    'type': 'supply_reallocation',
                    'action': f'Reallocate drivers from zone {int(best_source["cluster_id"])} to zone {int(best_dest["cluster_id"])}',
                    'source_zone': int(best_source['cluster_id']),
                    'dest_zone': int(best_dest['cluster_id']),
                    'source_demand': float(best_source['demand_pred']),
                    'dest_demand': float(best_dest['demand_pred']),
                    'expected_impact_$': float(best_dest['avg_fare_pred'] * 2),  # Estimated revenue from 2 more trips
                    'confidence': 'medium',
                    'rationale': f'Zone {int(best_dest["cluster_id"])} has {best_dest["demand_pred"]/best_source["demand_pred"]:.1f}x higher demand'
                }
                
                recommendations.append(rec)
        
        return recommendations
    
    def _recommend_cost_optimization(self) -> List[Dict]:
        """Recommend cost reduction opportunities"""
        recommendations = []
        
        # Find low-margin zones despite good demand
        low_margin_zones = self.predictions[
            (self.predictions['margin_pred'] < 0.20) &
            (self.predictions['demand_pred'] > self.predictions['demand_pred'].median())
        ]
        
        for idx, row in low_margin_zones.head(2).iterrows():
            # Calculate 5% ops cost reduction impact
            current_ops_cost = row['revenue_pred'] * 0.05  # Assuming 5% ops cost
            cost_reduction = current_ops_cost * 0.20  # 20% reduction
            
            rec = {
                'type': 'cost_optimization',
                'zone_id': int(row['cluster_id']),
                'action': 'Optimize operational costs (20% reduction target)',
                'current_margin': float(row['margin_pred']),
                'expected_margin': float(row['margin_pred'] + (cost_reduction / row['revenue_pred'])),
                'expected_impact_$': float(cost_reduction),
                'confidence': 'medium',
                'rationale': f'Good demand ({row["demand_pred"]:.1f} trips) but low margin ({row["margin_pred"]*100:.1f}%)'
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _recommend_event_strategies(self) -> List[Dict]:
        """Recommend event-based strategies"""
        recommendations = []
        
        # Check for weekend/rush hour patterns
        if 'is_weekend' in self.predictions.columns:
            weekend_data = self.predictions[self.predictions['is_weekend'] == 1]
            if not weekend_data.empty:
                top_weekend_zone = weekend_data.nlargest(1, 'revenue_pred').iloc[0]
                
                rec = {
                    'type': 'event_strategy',
                    'zone_id': int(top_weekend_zone['cluster_id']),
                    'action': 'Weekend driver incentive program',
                    'expected_impact_$': float(top_weekend_zone['revenue_pred'] * 0.10),  # 10% uplift
                    'confidence': 'medium',
                    'rationale': f'Top weekend zone with ${top_weekend_zone["revenue_pred"]:.2f} average revenue'
                }
                
                recommendations.append(rec)
        
        return recommendations
    
    def validate_with_nova(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Use Nova to validate and enhance recommendations
        
        Args:
            recommendations: List of recommendation dicts
            
        Returns:
            Enhanced recommendations with AI validation
        """
        validated_recs = []
        
        for rec in recommendations:
            validated_rec = self.nova.validate_recommendation(rec)
            validated_recs.append(validated_rec)
        
        return validated_recs
    
    def generate_recommendation_report(self, recommendations: List[Dict]) -> str:
        """
        Generate a comprehensive recommendation report with Nova
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            AI-generated report
        """
        # Prepare summary
        rec_summary = []
        for i, rec in enumerate(recommendations[:5], 1):
            rec_summary.append(
                f"{i}. {rec['type']}: {rec['action']} "
                f"(Expected Impact: ${rec.get('expected_impact_$', 0):.2f})"
            )
        
        prompt = f"""You are creating an executive recommendation report for a ride-sharing platform.

TOP 5 RECOMMENDATIONS:
{chr(10).join(rec_summary)}

Create a professional report (4-6 sentences) that:
1. Prioritizes the most impactful recommendations
2. Explains the strategic rationale
3. Highlights total expected revenue impact
4. Provides implementation timeline suggestion

Keep it executive-level and actionable."""

        return self.nova.generate_explanation(prompt, max_tokens=500, temperature=0.7)


if __name__ == "__main__":
    print("Recommendation Engine initialized. Use with predictions DataFrame and NovaExplainer.")
