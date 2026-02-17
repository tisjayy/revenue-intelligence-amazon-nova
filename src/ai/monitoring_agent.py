"""
Autonomous Monitoring Agent - Powered by Amazon Nova
Continuously monitors revenue predictions and takes action
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from .nova_explainer import NovaExplainer


class RevenueMonitoringAgent:
    """
    Autonomous agent that monitors revenue metrics and uses Nova AI
    to investigate anomalies and recommend actions
    """
    
    def __init__(self, predictions_df: pd.DataFrame, nova: NovaExplainer):
        self.df = predictions_df
        self.nova = nova
        self.anomalies = []
        self.actions_taken = []
        
    def detect_anomalies(self) -> List[Dict]:
        """
        Detect anomalies in predictions using statistical methods
        Returns list of detected anomalies with metadata
        """
        anomalies = []
        
        # Group by zone
        zone_stats = self.df.groupby('cluster_id').agg({
            'demand_pred': ['mean', 'std', 'sum'],
            'revenue_pred': ['mean', 'std', 'sum'],
            'margin_pred': ['mean', 'std']
        })
        
        # FOCUS ON HIGH-IMPACT ZONES (top 40% by revenue - zones worth optimizing)
        revenue_threshold = zone_stats[('revenue_pred', 'sum')].quantile(0.60)
        high_impact_zones = zone_stats[zone_stats[('revenue_pred', 'sum')] >= revenue_threshold].copy()
        
        # Skip if not enough zones
        if len(high_impact_zones) < 5:
            return []
        
        # Calculate zone-specific benchmarks (compare each zone to its nearest peers by size)
        # Group zones into terciles by revenue for fairness
        high_impact_zones['revenue_tercile'] = pd.qcut(
            high_impact_zones[('revenue_pred', 'sum')], 
            q=3, 
            labels=['small', 'medium', 'large'],
            duplicates='drop'
        )
        
        underperforming_zones = []
        
        # Find underperformers within each tercile
        for tercile in ['small', 'medium', 'large']:
            tercile_zones = high_impact_zones[high_impact_zones['revenue_tercile'] == tercile]
            if len(tercile_zones) < 2:
                continue
                
            # Benchmark: 75th percentile of THIS tercile
            tercile_benchmark = tercile_zones[('revenue_pred', 'sum')].quantile(0.75)
            
            # Find zones below 90% of their tercile benchmark
            underperforming = tercile_zones[
                tercile_zones[('revenue_pred', 'sum')] < tercile_benchmark * 0.90
            ].copy()
            
            for zone_id in underperforming.index[:1]:  # Top 1 per tercile
                zone_data = self.df[self.df['cluster_id'] == zone_id]
                total_rev = float(zone_stats.loc[zone_id, ('revenue_pred', 'sum')])
                
                # Target: reach 95% of tercile benchmark (realistic peer-based target)
                target_rev = tercile_benchmark * 0.95
                potential_gain = target_rev - total_rev
                upside_pct = ((target_rev - total_rev) / total_rev) * 100  # Positive growth opportunity
                
                underperforming_zones.append((zone_id, total_rev, target_rev, potential_gain, upside_pct))
        
        # Sort by potential gain and take top 3
        underperforming_zones.sort(key=lambda x: x[3], reverse=True)
        
        for zone_id, total_rev, target_rev, potential_gain, upside_pct in underperforming_zones[:3]:
            
            anomalies.append({
                'type': 'REVENUE_OPPORTUNITY',
                'zone_id': int(zone_id),
                'severity': 'HIGH',
                'metric': 'revenue_pred',
                'current_value': total_rev,
                'expected_value': float(target_rev),
                'deviation_pct': float(upside_pct),  # Positive upside to target
                'potential_gain': float(potential_gain),
                'detected_at': datetime.now(),
                'status': 'INVESTIGATING'
            })
        
        # Detect LOW EFFICIENCY zones (high demand but low revenue per trip)
        # Use the same high-impact zones (already filtered for top 40%)
        high_impact_zones['revenue_per_trip'] = (
            high_impact_zones[('revenue_pred', 'sum')] / 
            high_impact_zones[('demand_pred', 'sum')]
        )
        
        # Target: top 25% revenue per trip within high-impact zones
        rpt_target = high_impact_zones['revenue_per_trip'].quantile(0.75)
        
        # Find zones with efficiency below 90% of target
        low_efficiency = high_impact_zones[
            high_impact_zones['revenue_per_trip'] < rpt_target * 0.90
        ].copy()
        
        # Sort by total revenue (optimize bigger zones first)
        low_efficiency = low_efficiency.sort_values(('revenue_pred', 'sum'), ascending=False)
        
        for zone_id in low_efficiency.index[:2]:
            zone_data = self.df[self.df['cluster_id'] == zone_id]
            current_rpt = float(low_efficiency.loc[zone_id, 'revenue_per_trip'])
            total_demand = float(zone_stats.loc[zone_id, ('demand_pred', 'sum')])
            
            # Target: reach 95% of top 25% benchmark (realistic improvement)
            realistic_target = rpt_target * 0.95
            potential_gain = (realistic_target - current_rpt) * total_demand
            upside_pct = ((realistic_target - current_rpt) / current_rpt) * 100  # Positive growth opportunity
            
            anomalies.append({
                'type': 'LOW_EFFICIENCY',
                'zone_id': int(zone_id),
                'severity': 'HIGH',
                'metric': 'revenue_per_trip',
                'current_value': current_rpt,
                'expected_value': float(realistic_target),
                'deviation_pct': float(upside_pct),  # Positive upside to target
                'potential_gain': float(potential_gain),
                'detected_at': datetime.now(),
                'status': 'INVESTIGATING'
            })
        
        self.anomalies = anomalies
        return anomalies
    
    def investigate_with_nova(self, anomaly: Dict) -> Dict:
        """
        Use Nova AI to investigate anomaly and propose solutions
        Shows multi-step reasoning chain
        """
        zone_id = anomaly['zone_id']
        zone_data = self.df[self.df['cluster_id'] == zone_id]
        
        # Calculate additional context
        total_demand = zone_data['demand_pred'].sum()
        total_revenue = zone_data['revenue_pred'].sum()
        revenue_per_trip = total_revenue / total_demand if total_demand > 0 else 0
        potential_gain = anomaly.get('potential_gain', 0)
        
        # Prepare context for Nova
        context = f"""REVENUE OPTIMIZATION OPPORTUNITY - Autonomous Agent Analysis

Zone ID: {zone_id}
Opportunity Type: {anomaly['type'].replace('_', ' ').title()}
Severity: {anomaly['severity']}
POTENTIAL VALUE: ${potential_gain:,.2f}

CURRENT PERFORMANCE:
- Total Demand: {total_demand:.0f} trips
- Total Revenue: ${total_revenue:,.2f}
- Revenue per Trip: ${revenue_per_trip:.2f}
- Average Margin: {zone_data['margin_pred'].mean()*100:.1f}%

BENCHMARK COMPARISON:
- Current {anomaly['metric'].replace('_', ' ').title()}: {anomaly['current_value']:,.2f}
- Expected/Median Value: {anomaly['expected_value']:,.2f}
- Performance Gap: {anomaly['deviation_pct']:.1f}%

ZONE CHARACTERISTICS:
- Peak Hour: {zone_data.groupby('hour')['demand_pred'].sum().idxmax()}:00
- Weekend vs Weekday: {"Weekend-heavy (better)" if zone_data[zone_data['is_weekend']==1]['demand_pred'].sum() > zone_data[zone_data['is_weekend']==0]['demand_pred'].sum() else "Weekday-heavy (better)"}
- Rush Hour Contribution: {zone_data[zone_data['is_rush_hour']==1]['demand_pred'].sum() / zone_data['demand_pred'].sum() * 100:.0f}% of trips

TASK: As an autonomous revenue optimization agent, provide multi-step reasoning analysis:

STEP 1 - ROOT CAUSE ANALYSIS:
Why is this zone underperforming? List 2-3 most likely root causes based on the data.

STEP 2 - OPPORTUNITY ASSESSMENT:
What is the realistic revenue uplift we can achieve? Consider market constraints and execution risks.

STEP 3 - ACTION RECOMMENDATIONS:
Propose 2-3 specific, executable actions with priority ranking.

**CRITICAL CONSTRAINTS (Must Follow):**
- Pricing adjustments: Maximum +/-15% per change (price elasticity: -0.5 typical)
- Demand stimulation must be cost-effective (ROI > 2x)
- Recommendations must be implementable within 30 days

Suggest:
- Pricing adjustments (within +/-15% limit)
- Demand stimulation (specific tactics with expected ROI)
- Operational improvements (measurable changes)

STEP 4 - IMPLEMENTATION ROADMAP:
For the highest priority action, outline quick wins (Week 1), medium-term gains (Month 1), and sustained impact (Quarter 1).

Keep analysis concise and action-oriented (max 600 words)."""
        
        # Call Nova for investigation
        investigation = self.nova.generate_explanation(context, max_tokens=800)
        
        # Parse and structure the response
        anomaly['investigation'] = investigation
        anomaly['investigated_at'] = datetime.now()
        anomaly['status'] = 'PENDING_ACTION'
        
        # Extract actionable recommendations (simplified parsing)
        lines = investigation.split('\n')
        recommendations = []
        for i, line in enumerate(lines):
            if 'solution' in line.lower() or 'recommend' in line.lower() or 'action' in line.lower():
                recommendations.append(line.strip())
        
        anomaly['recommendations'] = recommendations if recommendations else ["Investigate zone characteristics further", "Consider pricing adjustments", "Monitor for trend changes"]
        
        return anomaly
    
    def propose_action(self, anomaly: Dict) -> Dict:
        """
        Generate specific action proposal based on investigation
        """
        zone_id = anomaly['zone_id']
        zone_data = self.df[self.df['cluster_id'] == zone_id]
        
        action = {
            'anomaly_id': id(anomaly),
            'zone_id': zone_id,
            'action_type': '',
            'description': '',
            'expected_impact': 0,
            'confidence': 'MEDIUM',
            'reasoning': anomaly.get('investigation', ''),
            'proposed_at': datetime.now(),
            'status': 'PROPOSED'
        }
        
        if anomaly['type'] == 'REVENUE_OPPORTUNITY':
            # Propose comprehensive optimization
            potential_gain = anomaly.get('potential_gain', 0)
            achievable_gain = potential_gain * 0.30  # Target 30% of gap to average
            
            action['action_type'] = 'COMPREHENSIVE_OPTIMIZATION'
            action['description'] = f'Optimize pricing, driver allocation, and demand stimulation to close revenue gap'
            action['expected_impact'] = achievable_gain
            action['confidence'] = 'HIGH'
            
        elif anomaly['type'] == 'LOW_EFFICIENCY':
            # Propose pricing increase
            potential_gain = anomaly.get('potential_gain', 0)
            achievable_gain = potential_gain * 0.40  # Target 40% improvement in efficiency
            
            action['action_type'] = 'PRICE_OPTIMIZATION'
            action['description'] = f'Increase pricing by 12-15% to improve revenue per trip efficiency'
            action['expected_impact'] = achievable_gain
            action['confidence'] = 'HIGH'
            
        elif anomaly['type'] == 'LOW_REVENUE':
            # Legacy support
            current_revenue = zone_data['revenue_pred'].sum()
            potential_increase = current_revenue * 0.15
            
            action['action_type'] = 'PRICE_INCREASE'
            action['description'] = f'Increase pricing by 10% to boost revenue'
            action['expected_impact'] = potential_increase
            action['confidence'] = 'HIGH'
            
        elif anomaly['type'] == 'UNDERPERFORMING':
            # Legacy support
            current_demand = zone_data['demand_pred'].sum()
            potential_increase = current_demand * 0.20
            
            action['action_type'] = 'DEMAND_BOOST'
            action['description'] = f'Launch promotional campaign to increase demand by 20%'
            action['expected_impact'] = potential_increase * zone_data['avg_fare_pred'].mean()
            action['confidence'] = 'MEDIUM'
        
        return action
    
    def run_monitoring_cycle(self) -> Dict:
        """
        Complete monitoring cycle: Detect → Investigate → Propose
        Returns full report
        """
        report = {
            'cycle_started': datetime.now(),
            'anomalies_detected': [],
            'investigations': [],
            'actions_proposed': [],
            'status': 'RUNNING'
        }
        
        # Step 1: Detect anomalies
        anomalies = self.detect_anomalies()
        report['anomalies_detected'] = anomalies
        
        # Step 2 & 3: Investigate and propose actions for top anomalies
        for anomaly in anomalies[:3]:  # Top 3 for demo
            investigated = self.investigate_with_nova(anomaly)
            report['investigations'].append(investigated)
            
            action = self.propose_action(investigated)
            report['actions_proposed'].append(action)
        
        report['cycle_completed'] = datetime.now()
        report['status'] = 'COMPLETED'
        
        return report
    
    def get_agent_status(self) -> Dict:
        """
        Get current agent status and metrics
        """
        return {
            'status': 'ACTIVE',
            'total_zones_monitored': self.df['cluster_id'].nunique(),
            'anomalies_detected': len(self.anomalies),
            'actions_taken': len(self.actions_taken),
            'last_check': datetime.now(),
            'health': 'HEALTHY'
        }
