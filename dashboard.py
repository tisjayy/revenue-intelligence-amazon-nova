"""
Agentic AI Revenue Intelligence Platform
Built with Amazon Nova Reasoning Capabilities - Autonomous agents tackle complex real-world revenue optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import yaml
import os

try:
    if 'AWS_BEARER_TOKEN_BEDROCK' in st.secrets:
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = st.secrets['AWS_BEARER_TOKEN_BEDROCK']
    elif 'AWS_ACCESS_KEY_ID' in st.secrets:
        os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['AWS_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['AWS_SECRET_ACCESS_KEY']
except FileNotFoundError:
    pass

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ai.nova_explainer import NovaExplainer
from ai.query_handler import QueryHandler
from ai.recommendations import RecommendationEngine
from ai.monitoring_agent import RevenueMonitoringAgent

def get_zone_coordinates():
    """Returns approximate centroid coordinates for NYC taxi zones"""
    # Manhattan: 1-103, Bronx: 104-130, Brooklyn: 131-177, Queens: 178-236, Staten Island: 237-263
    coords = {}
    
    # Manhattan zones (distributed across the island)
    for i in range(1, 104):
        lat = 40.730 + (i % 50) * 0.006  # North-South spread
        lon = -73.990 - (i % 20) * 0.008  # East-West spread
        coords[i] = {'lat': lat, 'lon': lon}
    
    # Bronx zones
    for i in range(104, 131):
        lat = 40.820 + (i % 15) * 0.008
        lon = -73.900 - (i % 10) * 0.007
        coords[i] = {'lat': lat, 'lon': lon}
    
    # Brooklyn zones
    for i in range(131, 178):
        lat = 40.650 + (i % 25) * 0.007
        lon = -73.950 - (i % 18) * 0.006
        coords[i] = {'lat': lat, 'lon': lon}
    
    # Queens zones
    for i in range(178, 237):
        lat = 40.700 + (i % 30) * 0.006
        lon = -73.800 - (i % 22) * 0.005
        coords[i] = {'lat': lat, 'lon': lon}
    
    # Staten Island zones
    for i in range(237, 264):
        lat = 40.580 + (i % 12) * 0.007
        lon = -74.140 - (i % 8) * 0.006
        coords[i] = {'lat': lat, 'lon': lon}
    
    return coords

# Page config
st.set_page_config(
    page_title="Revenue Intelligence - Amazon Nova",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .recommendation-box {border-left: 4px solid #1f77b4; padding-left: 15px; margin: 15px 0;}
    .nova-response {
        background-color: #e8f4f8; 
        color: #1a1a1a;
        padding: 15px; 
        border-radius: 8px; 
        margin: 10px 0;
        border-left: 3px solid #1f77b4;
    }
    .nova-response strong {
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    try:
        st.session_state.predictions = pd.read_csv('data/test_predictions.csv')
    except FileNotFoundError:
        st.error("Error: data/test_predictions.csv not found. Please run week1_quick_start.py first.")
        st.stop()

if 'nova' not in st.session_state:
    st.session_state.nova = NovaExplainer(region='us-east-1')

if 'query_handler' not in st.session_state:
    st.session_state.query_handler = QueryHandler(
        st.session_state.predictions,
        st.session_state.nova
    )

if 'rec_engine' not in st.session_state:
    st.session_state.rec_engine = RecommendationEngine(
        st.session_state.predictions,
        st.session_state.nova
    )

if 'monitoring_agent' not in st.session_state:
    st.session_state.monitoring_agent = RevenueMonitoringAgent(
        st.session_state.predictions,
        st.session_state.nova
    )

# Sidebar navigation
st.sidebar.title("Navigation 🚕")
page = st.sidebar.radio(
    "Choose a page:",
    ["Project Overview", "Executive Dashboard", "Autonomous Agent", "Nova Chat", "What-If Simulator", "Zone Explorer", "Recommendations"]
)

# ----- PAGE 0: PROJECT OVERVIEW -----
if page == "Project Overview":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 700; margin: 0; background: linear-gradient(90deg, #1f77b4, #00b4d8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            NovaOps
        </h1>
        <p style='font-size: 1.4rem; color: #666; margin: 0.5rem 0 0 0;'>
            Autonomous Revenue Intelligence 
        </p>
        <p style='font-size: 1rem; color: #999; margin: 0.3rem 0 0 0;'>
            <strong>Amazon Nova AI Hackathon</p></strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 10px;'>
            <h3 style='color: #1f77b4; margin: 0; font-size: 2.5rem;'>14.69%</h3>
            <p style='color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>XGBoost WMAPE</p>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #f0fff4; border-radius: 10px;'>
            <h3 style='color: #10b981; margin: 0; font-size: 2.5rem;'>12.7M</h3>
            <p style='color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>NYC Taxi Trips</p>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #fff7ed; border-radius: 10px;'>
            <h3 style='color: #f59e0b; margin: 0; font-size: 2.5rem;'>Nova 2</h3>
            <p style='color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Amazon Nova Lite</p>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #fdf2f8; border-radius: 10px;'>
            <h3 style='color: #ec4899; margin: 0; font-size: 2.5rem;'>20s</h3>
            <p style='color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Agent Analysis Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Concise Overview
    st.markdown("---")
    st.markdown("### The Agentic AI Approach")
    
    # Impact statement first - most important
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1f77b4, #00b4d8); padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0; font-size: 1.3rem;'>3-5 day human analysis → 20 second autonomous agent response</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Why Agentic AI?**")
        
        # Comparison table
        comparison_data = {
            "Approach": ["Traditional ML", "Pure GenAI", "**NovaOps (Agentic AI)**"],
            "Predictions": ["✅", "❌", "✅ XGBoost (14.69% WMAPE)"],
            "Explanations": ["❌ No reasoning", "✅ Natural language", "✅ Amazon Nova reasoning"],
            "Actions": ["❌ Manual only", "❌ No execution", "✅ **Autonomous recommendations**"]
        }
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
        
        st.markdown("""
        **How it works:**
        - **The Calculator:** XGBoost on 400K samples (12.7M trips)
        - **The Analyst:** Amazon Nova 2 Lite reasoning engine
        """)
    
    with col2:
        st.markdown("**NovaOps Workflow:**")
        st.markdown("""
        <div style='background: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #4caf50;'>
        <ol style='margin: 0; padding-left: 1.5rem;'>
            <li><strong>Detect</strong> → Autonomous anomaly detection across 40 clusters</li>
            <li><strong>Investigate</strong> → Multi-step reasoning with <strong>Amazon Nova 2 Lite</strong></li>
            <li><strong>Recommend</strong> → Autonomous strategy generation</li>
            <li><strong>Execute</strong> → Human approval needed</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("▸ Business Problem")
    
    st.markdown("""
    **Complex Real-World Problem:** Ride-sharing operators face revenue leakage across 260+ zones with 
    dynamic demand patterns, price elasticity variations, and operational constraints. Optimizing revenue 
    requires continuous monitoring, multi-factor analysis, and rapid decision-making.
    
    **Traditional Approach Fails:** Human analysts manually review dashboards, run SQL queries, build 
    spreadsheets, and present findings in weekly meetings - taking 3-5 days per analysis cycle. By then, 
    opportunities are lost.
    
    **Agentic AI Solution:** Autonomous agents powered by **Amazon Nova reasoning capabilities** complete 
    the entire optimization cycle (detect → investigate → propose → validate) in 20 seconds. Agents tackle 
    complex root cause analysis using multi-step reasoning chains, considering demand patterns, pricing 
    elasticity, cost structures, and competitive dynamics simultaneously.
    """)
    
    # Data & ML Models
    st.markdown("---")
    st.subheader("▦ Data & Machine Learning Models")
    
    tab1, tab2, tab3 = st.tabs(["Data Sources", "ML Architecture", "Model Performance"])
    
    with tab1:
        st.markdown("""
        **Dataset:** NYC Taxi & Limousine Commission (TLC) Trip Records
        - **Size:** 12.7M raw trips → 12.6M cleaned (1.1% outliers removed)
        - **Spatial Structure:** 260 NYC taxi zones → 40 K-means clusters for modeling
        - **Timeframe:** Multiple months of 2015-2016 historical data
        - **Features:** Pickup/dropoff locations, timestamps, fares, distances, durations
        
        **Data Processing Pipeline:**
        1. **Geographic Assignment:** 260 NYC TLC taxi zones mapped to trips
        2. **K-means Clustering:** Reduce 260 zones → 40 clusters for ML modeling
        3. **Temporal Binning:** 10-minute aggregation → 402,197 time-series points
        4. **Feature Engineering:** Lag features (ft_1-ft_5), exponential moving avg, rush hour flags
        5. **Final ML Dataset:** 400,926 records after lag filtering
        
        **Data Quality:**
        - Geographic filtering: NYC bounding box (40.58-40.92°N, -74.15 to -73.70°W)
        - Distance filtering: 0-23 miles (99th percentile cutoff)
        - Speed filtering: 0-45.31 mph (anomaly removal)
        - Fare validation: $0-$1000 range enforcement
        """)
    
    with tab2:
        st.markdown("""
        **Machine Learning Architecture:**
        
        **Model Type:** XGBoost (Extreme Gradient Boosting)
        
        **Why XGBoost?**
        - Superior performance on tabular data vs neural networks
        - Handles non-linear relationships and interactions
        - Built-in regularization prevents overfitting
        - Highly efficient with parallel processing
        - Industry-standard for demand forecasting
        
        **Three Parallel Models:**
        1. **Demand Model:** Predicts trip volume per zone-hour
        2. **Revenue Model:** Predicts total revenue per zone-hour  
        3. **Average Fare Model:** Predicts price per trip
        
        **Input Features (11 total):**
        - Zone ID (cluster_id) - 1 of 40 clusters
        - Time features: hour, day_of_week, is_weekend, is_rush_hour (4 features)
        - Historical patterns: exponential moving average (exp_avg)
        - Lag features: ft_1, ft_2, ft_3, ft_4, ft_5 (previous time bins)
        
        **Training Configuration:**
        - Train/Val/Test split: 70/15/15 temporal split (280K / 60K / 60K records)
        - No data leakage: Strict chronological ordering
        - Demand model hyperparameters: max_depth=3, learning_rate=0.1, n_estimators=1000, subsample=0.8
        - Revenue model hyperparameters: max_depth=4, learning_rate=0.1, n_estimators=1000, subsample=0.8
        """)
        
        # Model architecture diagram
        st.code("""
┌─────────────────────────────────────────────────────┐
│           INPUT: Zone + Time Features               │
│  (cluster_id, hour, day_of_week, rush_hour, etc.)  │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │                 │
┌───────▼────────┐ ┌──────▼───────┐ ┌──────▼──────────┐
│ Demand Model   │ │ Revenue Model│ │ Avg Fare Model  │
│(XGBoost-1000)  │ │(XGBoost-1000)│ │ (Rule-Based)    │
└───────┬────────┘ └──────┬───────┘ └──────┬──────────┘
        │                 │                 │
        ▼                 ▼                 ▼
    demand_pred      revenue_pred      avg_fare_pred
        │                 │                 │
        └────────┬────────┴─────────────────┘
                 │
        ┌────────▼────────────────────────┐
        │   COST MODEL (Rule-Based)       │
        │  - Driver: 65% of revenue       │
        │  - Fees: 2.9% of revenue        │
        │  - Ops: 5% of revenue           │
        └────────┬────────────────────────┘
                 │
        ┌────────▼────────────────────────┐
        │    OUTPUT: Business Metrics      │
        │  profit_pred, margin_pred        │
        └──────────────────────────────────┘
        """, language="text")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demand Prediction Performance:**")
            st.metric("WMAPE", "14.69%", help="Weighted Mean Absolute Percentage Error")
            st.metric("MAE", "4.72 trips", help="Mean Absolute Error")
            st.metric("R² Score", "0.966", help="Variance explained by model (96.6%)")
            
            st.markdown("**Revenue Prediction Performance:**")
            st.metric("WMAPE", "18.48%", help="Weighted Mean Absolute Percentage Error")
            st.metric("MAE", "$87.05", help="Mean Absolute Error per time bin")
            st.metric("R² Score", "0.944", help="Variance explained by model (94.4%)")
        
        with col2:
            st.markdown("**Feature Importance:**")
            st.code("""
Top 5 Features (XGBoost Demand Model):
1. exp_avg (rolling average)     38.2%
2. hour                           22.1%
3. is_rush_hour                   15.3%
4. cluster_id (zone)              12.8%
5. day_of_week                     6.4%

XGBoost Advantages:
- Gradient boosting builds 1000 trees sequentially
- Each tree corrects errors from previous trees
- L1/L2 regularization prevents overfitting (alpha=200/100)
- Handles missing data automatically
- Trained on 280K records (70% of 400K dataset)
            """, language="text")
    
    # Amazon Nova Integration
    st.markdown("---")
    st.subheader("◬ Amazon Nova Reasoning Capabilities for Agentic AI")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **How Amazon Nova Tackles Complex Real-World Problems:**
        
        **1. Autonomous Monitoring Agents**
        - Continuously monitor 260 zones (40 clusters) for complex anomaly patterns
        - Detect revenue opportunities using multi-factor analysis
        - Identify efficiency issues through statistical reasoning (Z-scores, percentiles)
        
        **2. Amazon Nova Multi-Step Reasoning Chain**
        - **STEP 1:** Root cause analysis - Nova reasons through 3+ hypotheses simultaneously
        - **STEP 2:** Opportunity assessment - Quantifies realistic uplift scenarios
        - **STEP 3:** Action recommendations - Prioritizes based on impact/feasibility
        - **STEP 4:** Implementation roadmap - Generates Week 1 → Quarter 1 execution plan
        - **Amazon Nova 2 Lite** completes all 4 reasoning steps in <2 seconds
        
        **3. Natural Language Reasoning Interface**
        - Amazon Nova understands complex business questions in plain English
        - Retrieves and synthesizes insights from 400K+ data points
        - Maintains conversation context for multi-turn reasoning
        
        **4. Scenario Reasoning & Validation**
        - Amazon Nova evaluates "what-if" scenarios across multiple dimensions
        - Reasons about feasibility, risks, and downstream impacts
        - Generates executive-level strategic recommendations
        """)
    
    with col2:
        st.markdown("**Nova Models Used:**")
        st.code("""
amazon.nova-2-lite
├─ Agent investigations
├─ Fast reasoning (<2s)
├─ Cost-efficient
└─ Multi-step analysis

amazon.nova-2-sonic
├─ Voice capabilities (future)
├─ Real-time streaming
└─ Multimodal support
        """, language="text")
        
        st.markdown("**API Configuration:**")
        st.code("""
Region: us-east-1
Max Tokens: 800
Temperature: 0.7
Streaming: Enabled
        """, language="yaml")
    
    # Technical Stack
    st.markdown("---")
    st.subheader("⎔ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Frontend & Visualization:**")
        st.code("""
- Streamlit 1.54.0
- Plotly 6.5.2
- Mapbox (NYC maps)
- Custom CSS styling
        """, language="text")
    
    with col2:
        st.markdown("**ML & Data:**")
        st.code("""
- XGBoost 2.0.3
- scikit-learn 1.5.2
- pandas 2.2.3
- numpy 2.1.3
        """, language="text")
    
    with col3:
        st.markdown("**AI & Cloud:**")
        st.code("""
- boto3 (AWS SDK)
- Amazon Bedrock
- Nova 2 Lite & Sonic
- IAM authentication
        """, language="text")
    
    # Demo Instructions
    st.markdown("---")
    st.subheader("Navigation Guide")
    
    st.markdown("""
    **Recommended Demo Flow:**
    
    1. **Project Overview** (this page) - Set technical context
    2. **Executive Dashboard** - Show ML predictions and NYC geographic map
    3. **Autonomous Agent** - Live Agentic AI demonstration ⭐⭐⭐
    4. **Nova Chat** - Amazon Nova 2 Lite natural language interface
    5. **What-If Simulator** - Multi-step reasoning for scenario analysis
    
    **Key Highlight for Judges:** The **Autonomous Agent** page demonstrates Amazon Nova tackling 
    complex real-world problems through Agentic AI. Watch the autonomous agent detect revenue anomalies, 
    perform sophisticated multi-step reasoning, and generate actionable recommendations using Amazon 
    Nova 2 Lite reasoning capabilities - all without human intervention.
    """)
    
    st.success("⭐ Ready to demo! Navigate to 'Autonomous Agent' to see Amazon Nova Agentic AI in action.")

# ----- PAGE 1: EXECUTIVE DASHBOARD -----
elif page == "Executive Dashboard":
    st.markdown('<p class="main-header">Executive Dashboard - ML Predictions</p>', unsafe_allow_html=True)
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predicted Demand",
            f"{st.session_state.predictions['demand_pred'].sum():,.0f} trips",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Predicted Revenue",
            f"${st.session_state.predictions['revenue_pred'].sum():,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Total Predicted Profit",
            f"${st.session_state.predictions['profit_pred'].sum():,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Average Margin",
            f"{st.session_state.predictions['margin_pred'].mean()*100:.1f}%",
            delta=None
        )
    
    # Charts
    st.subheader("Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top zones by revenue with actual values
        top_zones = st.session_state.predictions.groupby('cluster_id')['revenue_pred'].sum().nlargest(10).reset_index()
        top_zones.columns = ['Zone ID', 'Total Revenue']
        top_zones['Zone ID'] = top_zones['Zone ID'].astype(str)  # Make categorical
        
        fig = px.bar(
            top_zones,
            x='Zone ID',
            y='Total Revenue',
            title='Top 10 Revenue Zones (Expand for details)',
            text='Total Revenue',
            color='Total Revenue',
            color_continuous_scale='Blues'
        )
        fig.update_traces(
            texttemplate='$%{text:,.0f}', 
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )
        fig.update_layout(
            xaxis_title='Zone ID',
            yaxis_title='Total Revenue ($)',
            showlegend=False,
            height=450,
            xaxis={'type': 'category'}  # Force categorical
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Zone performance heatmap/treemap
        zone_summary = st.session_state.predictions.groupby('cluster_id').agg({
            'revenue_pred': 'sum',
            'demand_pred': 'sum',
            'margin_pred': 'mean'
        }).reset_index()
        zone_summary = zone_summary.nlargest(20, 'revenue_pred')
        zone_summary['Zone'] = 'Zone ' + zone_summary['cluster_id'].astype(str)
        zone_summary['Revenue per Trip'] = zone_summary['revenue_pred'] / zone_summary['demand_pred']
        
        fig = px.treemap(
            zone_summary,
            path=[px.Constant("All Zones"), 'Zone'],
            values='revenue_pred',
            color='Revenue per Trip',
            hover_data={
                'revenue_pred': ':$,.0f',
                'demand_pred': ':,.0f',
                'Revenue per Trip': ':$.2f'
            },
            color_continuous_scale='Viridis',
            title='Zone Revenue Map (Size=Revenue, Color=$/Trip)'
        )
        fig.update_layout(height=450)
        fig.update_traces(
            textinfo="label+value",
            texttemplate="<b>%{label}</b><br>$%{value:,.0f}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # NYC Geographic Revenue Map - Full Width
    st.subheader("NYC Zone Revenue Map")
    all_zones = st.session_state.predictions.groupby('cluster_id').agg({
        'revenue_pred': 'sum',
        'demand_pred': 'sum',
        'margin_pred': 'mean'
    }).reset_index()
    all_zones['revenue_per_trip'] = all_zones['revenue_pred'] / all_zones['demand_pred']
    
    # Filter out zones with negative or zero revenue for visualization
    all_zones = all_zones[all_zones['revenue_pred'] > 0]
    
    # Add geographic coordinates
    zone_coords = get_zone_coordinates()
    all_zones['lat'] = all_zones['cluster_id'].map(lambda x: zone_coords.get(x, {}).get('lat', 40.7))
    all_zones['lon'] = all_zones['cluster_id'].map(lambda x: zone_coords.get(x, {}).get('lon', -73.9))
    
    # Create real NYC map
    fig = px.scatter_mapbox(
        all_zones,
        lat='lat',
        lon='lon',
        size='revenue_pred',
        color='revenue_per_trip',
        hover_data={
            'cluster_id': True,
            'revenue_pred': ':$,.0f',
            'demand_pred': ':,.0f',
            'revenue_per_trip': ':$.2f',
            'lat': ':.4f',
            'lon': ':.4f'
        },
        title=f'All {len(all_zones)} zones - Size: Revenue | Color: Efficiency ($/trip)',
        color_continuous_scale='Plasma',
        size_max=30,
        zoom=9.5,
        center={"lat": 40.730, "lon": -73.935}
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        height=600,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Nova-generated executive summary
    st.subheader("Nova-Generated Summary")
    
    if st.button("Generate Summary"):
        with st.spinner("Generating AI summary..."):
            summary = st.session_state.nova.generate_executive_summary(st.session_state.predictions)
            st.markdown(f'<div class="nova-response">{summary}</div>', unsafe_allow_html=True)

# ----- PAGE 2: NOVA CHAT -----
elif page == "Nova Chat":
    st.markdown('<p class="main-header">Amazon Nova Chat - Natural Language Reasoning</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your revenue predictions in natural language")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    query = st.text_input("Your question:", placeholder="e.g., Which zones have the highest revenue potential?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Ask Nova"):
            if query:
                with st.spinner("Thinking..."):
                    answer = st.session_state.query_handler.answer_query(query)
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now(),
                        'query': query,
                        'answer': answer
                    })
    
    with col2:
        if st.button("Clear History"):
            st.session_state.chat_history = []
    
    # Display chat history
    st.subheader("Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You ({chat['timestamp'].strftime('%H:%M:%S')}):** {chat['query']}")
        st.markdown(f'<div class="nova-response"><strong>Nova:</strong> {chat["answer"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # Sample questions
    st.sidebar.subheader("Try these questions:")
    sample_questions = [
        "Which zones have the highest revenue potential?",
        "What's the average profit margin?",
        "Which zones should we focus on for improvement?",
        "Compare zones 237 and 161",
        "What zones have lowest demand?"
    ]
    
    for q in sample_questions:
        if st.sidebar.button(q, key=f"sample_{q[:20]}"):
            st.session_state.chat_history.append({
                'timestamp': datetime.now(),
                'query': q,
                'answer': st.session_state.query_handler.answer_query(q)
            })
            st.rerun()

# ----- PAGE 3: WHAT-IF SIMULATOR -----
elif page == "What-If Simulator":
    st.markdown('<p class="main-header">What-If Simulator - Amazon Nova Scenario Reasoning</p>', unsafe_allow_html=True)
    st.markdown("Simulate the impact of pricing changes on demand, revenue, and profit")
    
    # Zone selection
    zones = sorted(st.session_state.predictions['cluster_id'].unique())
    selected_zone = st.selectbox("Select Zone", zones)
    
    # Get zone data
    zone_data = st.session_state.predictions[
        st.session_state.predictions['cluster_id'] == selected_zone
    ]
    
    if not zone_data.empty:
        baseline = zone_data.iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Baseline Metrics")
            st.metric("Demand", f"{baseline['demand_pred']:.1f} trips")
            st.metric("Revenue", f"${baseline['revenue_pred']:,.2f}")
            st.metric("Profit", f"${baseline['profit_pred']:,.2f}")
            st.metric("Margin", f"{baseline['margin_pred']*100:.1f}%")
            
            st.subheader("Pricing Adjustment")
            price_change = st.slider(
                "Price Change (%)",
                min_value=-30,
                max_value=30,
                value=0,
                step=5
            )
            
            elasticity = st.number_input(
                "Price Elasticity",
                min_value=-1.0,
                max_value=0.0,
                value=-0.5,
                step=0.1
            )
        
        with col2:
            # Calculate simulation
            price_multiplier = 1 + (price_change / 100)
            demand_change_pct = elasticity * (price_change / 100)
            new_demand = baseline['demand_pred'] * (1 + demand_change_pct)
            new_revenue = new_demand * baseline['avg_fare_pred'] * price_multiplier
            
            # Recalculate costs
            cost_ratio = new_demand / baseline['demand_pred'] if baseline['demand_pred'] > 0 else 1
            new_costs = baseline['total_costs'] * cost_ratio
            new_profit = new_revenue - new_costs
            new_margin = new_profit / new_revenue if new_revenue > 0 else 0
            
            # Display results
            st.subheader("Simulation Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                demand_delta = ((new_demand - baseline['demand_pred']) / baseline['demand_pred']) * 100
                st.metric("New Demand", f"{new_demand:.1f} trips", delta=f"{demand_delta:+.1f}%")
            
            with col_b:
                revenue_delta = ((new_revenue - baseline['revenue_pred']) / baseline['revenue_pred']) * 100
                st.metric("New Revenue", f"${new_revenue:,.2f}", delta=f"{revenue_delta:+.1f}%")
            
            with col_c:
                profit_delta = ((new_profit - baseline['profit_pred']) / baseline['profit_pred']) * 100
                st.metric("New Profit", f"${new_profit:,.2f}", delta=f"{profit_delta:+.1f}%")
            
            # Visualization
            comparison_df = pd.DataFrame({
                'Metric': ['Demand', 'Revenue', 'Profit'],
                'Baseline': [
                    baseline['demand_pred'],
                    baseline['revenue_pred'],
                    baseline['profit_pred']
                ],
                'Simulated': [new_demand, new_revenue, new_profit]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline', x=comparison_df['Metric'], y=comparison_df['Baseline']))
            fig.add_trace(go.Bar(name='Simulated', x=comparison_df['Metric'], y=comparison_df['Simulated']))
            fig.update_layout(title='Baseline vs Simulated Comparison', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Nova explanation
            if price_change != 0:
                if st.button("Get Nova Analysis"):
                    with st.spinner("Analyzing..."):
                        prompt = f"""Analyze this pricing simulation for a ride-sharing zone:

Baseline: {baseline['demand_pred']:.1f} trips, ${baseline['revenue_pred']:.2f} revenue, ${baseline['profit_pred']:.2f} profit
Price Change: {price_change:+d}%
Simulated: {new_demand:.1f} trips, ${new_revenue:.2f} revenue, ${new_profit:.2f} profit

Provide a 2-3 sentence analysis: Is this a good strategy? What are the risks?"""
                        
                        analysis = st.session_state.nova.generate_explanation(prompt, max_tokens=300)
                        st.markdown(f'<div class="nova-response"><strong>Nova Analysis:</strong><br>{analysis}</div>', unsafe_allow_html=True)

# ----- PAGE 4: ZONE EXPLORER -----
elif page == "Zone Explorer":
    st.markdown('<p class="main-header">Zone Deep-Dive Explorer</p>', unsafe_allow_html=True)
    
    selected_zone = st.selectbox(
        "Select Zone to Explore",
        sorted(st.session_state.predictions['cluster_id'].unique())
    )
    
    if st.button("Get Zone Insights"):
        with st.spinner("Analyzing zone..."):
            insights = st.session_state.query_handler.get_zone_insights(selected_zone)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Demand", f"{insights['avg_demand']:.1f} trips")
                st.metric("Avg Revenue", f"${insights['avg_revenue']:.2f}")
            
            with col2:
                st.metric("Avg Profit", f"${insights['avg_profit']:.2f}")
                st.metric("Avg Margin", f"{insights['avg_margin']*100:.1f}%")
            
            with col3:
                st.metric("Peak Hour", f"{insights.get('peak_hour', 'N/A')}:00")
                st.metric("Total Predictions", insights['total_predictions'])
            
            st.subheader("AI Analysis")
            st.markdown(f'<div class="nova-response">{insights["ai_summary"]}</div>', unsafe_allow_html=True)
            
            # Zone time series - Improved visualization
            st.subheader("Revenue Trend Analysis")
            
            # Explanation box for correlation
            st.info("📊 **Why Revenue & Demand Correlate:** Revenue = Demand × Average Fare. NYC taxi fares are regulated (base + distance/time), so prices are relatively stable within zones. Divergence occurs during surge pricing or route efficiency changes.")
            
            zone_data = st.session_state.predictions[
                st.session_state.predictions['cluster_id'] == selected_zone
            ].sort_values('time_bin').reset_index(drop=True)
            
            if len(zone_data) > 0:
                # Create meaningful time labels
                zone_data['period'] = range(1, len(zone_data) + 1)
                
                # Calculate effective price per trip
                zone_data['price_per_trip'] = zone_data['revenue_pred'] / zone_data['demand_pred'].replace(0, 1)
                
                fig = go.Figure()
                
                # Revenue line
                fig.add_trace(go.Scatter(
                    x=zone_data['period'],
                    y=zone_data['revenue_pred'],
                    mode='lines+markers',
                    name='Predicted Revenue',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4),
                    hovertemplate='Period: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
                ))
                
                # Add demand as secondary line
                if 'demand_pred' in zone_data.columns:
                    fig.add_trace(go.Scatter(
                        x=zone_data['period'],
                        y=zone_data['demand_pred'] * 10,  # Scale for visibility
                        mode='lines',
                        name='Demand (×10)',
                        line=dict(color='#ff7f0e', width=1, dash='dash'),
                        yaxis='y2',
                        hovertemplate='Period: %{x}<br>Demand: %{customdata:.1f} trips<extra></extra>',
                        customdata=zone_data['demand_pred']
                    ))
                
                fig.update_layout(
                    title=f'Zone {selected_zone} - Revenue & Demand Trends ({len(zone_data)} Periods)',
                    xaxis=dict(
                        title='Time Period',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        title='Revenue ($)',
                        tickformat='$,.0f',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis2=dict(
                        title='Demand (trips)',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    hovermode='x unified',
                    height=450,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats - Now show BOTH revenue and price dynamics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Highest Revenue Period", f"#{zone_data.loc[zone_data['revenue_pred'].idxmax(), 'period']:.0f}", 
                             f"${zone_data['revenue_pred'].max():,.2f}")
                with col2:
                    st.metric("Lowest Revenue Period", f"#{zone_data.loc[zone_data['revenue_pred'].idxmin(), 'period']:.0f}",
                             f"${zone_data['revenue_pred'].min():,.2f}")
                with col3:
                    volatility = zone_data['revenue_pred'].std() / zone_data['revenue_pred'].mean() * 100
                    st.metric("Revenue Volatility", f"{volatility:.1f}%", "Coefficient of Variation")
                with col4:
                    price_volatility = zone_data['price_per_trip'].std() / zone_data['price_per_trip'].mean() * 100
                    st.metric("Price Volatility", f"{price_volatility:.1f}%", "Shows price dynamics")
                
                # Add price dynamics chart
                st.subheader("Price Dynamics Analysis")
                st.markdown("**Revenue = Demand × Price**. Below shows how average fare per trip varies over time:")
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=zone_data['period'],
                    y=zone_data['price_per_trip'],
                    mode='lines+markers',
                    name='Avg Fare per Trip',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=6, color=zone_data['price_per_trip'], colorscale='Viridis', showscale=True),
                    hovertemplate='Period: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Add average line
                avg_price = zone_data['price_per_trip'].mean()
                fig2.add_hline(
                    y=avg_price, 
                    line_dash="dash", 
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Avg: ${avg_price:.2f}",
                    annotation_position="top left",
                    annotation=dict(
                        font=dict(size=11, color="red"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1
                    )
                )
                
                fig2.update_layout(
                    title=f'Average Fare per Trip Over Time (Zone {selected_zone})',
                    xaxis_title='Time Period',
                    yaxis_title='Price per Trip ($)',
                    yaxis=dict(tickformat='$.2f'),
                    height=350,
                    hovermode='x',
                    margin=dict(t=40, b=40, l=50, r=50)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Price range explanation
                price_range = zone_data['price_per_trip'].max() - zone_data['price_per_trip'].min()
                st.markdown(f"""
                **Price Range:** ${zone_data['price_per_trip'].min():.2f} - ${zone_data['price_per_trip'].max():.2f} (${price_range:.2f} variation)
                
                **Why correlation exists:** NYC taxi pricing is regulated with fixed base fare + metered distance/time. 
                Within a zone, trip distances are similar, resulting in relatively stable average fares. 
                
                **When divergence occurs:**
                - Rush hour surges (higher $/trip, same demand)
                - Airport trips (long distance premium)
                - Tip variations in predicted revenue
                - Route efficiency changes (shorter trips = lower $/trip)
                """)

# ----- PAGE 5: RECOMMENDATIONS -----
elif page == "Recommendations":
    st.markdown('<p class="main-header">Amazon Nova-Powered Recommendations</p>', unsafe_allow_html=True)
    
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations with Nova AI..."):
            recommendations = st.session_state.rec_engine.generate_recommendations(top_n=5)
            validated_recs = st.session_state.rec_engine.validate_with_nova(recommendations)
            
            st.subheader(f"Top {len(validated_recs)} Recommendations")
            
            for i, rec in enumerate(validated_recs, 1):
                with st.expander(f"{i}. {rec['type'].upper().replace('_', ' ')} - Zone {rec.get('zone_id', 'Multiple')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Action:** {rec['action']}")
                        st.markdown(f"**Rationale:** {rec['rationale']}")
                        st.markdown(f"**Confidence:** {rec['confidence']}")
                    
                    with col2:
                        st.metric("Expected Impact", f"${rec.get('expected_impact_$', 0):,.2f}")
                    
                    if 'ai_validation' in rec:
                        st.markdown("**Nova Validation:**")
                        st.markdown(f'<div class="nova-response">{rec["ai_validation"]}</div>', unsafe_allow_html=True)
            
            # Generate report
            st.subheader("Executive Report")
            report = st.session_state.rec_engine.generate_recommendation_report(validated_recs)
            st.markdown(f'<div class="nova-response">{report}</div>', unsafe_allow_html=True)

# ----- PAGE 6: AUTONOMOUS AGENT (Agentic AI) -----
elif page == "Autonomous Agent":
    st.markdown('<p class="main-header">🤖 Autonomous Agent - Amazon Nova Agentic AI</p>', unsafe_allow_html=True)
    st.markdown("**Autonomous revenue optimization using Amazon Nova 2 Lite reasoning capabilities to tackle complex real-world problems**")
    
    # Agent status
    col1, col2, col3, col4 = st.columns(4)
    
    agent_status = st.session_state.monitoring_agent.get_agent_status()
    
    with col1:
        st.metric("Agent Status", agent_status['status'], delta=None)
    
    with col2:
        st.metric("Zones Monitored", agent_status['total_zones_monitored'], delta=None)
    
    with col3:
        st.metric("Anomalies Detected", agent_status['anomalies_detected'], delta=None)
    
    with col4:
        st.metric("Actions Proposed", agent_status['actions_taken'], delta=None)
    
    st.markdown("---")
    
    # Control panel
    st.subheader("Agent Control Panel")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown("""
        **Amazon Nova Agentic AI Workflow:**
        1. **Detect** - Autonomous scanning of 260 zones for complex anomaly patterns
        2. **Investigate** - Amazon Nova 2 Lite multi-step reasoning analyzes root causes
        3. **Propose** - AI agent generates actionable recommendations with impact forecasts
        4. **Execute** - Autonomous action implementation (with approval gates)
        
        **Complex Problem Tackled:** Revenue optimization requires simultaneous analysis of demand 
        volatility, price elasticity, cost structures, competitive dynamics, and temporal patterns - 
        a task Amazon Nova reasoning capabilities handle in seconds vs. days for human analysts.
        """)
    
    with col_b:
        run_clicked = st.button("Run Monitoring Cycle", type="primary", use_container_width=True, key="run_agent")
        
        if st.button("🔄 Reset Agent", use_container_width=True):
            st.session_state.monitoring_agent = RevenueMonitoringAgent(
                st.session_state.predictions,
                st.session_state.nova
            )
            if 'agent_report' in st.session_state:
                del st.session_state.agent_report
            st.rerun()
    
    # Run monitoring cycle
    if run_clicked:
        with st.spinner("🔍 STEP 1/3: Detecting anomalies across all zones..."):
            import time
            time.sleep(1)
            report = st.session_state.monitoring_agent.run_monitoring_cycle()
            st.session_state.agent_report = report
        
        st.success(f"Monitoring cycle completed! Found {len(report['anomalies_detected'])} anomalies")
    
    # Display results
    if 'agent_report' in st.session_state and st.session_state.agent_report:
        report = st.session_state.agent_report
        
        st.markdown("---")
        st.subheader("Anomalies Detected")
        
        if report['anomalies_detected']:
            for i, anomaly in enumerate(report['anomalies_detected'][:5], 1):
                potential_value = anomaly.get('potential_gain', 0)
                with st.expander(f"Anomaly {i}: {anomaly['type'].replace('_', ' ').title()} - Zone {anomaly['zone_id']} (${potential_value:,.0f} opportunity)", expanded=(i==1)):
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Zone ID:** {anomaly['zone_id']}")
                        st.markdown(f"**Type:** {anomaly['type'].replace('_', ' ').title()}")
                        st.markdown(f"**Severity:** {anomaly['severity']}")
                        st.markdown(f"**Detected:** {anomaly['detected_at'].strftime('%H:%M:%S')}")
                    
                    with col2:
                        metric_label = anomaly['metric'].replace('_', ' ').title()
                        if 'revenue' in anomaly['metric']:
                            current_display = f"${anomaly['current_value']:,.0f}"
                        else:
                            current_display = f"{anomaly['current_value']:.2f}"
                        
                        st.metric(
                            metric_label, 
                            current_display,
                            delta=f"+{anomaly['deviation_pct']:.1f}% to target",
                            delta_color="off"
                        )
                    
                    with col3:
                        st.metric(
                            "Opportunity Value",
                            f"${potential_value:,.0f}",
                            delta="Revenue gap to close",
                            delta_color="off"
                        )
        
        # Show investigations
        st.markdown("---")
        st.subheader("🧠 Amazon Nova 2 Lite: Autonomous Multi-Step Reasoning")
        
        if report['investigations']:
            for i, investigation in enumerate(report['investigations'][:3], 1):
                with st.expander(f"Investigation {i}: Zone {investigation['zone_id']} - Autonomous Reasoning Chain", expanded=(i==1)):
                    st.markdown("**Amazon Nova's Autonomous 4-Step Reasoning:**")
                    st.markdown(f'<div class="nova-response">{investigation["investigation"]}</div>', unsafe_allow_html=True)
                    
                    if investigation.get('recommendations'):
                        st.markdown("**Extracted Recommendations:**")
                        for rec in investigation['recommendations'][:3]:
                            st.markdown(f"- {rec}")
        
        # Show proposed actions
        st.markdown("---")
        st.subheader("Autonomous Agent: Proposed Actions")
        st.markdown("*Agentic AI autonomously generates actionable recommendations from Amazon Nova reasoning*")
        
        if report['actions_proposed']:
            for i, action in enumerate(report['actions_proposed'], 1):
                with st.expander(f"Action {i}: {action['action_type']} - Zone {action['zone_id']}", expanded=(i==1)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Autonomous Action:** {action['description']}")
                        st.markdown(f"**Type:** {action['action_type']}")
                        st.markdown(f"**Agent Confidence:** {action['confidence']}")
                        st.markdown(f"**Status:** {action['status']}")
                    
                    with col2:
                        st.metric("Expected Impact", f"${action['expected_impact']:,.2f}")
                        
                        if st.button(f"Approve & Execute {i}", key=f"execute_{i}"):
                            st.success(f"Autonomous agent executing Action {i}!")
                            st.info("**Agentic AI in Action:** Agent would now automatically adjust pricing/resource allocation.")
        
        # Agent insights
        st.markdown("---")
        st.subheader("Agentic AI Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cycle_duration = (report['cycle_completed'] - report['cycle_started']).total_seconds()
            st.metric("Cycle Duration", f"{cycle_duration:.1f}s")
        
        with col2:
            st.metric("Investigations Run", len(report['investigations']))
        
        with col3:
            total_impact = sum(action['expected_impact'] for action in report['actions_proposed'])
            st.metric("Total Potential Impact", f"${total_impact:,.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Agentic AI Platform**")
st.sidebar.markdown("**Powered by Amazon Nova 2 Lite**")
st.sidebar.markdown("*Reasoning capabilities for complex problems*")
st.sidebar.markdown(f"v1.0 | {datetime.now().strftime('%Y-%m-%d')}")
