import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page configuration layout
st.set_page_config(page_title="Telco Churn Business ROI Engine", layout="wide")

@st.cache_resource
def load_production_artifacts():
    """Loads the pre-trained, serialized pipeline objects instantly to avoid database strain."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
    model_path = os.path.join(base_dir, 'models', 'xgboost_model.joblib')
    
    # Load objects from local disk arrays
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model

# Execute structural artifacts load
try:
    preprocessor, model = load_production_artifacts()
    st.sidebar.success("⚡ Production Machine Learning Engine Active (Serialized Mode)")
except Exception as e:
    st.error(f"Failed to load production engine artifacts: {e}")
    st.stop()

# Header block architecture
st.title("📊 Production-Grade Telco Churn & Financial ROI Engine")
st.markdown("""
### Enterprise Machine Learning Pipeline for Data-Driven Retention
*This system utilizes optimized static binary assets (`.joblib`) deployed on cloud container layers to process inputs and evaluate corporate customer retention margins in real-time.*
""")

st.write("---")

# Layout segmentation blocks
col1, col2 = st.columns()

with col1:
    st.subheader("💡 Interactive ROI Business Calculator")
    st.markdown("Estimate the actual **financial savings** of deploying this model in production.")
    
    # Financial KPI inputs 
    avg_customer_value = st.number_input("Average Monthly Customer Revenue (₹)", min_value=100, max_value=5000, value=650, step=50)
    retention_offer_cost = st.number_input("Cost of Retention Incentive Offer (₹)", min_value=10, max_value=2000, value=150, step=10)
    offer_acceptance_rate = st.slider("Incentive Acceptance Success Rate (%)", min_value=10, max_value=100, value=60) / 100.0

with col2:
    st.subheader("🎯 Pipeline Performance Overview")
    
    # Structural KPI components
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("ROC-AUC Score", "84.3%", "Production Grade")
    kpi2.metric("True Positive Capture (Recall)", "81.0%", "Imbalance Resilient")
    kpi3.metric("Dataset Base Size", "7,043 Rows", "Supabase Cloud")

    # Financial Matrix Computations
    total_eval_pool = 1409
    historical_churners = 374
    
    # Calculations based on model performance metrics
    predicted_churners = int(historical_churners * 0.81)
    prevented_churn_saves = int(predicted_churners * offer_acceptance_rate)
    
    # Financial balance equations
    gross_revenue_saved = prevented_churn_saves * avg_customer_value
    total_campaign_cost = predicted_churners * retention_offer_cost
    net_pipeline_profit = gross_revenue_saved - total_campaign_cost

    st.write("---")
    st.subheader("📈 Projected Campaign Business Impact Report")
    
    rep1, rep2, rep3 = st.columns(3)
    rep1.metric("Customers Saved", f"{prevented_churn_saves} Accounts", f"Out of {predicted_churners} caught")
    rep2.metric("Gross Revenue Protected", f"₹{gross_revenue_saved:,}", "Recovered Value")
    
    if net_pipeline_profit > 0:
        rep3.metric("Net Financial ROI Profit", f"₹{net_pipeline_profit:,}", "💰 Positive Impact", delta_color="normal")
    else:
        rep3.metric("Net Financial ROI Profit", f"₹{net_pipeline_profit:,}", "⚠️ Cost Exceeds Return", delta_color="inverse")

    st.info(f"**Business Logic Rule applied:** Out of {historical_churners} actual churners, our XGBoost engine flags {predicted_churners}. Sending an incentive offer costing ₹{retention_offer_cost} to each flagged account with a {offer_acceptance_rate*100:.0f}% save rate prevents {prevented_churn_saves} cancellations.")
