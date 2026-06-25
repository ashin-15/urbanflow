import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import TrafficPredictor
from app.components.sidebar import render_sidebar
from app.components.prediction_card import render_prediction_card
from app.components.charts import render_gauge_chart, render_forecast_chart, render_feature_importance_chart

# 1. Page Configuration
st.set_page_config(
    page_title="UrbanFlow — Smart City Traffic Demand Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Custom CSS Stylesheet
def local_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at {css_path}")

local_css("assets/style.css")

# 3. Cache Model Predictor Loading
@st.cache_resource(show_spinner="Initializing machine learning models...")
def get_predictor():
    try:
        # Load from models dir in project root
        return TrafficPredictor(models_dir="models")
    except Exception as e:
        st.error(f"Failed to load prediction models. Ensure models are trained first: {e}")
        return None

predictor = get_predictor()

# 4. Main App Layout
st.title("🚦 UrbanFlow — Smart City Traffic Demand Predictor")
st.markdown("An advanced AI prediction engine that forecasts vehicular traffic demand at road segments, times, and environmental conditions using a LightGBM and XGBoost weighted ensemble.")

if predictor is None:
    st.info("💡 ML models are not available. Please run `python src/train.py` from your terminal to train and serialize the models, then refresh this page.")
else:
    # Render Sidebar and retrieve user inputs
    inputs = render_sidebar()
    
    # 5. Core Prediction Execution
    try:
        with st.spinner("Analyzing traffic dynamics..."):
            res = predictor.predict_single(inputs)
            
        # Structure the Dashboard Panel
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown('<div class="glass-card" style="padding: 10px 24px;">', unsafe_allow_html=True)
            render_prediction_card(res)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Forecast chart
            selected_hour = inputs["timestamp"].split(" ")[1].split(":")[0]
            render_forecast_chart(inputs, predictor, int(selected_hour))
            
        with col2:
            # Radial Gauge for visual level status
            render_gauge_chart(res["gauge_score_pct"], res["predicted_demand"])
            
            # Feature Drivers Analysis
            render_feature_importance_chart(predictor)
            
    except Exception as e:
        st.error(f"Inference error occurred: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem;'>"
    "Innovexa Catalyst — ML Engineering Team &copy; 2026. Approved for production deployment."
    "</p>",
    unsafe_allow_html=True
)
