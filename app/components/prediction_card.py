import streamlit as st

def render_prediction_card(res):
    """Renders the prediction results, congestion gauge, and peak traffic alert banners."""
    predicted_demand = res["predicted_demand"]
    congestion_level = res["congestion_level"]
    badge_label = res["badge"]  # Includes emoji and color
    gauge_score = res["gauge_score_pct"]
    peak_alert = res["peak_traffic_alert"]
    alert_message = res["alert_message"]
    
    # Map badge colors
    badge_class = "badge-low"
    if congestion_level == "MEDIUM":
        badge_class = "badge-medium"
    elif congestion_level == "HIGH":
        badge_class = "badge-high"
        
    # Render main prediction card using custom HTML
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin-top:0; color:rgba(255,255,255,0.7); font-size:1.1rem; text-transform:uppercase; letter-spacing:1px;">🔮 Forecasted Traffic</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="prediction-value">{predicted_demand:,} <span class="prediction-unit">vehicles/hour</span></div>
            </div>
            <div>
                <span class="badge {badge_class}">{congestion_level}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render Peak Traffic Alert Banner
    alert_class = "alert-banner-no" if peak_alert == "No" else "alert-banner-yes"
    alert_title = "✅ ALL CLEAR" if peak_alert == "No" else "⚠️ CONGESTION ALERT"
    
    st.markdown(f"""
    <div class="alert-banner {alert_class}">
        <h4 style="margin:0 0 5px 0; font-size:1rem; font-weight:700;">{alert_title}</h4>
        <p style="margin:0; font-size:0.9rem; font-weight:400; opacity:0.9;">{alert_message}</p>
    </div>
    """, unsafe_allow_html=True)
