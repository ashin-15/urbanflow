import streamlit as st
import textwrap

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
    st.markdown(textwrap.dedent(f"""
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
    """), unsafe_allow_html=True)
    
    # Render Peak Traffic Alert Banner
    alert_class = "alert-banner-no" if peak_alert == "No" else "alert-banner-yes"
    alert_title = "✅ ALL CLEAR" if peak_alert == "No" else "⚠️ CONGESTION ALERT"
    
    st.markdown(textwrap.dedent(f"""
    <div class="alert-banner {alert_class}">
        <h4 style="margin:0 0 5px 0; font-size:1rem; font-weight:700;">{alert_title}</h4>
        <p style="margin:0; font-size:0.9rem; font-weight:400; opacity:0.9;">{alert_message}</p>
    </div>
    """), unsafe_allow_html=True)
    
    # Render Accident Risk Assessment
    risk_score = res.get("accident_risk_score", 0)
    risk_level = res.get("accident_risk_level", "Low")
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(textwrap.dedent(f"""
    <div class="glass-card {risk_class}" style="padding: 16px; margin-top: 15px;">
        <h4 style="margin:0 0 5px 0; font-size:1rem; font-weight:700;">🚨 Accident Risk: {risk_level}</h4>
        <p style="margin:0; font-size:0.9rem;">Estimated risk probability: <b>{risk_score}%</b> based on current weather and traffic conditions.</p>
    </div>
    """), unsafe_allow_html=True)
    
    # Render Alternative Routes Recommendation
    alt_routes = res.get("alternative_routes", [])
    if alt_routes:
        routes_html = ""
        for route in alt_routes:
            # Highlight if demand is lower than predicted demand
            demand = route['predicted_demand']
            diff = predicted_demand - demand
            saving = f"<span style='color: #2ecc71; font-size: 0.8rem;'>(-{diff} veh/hr)</span>" if diff > 0 else ""
            
            routes_html += f"""<div class="route-item">
<span class="route-type">📍 {route['road_type']}</span>
<span><span class="route-demand">{demand}</span> {saving}</span>
</div>"""
            
        st.markdown(f"""<div class="glass-card" style="padding: 16px; margin-top: 15px;">
<h4 style="margin:0 0 10px 0; font-size:1rem; font-weight:700;">🗺️ Recommended Alternatives</h4>
{routes_html}</div>""", unsafe_allow_html=True)
