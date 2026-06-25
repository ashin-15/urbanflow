import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def render_gauge_chart(score_pct, predicted_demand):
    """Renders a beautiful radial gauge indicating the congestion level."""
    # Define colors based on score
    if score_pct < 33.3:
        bar_color = "#2ecc71"
        level_text = "LOW"
    elif score_pct < 66.6:
        bar_color = "#f1c40f"
        level_text = "MEDIUM"
    else:
        bar_color = "#e74c3c"
        level_text = "HIGH"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_demand,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Congestion Status: {level_text}", 'font': {'size': 18, 'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [0, 8000], 'tickwidth': 1, 'tickcolor': "#888888"},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 1500], 'color': 'rgba(46, 204, 113, 0.05)'},
                {'range': [1500, 3000], 'color': 'rgba(241, 196, 15, 0.05)'},
                {'range': [3000, 8000], 'color': 'rgba(231, 76, 60, 0.05)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.75,
                'value': predicted_demand
            }
        },
        number = {'font': {'color': '#ffffff', 'size': 32}, 'suffix': " veh/hr"}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_forecast_chart(inputs, predictor, selected_hour):
    """Generates and renders a 24-hour line chart forecast for the selected date/location."""
    # Build 24 hours of inputs
    timestamp = pd.to_datetime(inputs["timestamp"])
    base_date_str = timestamp.strftime("%Y-%m-%d")
    
    hours = list(range(24))
    predictions = []
    
    features_list = []
    for hr in hours:
        hr_input = inputs.copy()
        hr_input["timestamp"] = f"{base_date_str} {hr:02d}:00:00"
        
        # Predict for this hour
        df_single = pd.DataFrame([hr_input])
        df_trans = predictor.fe.transform(df_single)
        features = predictor.fe.transform(df_single)[predictor.ensemble.lgbm_model.feature_name_] # Use exact matching features
        features_list.append(df_trans)
        
    df_all_hours = pd.concat(features_list, axis=0).reset_index(drop=True)
    all_model_features = df_all_hours[predictor.ensemble.lgbm_model.feature_name_]
    
    # Batch predict
    y_preds = predictor.ensemble.predict(all_model_features)
    y_preds = np.clip(y_preds, 0, None)
    
    forecast_df = pd.DataFrame({
        "Hour": hours,
        "Forecasted Demand": y_preds
    })
    
    # Plotly Line Chart
    fig = px.line(
        forecast_df, 
        x="Hour", 
        y="Forecasted Demand",
        title=f"24-Hour Traffic Demand Forecast ({base_date_str})",
        labels={"Forecasted Demand": "Vehicles/Hour"},
        template="plotly_dark"
    )
    
    fig.update_traces(line=dict(color='#3498db', width=3))
    
    # Add a marker/dot for the selected hour
    selected_val = forecast_df.loc[forecast_df["Hour"] == selected_hour, "Forecasted Demand"].values[0]
    fig.add_trace(go.Scatter(
        x=[selected_hour], 
        y=[selected_val],
        mode="markers+text",
        marker=dict(color='#e74c3c', size=12, line=dict(color='white', width=2)),
        name="Selected Hour",
        text=[f"Active Hour: {selected_hour}:00<br>{int(round(selected_val))} veh/hr"],
        textposition="top center",
        showlegend=False
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.01)',
        font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
        xaxis=dict(tickmode='array', tickvals=list(range(24)), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=20, r=20, t=60, b=20),
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance_chart(predictor):
    """Renders the top 10 feature importances from the LightGBM model."""
    lgbm_model = predictor.ensemble.lgbm_model
    importances = lgbm_model.feature_importances_
    feature_names = lgbm_model.feature_name_
    
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)
    
    # Take top 10
    top_10 = feat_imp.tail(10)
    
    # Replace suffix encoded names with cleaner labels for UI
    clean_map = {
        "geohash_location_encoded": "Location Demographics",
        "road_type_encoded": "Road Design Capacity",
        "location_encoded": "Area Classification",
        "traffic_density_score": "Current Lane Utilization",
        "weather_impact_score": "Weather Severity Index",
        "large_vehicles_count": "Heavy Vehicle Volume",
        "number_of_lanes": "Number of Lanes",
        "traffic_signals": "Traffic Control Signals",
        "nearby_landmarks_count": "Landmark Density",
        "event_indicator": "Public Event Flag",
        "lane_signal_interaction": "Signal-Lane Interaction",
        "landmark_event_interaction": "Event-Landmark Interaction"
    }
    top_10["Feature Label"] = top_10["Feature"].map(lambda x: clean_map.get(x, x.replace("_", " ").title()))
    
    fig = px.bar(
        top_10,
        y="Feature Label",
        x="Importance",
        title="Top 10 Drivers of Traffic Demand (LightGBM Drivers)",
        orientation="h",
        template="plotly_dark",
        color="Importance",
        color_continuous_scale=px.colors.sequential.Bluyl
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
        margin=dict(l=20, r=20, t=60, b=20),
        height=350,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', title=None)
    )
    
    st.plotly_chart(fig, use_container_width=True)
