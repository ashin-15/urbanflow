import streamlit as st
import datetime

def render_sidebar():
    """Renders the sidebar controls and returns a dictionary of inputs."""
    st.sidebar.header("🕹️ Control Panel")
    
    # 1. Location selection
    location_map = {
        "Airport": "dr5reg",
        "Downtown": "ezs42",
        "Industrial": "ezs4b1",
        "Residential": "dr5rg7",
        "Suburbs": "drt4ej"
    }
    selected_loc_name = st.sidebar.selectbox(
        "Location",
        options=list(location_map.keys()),
        index=1  # Default to Downtown
    )
    selected_geohash = location_map[selected_loc_name]
    
    # 2. Road details
    st.sidebar.subheader("🛣️ Infrastructure")
    road_type = st.sidebar.selectbox(
        "Road Type",
        options=["Highway", "Arterial", "Collector", "Local Street", "Residential Street"],
        index=0
    )
    
    number_of_lanes = st.sidebar.slider("Number of Lanes", min_value=1, max_value=6, value=3)
    traffic_signals = st.sidebar.checkbox("Has Traffic Signals", value=True)
    
    # 3. Environmental conditions
    st.sidebar.subheader("⛅ Environmental conditions")
    weather_condition = st.sidebar.selectbox(
        "Weather Condition",
        options=["Clear", "Cloudy", "Foggy", "Rainy", "Snowy", "Thunderstorm"],
        index=0
    )
    
    temperature = st.sidebar.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=25.0, step=0.5)
    humidity = st.sidebar.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    
    # Sensible defaults based on weather
    default_rainfall = 0.0
    if weather_condition in ["Rainy", "Thunderstorm"]:
        default_rainfall = 15.0
    elif weather_condition == "Snowy":
        default_rainfall = 2.0
        
    rainfall = st.sidebar.slider("Rainfall (mm)", min_value=0.0, max_value=100.0, value=default_rainfall, step=0.1)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
    
    # 4. Traffic attributes
    st.sidebar.subheader("🚗 Traffic Profile")
    large_vehicles_count = st.sidebar.slider("Large Vehicles Count", min_value=0, max_value=150, value=20)
    event_indicator = st.sidebar.checkbox("Public Event Nearby", value=False)
    
    # Landmark multi-select
    all_landmarks = ["School", "Park", "Hospital", "University", "Warehouse", "Port", "Shopping Mall", "Central Station", "City Hall", "Terminal 1", "Cargo Area", "Factory Complex", "Parking Lot"]
    selected_landmarks = st.sidebar.multiselect(
        "Nearby Landmarks",
        options=all_landmarks,
        default=["Shopping Mall", "Hospital"]
    )
    # Join into comma-separated string
    nearby_landmarks = ", ".join(selected_landmarks) if selected_landmarks else "None"
    
    # 5. Datetime picker
    st.sidebar.subheader("🕒 Target Date & Time")
    date_val = st.sidebar.date_input("Date", datetime.date(2026, 6, 25))
    hour_val = st.sidebar.selectbox("Hour (0-23)", options=list(range(24)), index=17) # Default to 17 (5 PM)
    
    # Construct timestamp string
    timestamp_str = f"{date_val} {hour_val:02d}:00:00"
    
    # Construct input dictionary
    inputs = {
        "timestamp": timestamp_str,
        "location": selected_loc_name,
        "geohash_location": selected_geohash,
        "road_type": road_type,
        "number_of_lanes": number_of_lanes,
        "traffic_signals": 1 if traffic_signals else 0,
        "large_vehicles_count": large_vehicles_count,
        "temperature": float(temperature),
        "humidity": float(humidity),
        "rainfall": float(rainfall),
        "wind_speed": float(wind_speed),
        "weather_condition": weather_condition,
        "event_indicator": 1 if event_indicator else 0,
        "nearby_landmarks": nearby_landmarks
    }
    
    return inputs
