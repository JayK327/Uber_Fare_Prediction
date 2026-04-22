"""
streamlit_app/app.py
Run: streamlit run streamlit_app/app.py
"""
import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from src.feature_engineering import engineer, FEATURE_COLS
from pathlib import Path

#set base path
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
print(f"Looking for models in: {MODEL_DIR},{BASE_DIR}")


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber Fare Predictor",
    page_icon="🚕",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white; padding: 1.5rem 2rem;
        border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 14px;
        text-align: center; margin: 1rem 0;
    }
    .prediction-amount {
        font-size: 3.5rem; font-weight: 800; margin: 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; cursor: pointer;
    }
    .tag { display:inline-block; background:#eff6ff; color:#1d4ed8;
           padding:3px 10px; border-radius:20px; font-size:0.8rem;
           margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        MODEL_DIR / "lgbm_tuned.joblib",
        MODEL_DIR / "XGBoost.joblib",
        MODEL_DIR / "LightGBM.joblib",
    ]

    for p in paths:
        if p.exists():
            return joblib.load(p), p.stem

    st.error(f"No model found in {MODEL_DIR}")
    st.stop()

model, model_name = load_model()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    🚕 Uber Fare Predictor — NYC
    <div style="font-size:1rem; font-weight:400; opacity:0.85; margin-top:0.4rem">
        Spatial-Temporal ML Model &nbsp;·&nbsp; XGBoost + LightGBM
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Trip Details")
    st.markdown("**Pickup Location**")
    pu_lat = st.number_input("Pickup Latitude",  value=40.7580, format="%.4f", step=0.0001)
    pu_lon = st.number_input("Pickup Longitude", value=-73.9855, format="%.4f", step=0.0001)

    st.markdown("**Dropoff Location**")
    do_lat = st.number_input("Dropoff Latitude",  value=40.6413, format="%.4f", step=0.0001)
    do_lon = st.number_input("Dropoff Longitude", value=-73.7781, format="%.4f", step=0.0001)

    st.markdown("**Trip Info**")
    pickup_dt   = st.datetime_input("Pickup Date & Time", value=datetime(2024, 6, 15, 18, 0))
    pax         = st.slider("Passengers", 1, 6, 1)

    st.markdown("---")
    st.markdown("**Quick presets**")
    col1, col2 = st.columns(2)
    if col1.button("🛫 JFK Trip"):
        st.session_state["preset"] = "jfk"
    if col2.button("🏙️ Midtown"):
        st.session_state["preset"] = "midtown"

    predict_btn = st.button(" Predict Fare", use_container_width=True)

# ── Apply presets ─────────────────────────────────────────────────────
presets = {
    "jfk":     dict(pu_lat=40.7580, pu_lon=-73.9855,
                    do_lat=40.6413, do_lon=-73.7781,
                    pickup_dt=datetime(2024, 6, 15, 18, 0), pax=2),
    "midtown": dict(pu_lat=40.7128, pu_lon=-74.0060,
                    do_lat=40.7614, do_lon=-73.9776,
                    pickup_dt=datetime(2024, 6, 15, 9, 0), pax=1),
}
if "preset" in st.session_state:
    p = presets[st.session_state.pop("preset")]
    pu_lat, pu_lon = p["pu_lat"], p["pu_lon"]
    do_lat, do_lon = p["do_lat"], p["do_lon"]
    pickup_dt, pax = p["pickup_dt"], p["pax"]


# ── Feature helper ────────────────────────────────────────────────────
def make_features(pu_lat, pu_lon, do_lat, do_lon, dt, pax):
    row = pd.DataFrame([{
        "key": "live",
        "fare_amount": 0,
        "pickup_datetime": dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pickup_longitude": pu_lon, "pickup_latitude": pu_lat,
        "dropoff_longitude": do_lon, "dropoff_latitude": do_lat,
        "passenger_count": pax,
    }])
    feat = engineer(row)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    return feat[cols]


# ── Main panel ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Prediction", " Map", " Insights"])

with tab1:
    if predict_btn or True:  # always show on load with defaults
        X = make_features(pu_lat, pu_lon, do_lat, do_lon, pickup_dt, pax)
        fare = float(model.predict(X)[0])
        fare = max(2.50, round(fare, 2))

        # Compute distance
        from src.feature_engineering import haversine
        dist = float(haversine(
            np.array([pu_lat]), np.array([pu_lon]),
            np.array([do_lat]), np.array([do_lon])
        )[0])

        # Tags
        hour = pickup_dt.hour
        dow  = pickup_dt.weekday()
        is_rush   = (7 <= hour <= 9) or (17 <= hour <= 19)
        is_night  = hour >= 22 or hour <= 4
        is_wknd   = dow >= 5
        is_airport= dist < 30 and (
            haversine(np.array([do_lat]),np.array([do_lon]),np.array([40.6413]),np.array([-73.7781]))[0] < 3
        )
        tags = []
        if is_rush:   tags.append("🔴 Rush Hour Surge")
        if is_night:  tags.append("🌙 Late Night")
        if is_wknd:   tags.append("📅 Weekend")
        if is_airport:tags.append("✈️ Airport Trip")

        c1, c2, c3 = st.columns([1.2, 1, 1])

        with c1:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:1rem;opacity:0.9;margin-bottom:0.5rem">Predicted Fare</div>
                <div class="prediction-amount">${fare:.2f}</div>
                <div style="font-size:0.9rem;opacity:0.75;margin-top:0.5rem">
                    Model: {model_name}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if tags:
                for t in tags:
                    st.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)

        with c2:
            st.metric("Distance", f"{dist:.1f} km")
            st.metric("Pickup Hour", f"{hour:02d}:00")
            st.metric("Passengers", pax)

        with c3:
            st.metric("Est. Rate/km", f"${fare/max(dist,0.1):.2f}")
            day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            st.metric("Day", day_names[dow])
            st.metric("Month", pickup_dt.strftime("%B"))

        st.markdown("---")

        # Fare range bar
        low, high = fare * 0.85, fare * 1.18
        fig = go.Figure(go.Bar(
            x=[fare], y=["Fare"],
            orientation="h",
            marker_color="#667eea",
            error_x=dict(type="data", symmetric=False,
                         array=[high-fare], arrayminus=[fare-low], color="#764ba2"),
            text=[f"${fare:.2f}"],
            textposition="inside",
        ))
        fig.update_layout(
            title="Predicted Fare with Confidence Range",
            xaxis_title="Fare ($)", height=150,
            margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hour-of-day fare curve
        hours = list(range(24))
        fares_by_hour = []
        for h in hours:
            dt_h = pickup_dt.replace(hour=h)
            X_h  = make_features(pu_lat, pu_lon, do_lat, do_lon, dt_h, pax)
            fares_by_hour.append(max(2.50, float(model.predict(X_h)[0])))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hours, y=fares_by_hour, mode="lines+markers",
            line=dict(color="#667eea", width=2.5),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(102,126,234,0.1)",
            name="Fare by hour"
        ))
        fig2.add_vline(x=hour, line_dash="dash", line_color="#ef4444",
                       annotation_text="Your pickup")
        fig2.add_vrect(x0=7, x1=9,   fillcolor="orange", opacity=0.1, annotation_text="AM rush")
        fig2.add_vrect(x0=17, x1=19, fillcolor="orange", opacity=0.1, annotation_text="PM rush")
        fig2.update_layout(
            title="Predicted Fare by Hour of Day — Same Route",
            xaxis_title="Hour", yaxis_title="Fare ($)",
            height=280, margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickmode="array",
                       tickvals=list(range(0,24,3)),
                       ticktext=[f"{h:02d}:00" for h in range(0,24,3)]),
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("### Route Map")
    map_df = pd.DataFrame([
        {"lat": pu_lat, "lon": pu_lon, "label": "Pickup 🟢", "color": "#22c55e", "size": 200},
        {"lat": do_lat, "lon": do_lon, "label": "Dropoff 🔴", "color": "#ef4444", "size": 200},
    ])
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=[pu_lat, do_lat], lon=[pu_lon, do_lon],
        mode="lines+markers",
        line=dict(width=3, color="#667eea"),
        marker=dict(size=[14, 14], color=["#22c55e","#ef4444"]),
        text=["Pickup","Dropoff"],
        hoverinfo="text",
    ))
    mid_lat = (pu_lat + do_lat) / 2
    mid_lon = (pu_lon + do_lon) / 2
    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=mid_lat, lon=mid_lon), zoom=11),
        margin=dict(l=0,r=0,t=0,b=0), height=500,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Pickup",  f"{pu_lat:.4f}°N, {abs(pu_lon):.4f}°W")
    col2.metric("Dropoff", f"{do_lat:.4f}°N, {abs(do_lon):.4f}°W")

with tab3:
    st.markdown("### Model Insights")
    model_data = {
        "Model": ["XGBoost ","LightGBM","Random Forest","Ridge","Linear Reg."],
        "Test RMSE": [1.68, 1.70, 2.24, 3.00, 3.00],
        "R²":       [0.9936, 0.9934, 0.9886, 0.9796, 0.9796],
        "MAPE (%)": [4.81, 4.85, 5.64, 7.06, 7.06],
    }
    df_models = pd.DataFrame(model_data)
    fig_cmp = px.bar(df_models, x="Model", y="Test RMSE",
                     color="Test RMSE", color_continuous_scale="Blues_r",
                     title="Model Comparison — Test RMSE (lower is better)")
    fig_cmp.update_layout(height=320, margin=dict(t=40,b=0,l=0,r=0),
                           plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.dataframe(df_models.set_index("Model"), use_container_width=True)

    st.markdown("### Feature Categories")
    feat_cats = {
        "Spatial core (5)": "Haversine + Manhattan distance, bearing, lat/lon diff",
        "Airport proximity (7)": "Distance to JFK, LGA, EWR + is_airport flag",
        "Temporal flags (4)": "Rush hour, night, weekend, holiday season",
        "Cyclic encodings (6)": "Sin/cos for hour, day-of-week, month",
        "Interaction terms (4)": "Distance × surge context multipliers",
        "Passenger (2)": "Raw count + binned category",
        "Raw temporal (4)": "Hour, DOW, month, year",
    }
    for cat, desc in feat_cats.items():
        st.markdown(f"**{cat}** — {desc}")
