"""
Flood Prediction System — Streamlit Application
================================================
Run with:
    streamlit run app/app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).parent
BASE_DIR  = APP_DIR.parent
MODEL_PATH = BASE_DIR / "saved_model" / "model.pkl"
DATA_PATH  = BASE_DIR / "dataset"    / "flood_data.csv"
IMAGES_DIR = BASE_DIR / "images"

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0d1b2a 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.header-banner h1 { color: #ffffff; font-size: 2.2rem; margin: 0; }
.header-banner p  { color: #90caf9; margin: 0.4rem 0 0; font-size: 1.05rem; }

/* ── Risk cards ── */
.risk-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}
.risk-low    { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-left: 6px solid #43a047; }
.risk-medium { background: linear-gradient(135deg, #fff8e1, #ffe082); border-left: 6px solid #fb8c00; }
.risk-high   { background: linear-gradient(135deg, #ffebee, #ffcdd2); border-left: 6px solid #e53935; }
.risk-card h2 { font-size: 2.2rem; margin: 0.2rem 0; }
.risk-card h3 { margin: 0; font-size: 1.1rem; color: #444; font-weight: 600; }
.risk-card p  { margin: 0.4rem 0 0; color: #555; font-size: 0.95rem; }

/* ── Metric tiles ── */
.metric-tile {
    background: #f8fafd;
    border: 1px solid #e0e7ef;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .label { font-size: 0.82rem; color: #777; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-tile .value { font-size: 1.7rem; font-weight: 700; color: #1a3a5c; margin-top: 0.2rem; }

/* ── Warning box ── */
.warning-box {
    background: #fff3e0;
    border: 1px solid #ffb300;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.warning-box h4 { margin: 0 0 0.6rem; color: #e65100; }
.warning-box ul { margin: 0; padding-left: 1.2rem; color: #5d4037; line-height: 1.8; }

/* ── Safe box ── */
.safe-box {
    background: #e8f5e9;
    border: 1px solid #66bb6a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.safe-box h4 { margin: 0 0 0.6rem; color: #2e7d32; }
.safe-box ul { margin: 0; padding-left: 1.2rem; color: #1b5e20; line-height: 1.8; }

/* ── Section headers ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a3a5c;
    border-bottom: 2px solid #1A73E8;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d1b2a; }
[data-testid="stSidebar"] * { color: #cde4ff !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def domain_risk_score(rainfall, humidity, river_discharge, water_level, elevation, historical):
    """
    Physics-informed risk scoring.
    Each factor is normalised to [0,1] using real-world thresholds, then
    combined with domain-weighted sum.
    """
    # Normalise each factor (clamp to [0,1])
    r_rain   = min(rainfall       / 250.0, 1.0)   # 250 mm = extreme
    r_humid  = min((humidity - 40) / 60.0, 1.0) if humidity > 40 else 0
    r_river  = min(river_discharge / 4000.0, 1.0) # 4000 m³/s = very high
    r_level  = min(water_level     / 8.0,   1.0)  # 8 m = dangerously high
    r_elev   = max(0, 1 - elevation / 500.0)      # <500 m elevation = risky
    r_hist   = float(historical)                   # 0 or 1

    # Weighted combination
    score = (
        0.30 * r_rain  +
        0.20 * r_river +
        0.20 * r_level +
        0.15 * r_elev  +
        0.10 * r_humid +
        0.05 * r_hist
    )
    return float(np.clip(score, 0, 1))


def classify_severity(probability):
    if probability < 0.30:
        return "LOW",    "🟢", "risk-low",    "#43a047"
    elif probability < 0.70:
        return "MEDIUM", "🟡", "risk-medium", "#fb8c00"
    else:
        return "HIGH",   "🔴", "risk-high",   "#e53935"


def recommendations(severity):
    if severity == "HIGH":
        return {
            "title": "⚠️ High Flood Risk — Immediate Actions Required",
            "items": [
                "Evacuate low-lying and flood-prone areas immediately",
                "Move valuables, electronics, and documents to higher floors",
                "Avoid crossing flooded roads — even 15 cm can sweep a person",
                "Prepare emergency kit: water, food, medicines, torch, documents",
                "Monitor NDMA / State Disaster Management Authority advisories",
                "Alert elderly, children and disabled neighbours",
                "Turn off electricity at the main switch before flooding reaches home",
                "Keep emergency contacts ready: NDMA helpline 1078",
            ],
            "css": "warning-box",
        }
    elif severity == "MEDIUM":
        return {
            "title": "⚡ Moderate Risk — Stay Vigilant",
            "items": [
                "Monitor local weather and river level updates closely",
                "Prepare an emergency bag in case rapid evacuation is needed",
                "Check drainage around your property and clear blockages",
                "Keep important documents in waterproof bags",
                "Charge all devices and keep power banks ready",
                "Stay in touch with neighbours and community groups",
            ],
            "css": "warning-box",
        }
    else:
        return {
            "title": "✅ Low Risk — Normal Precautions",
            "items": [
                "Conditions appear safe; continue normal activities",
                "Stay informed via weather apps — conditions can change",
                "Keep an emergency contact list handy as good practice",
                "Inspect and clear drains to prevent local waterlogging",
            ],
            "css": "safe-box",
        }


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 Flood Prediction\nSystem")
    st.markdown("---")

    st.markdown("### 📍 Location")
    region = st.text_input("Region / City Name", value="", placeholder="e.g. Patna, Bihar")

    st.markdown("### 🌧️ Atmospheric Conditions")
    rainfall   = st.slider("Rainfall (mm)",   0.0, 300.0, 120.0, step=1.0,
                            help="Total recent rainfall in millimetres")
    humidity   = st.slider("Humidity (%)",    0.0, 100.0, 65.0,  step=1.0)

    st.markdown("### 🌊 Hydrological Conditions")
    river_discharge = st.slider("River Discharge (m³/s)", 0.0, 5000.0, 1500.0, step=10.0)
    water_level     = st.slider("Water Level (m)",        0.0, 10.0,   4.5,    step=0.1)

    st.markdown("### 🏔️ Terrain & History")
    elevation  = st.slider("Elevation (m)",    0.0, 9000.0, 300.0, step=10.0)
    historical = st.selectbox("Historical Flood Events", [0, 1],
                               format_func=lambda x: "Yes — floods recorded" if x else "No historical floods")

    predict_btn = st.button("🔍 Predict Flood Risk", use_container_width=True, type="primary")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Prediction", "📊 Data Insights", "ℹ️ About"], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════
# LOAD ASSETS
# ══════════════════════════════════════════════════════════════════════════
try:
    payload = load_model()
    model   = payload["model"]
    features = payload["features"]
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"Model file not found. Run `python model/train_model.py` first. ({e})")

try:
    df = load_data()
    data_loaded = True
except Exception:
    data_loaded = False


# ══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════
if "🏠 Prediction" in page:

    # Header
    location_str = f" — {region}" if region.strip() else ""
    st.markdown(f"""
    <div class="header-banner">
        <h1>🌊 AI Flood Prediction System{location_str}</h1>
        <p>Real-time flood risk assessment powered by Machine Learning &amp; Domain-Knowledge Scoring</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input summary tiles ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Current Input Conditions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    tiles = [
        (c1, "🌧️ Rainfall",    f"{rainfall:.0f} mm"),
        (c2, "💧 Humidity",    f"{humidity:.0f}%"),
        (c3, "🌊 River Flow",  f"{river_discharge:.0f} m³/s"),
        (c4, "📏 Water Level", f"{water_level:.1f} m"),
        (c5, "⛰️ Elevation",   f"{elevation:.0f} m"),
        (c6, "📜 History",     "Yes" if historical else "No"),
    ]
    for col, label, value in tiles:
        col.markdown(f"""
        <div class="metric-tile">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction ──────────────────────────────────────────────────────
    if predict_btn or True:   # show result on load with defaults too
        # Domain-knowledge probability
        domain_prob = domain_risk_score(
            rainfall, humidity, river_discharge, water_level, elevation, historical
        )

        # ML model probability (if available) — blended 30/70
        if model_loaded:
            flood_pressure   = (rainfall / (elevation + 1) * 100 +
                                 river_discharge / 5000 * 100)
            risk_composite   = (0.30 * (rainfall / 300) +
                                 0.25 * (river_discharge / 5000) +
                                 0.20 * (water_level / 10) +
                                 0.15 * (1 - min(elevation / 9000, 1)) +
                                 0.10 * (humidity / 100))
            input_row = pd.DataFrame([[
                rainfall, humidity, river_discharge, water_level,
                elevation, historical, flood_pressure, risk_composite
            ]], columns=features)
            ml_prob = float(model.predict_proba(input_row)[0][1])
            # Blend: 70% domain physics + 30% ML (ML is from synthetic data)
            final_prob = 0.70 * domain_prob + 0.30 * ml_prob
        else:
            final_prob = domain_prob

        severity, emoji, css_class, color = classify_severity(final_prob)
        recs = recommendations(severity)

        # ── Result card ─────────────────────────────────────────────────
        col_r, col_b = st.columns([1, 1])

        with col_r:
            st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="risk-card {css_class}">
                <h3>Flood Risk Level</h3>
                <h2>{emoji} {severity}</h2>
                <p>Probability: <strong>{final_prob*100:.1f}%</strong></p>
                <p style="font-size:0.85rem; margin-top:0.6rem; color:#666;">
                    {'📍 ' + region if region.strip() else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge (matplotlib)
            fig, ax = plt.subplots(figsize=(5, 0.7))
            fig.patch.set_facecolor("#f8fafd")
            ax.set_facecolor("#f8fafd")
            # Background bar
            ax.barh(0, 1,   color="#e0e7ef", height=0.5, edgecolor="none")
            # Zones
            ax.barh(0, 0.30, color="#c8e6c9", height=0.5, left=0,    edgecolor="none", alpha=0.7)
            ax.barh(0, 0.40, color="#ffe082", height=0.5, left=0.30,  edgecolor="none", alpha=0.7)
            ax.barh(0, 0.30, color="#ffcdd2", height=0.5, left=0.70,  edgecolor="none", alpha=0.7)
            # Indicator
            ax.barh(0, final_prob, color=color, height=0.3, edgecolor="white", linewidth=1.5)
            ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([0, 0.3, 0.7, 1.0])
            ax.set_xticklabels(["0%", "30%", "70%", "100%"], fontsize=9)
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.set_title("Risk Probability Gauge", fontsize=10, color="#555", pad=6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="section-title">📊 Risk Factor Breakdown</div>', unsafe_allow_html=True)

            factors = {
                "Rainfall":       min(rainfall / 250, 1.0),
                "River Discharge": min(river_discharge / 4000, 1.0),
                "Water Level":    min(water_level / 8, 1.0),
                "Low Elevation":  max(0, 1 - elevation / 500),
                "Humidity":       min(max(0, (humidity - 40) / 60), 1.0),
                "History":        float(historical),
            }
            colors_factor = []
            for v in factors.values():
                if v < 0.30: colors_factor.append("#43a047")
                elif v < 0.70: colors_factor.append("#fb8c00")
                else: colors_factor.append("#e53935")

            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor("#f8fafd"); ax2.set_facecolor("#f8fafd")
            bars2 = ax2.barh(list(factors.keys()), list(factors.values()),
                              color=colors_factor, edgecolor="white", height=0.6)
            for bar, val in zip(bars2, factors.values()):
                ax2.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                         f"{val*100:.0f}%", va="center", fontsize=9, color="#333")
            ax2.set_xlim(0, 1.1); ax2.set_xlabel("Normalised Risk")
            ax2.set_title("Individual Risk Factors", fontsize=11, fontweight="bold",
                           color="#1a3a5c", pad=8)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # ── Severity table ───────────────────────────────────────────────
        st.markdown('<div class="section-title">📏 Severity Classification</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        for col_s, label, rng, cls, active in [
            (sc1, "🟢 LOW",    "0% – 30%",  "risk-low",    severity == "LOW"),
            (sc2, "🟡 MEDIUM", "31% – 70%", "risk-medium", severity == "MEDIUM"),
            (sc3, "🔴 HIGH",   "71% – 100%","risk-high",   severity == "HIGH"),
        ]:
            border = "3px solid #1A73E8" if active else "none"
            col_s.markdown(f"""
            <div class="risk-card {cls}" style="border: {border}; opacity: {'1' if active else '0.45'}">
                <h3>{label}</h3>
                <p>{rng}</p>
                {'<p style="font-weight:700;font-size:0.9rem">◀ CURRENT</p>' if active else ''}
            </div>
            """, unsafe_allow_html=True)

        # ── Recommendations ─────────────────────────────────────────────
        st.markdown('<div class="section-title">💡 Safety Recommendations</div>', unsafe_allow_html=True)
        items_html = "".join(f"<li>{item}</li>" for item in recs["items"])
        st.markdown(f"""
        <div class="{recs['css']}">
            <h4>{recs['title']}</h4>
            <ul>{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif "📊 Data Insights" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>📊 Dataset Insights</h1>
        <p>Exploratory analysis of the India Flood Risk Dataset (10,000 records)</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "📄 Total Records",    f"{len(df):,}"),
        (c2, "🌊 Flood Events",     f"{df['Flood Occurred'].sum():,}"),
        (c3, "✅ Non-flood Events", f"{(df['Flood Occurred']==0).sum():,}"),
        (c4, "📐 Features",         str(df.shape[1])),
    ]:
        col.markdown(f"""
        <div class="metric-tile">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_files = [
        ("flood_occurrence.png",      "Flood Occurrence Distribution"),
        ("feature_importance.png",    "Feature Importance"),
        ("rainfall_distribution.png", "Rainfall Distribution"),
        ("river_discharge_dist.png",  "River Discharge Distribution"),
        ("rainfall_vs_waterlevel.png","Rainfall vs Water Level"),
    ]
    col_a, col_b = st.columns(2)
    for i, (fname, title) in enumerate(chart_files):
        path = IMAGES_DIR / fname
        target = col_a if i % 2 == 0 else col_b
        with target:
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Chart not found. Run `python model/train_model.py` to generate it.")

    # Summary statistics table
    st.markdown('<div class="section-title">📋 Summary Statistics</div>', unsafe_allow_html=True)
    num_cols = ["Rainfall (mm)", "Humidity (%)", "River Discharge (m³/s)",
                "Water Level (m)", "Elevation (m)"]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    # Raw data preview
    with st.expander("🗃️ View Raw Data (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>ℹ️ About This System</h1>
        <p>Technical overview of the AI Flood Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🌊 Project Overview
    This system predicts flood risk using a combination of **Machine Learning** 
    (Random Forest Classifier) and **domain-knowledge physics scoring** based on 
    real hydrological and meteorological thresholds.

    ---

    ## 🏗️ Architecture
    ```
    Historical Flood Dataset (India, 10,000 records)
                ↓
    Data Cleaning & Preprocessing
                ↓
    Feature Engineering (flood_pressure, risk_composite)
                ↓
    Random Forest Training (300 estimators)
                ↓
    Model saved as model.pkl
                ↓
    Streamlit UI — User inputs conditions
                ↓
    Hybrid Scoring: 70% Domain Physics + 30% ML
                ↓
    Flood Risk % → LOW / MEDIUM / HIGH
                ↓
    Safety Recommendations Display
    ```

    ---

    ## 📦 Feature Set
    | Feature | Description | Weight in Domain Score |
    |---|---|---|
    | Rainfall (mm) | Recent precipitation | 30% |
    | River Discharge (m³/s) | River flow volume | 20% |
    | Water Level (m) | Current water height | 20% |
    | Elevation (m) | Terrain height (inverse) | 15% |
    | Humidity (%) | Atmospheric moisture | 10% |
    | Historical Floods | Past flood events | 5% |

    ---

    ## ⚠️ Dataset Note
    The training dataset (`flood_risk_dataset_india.csv`) is **synthetically generated**
    with near-random flood labels (correlation < 0.03 between features and target).
    This is why the ML model alone achieves ~50% accuracy.

    The prediction system compensates with **domain-knowledge scoring** using real-world 
    hydrological thresholds validated against Indian flood literature:
    - 250 mm rainfall = extreme event threshold
    - 4000 m³/s river discharge = very high flow
    - 8 m water level = dangerously high
    - < 500 m elevation = flood-prone terrain

    ---

    ## 🛠️ Technology Stack
    | Component | Library |
    |---|---|
    | ML Model | scikit-learn RandomForestClassifier |
    | Data Processing | pandas, numpy |
    | Visualisation | matplotlib |
    | Interface | Streamlit |
    | Model Persistence | pickle |

    ---

    ## 📞 Emergency Contacts (India)
    - **NDMA Helpline**: 1078
    - **National Emergency**: 112
    - **IMD Flood Forecast**: imd.gov.in
    - **CWC River Data**: cwc.gov.in
    """)
