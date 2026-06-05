"""
╔══════════════════════════════════════════════════════════════╗
║         WIND ENERGY ANALYTICS PLATFORM                      ║
║         Production-ready Streamlit Application              ║
║         Modules: Historical Analysis | ML Prediction |      ║
║                  Geographic Site Evaluation                  ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────
import io
import math
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Optional folium – graceful fallback to st.pydeck_chart
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    import pydeck as pdk
    HAS_FOLIUM = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Wind Energy Analytics",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --teal:    #0dcfb2;
    --blue:    #0a84ff;
    --navy:    #050f20;
    --card:    rgba(255,255,255,0.04);
    --border:  rgba(13,207,178,0.20);
    --text-hi: #e8f4f0;
    --text-lo: #6b8fa8;
    --green:   #23e87a;
    --amber:   #ffb830;
    --red:     #ff5757;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--navy) !important;
    color: var(--text-hi) !important;
}
.main .block-container { padding: 1.6rem 2.2rem 3rem; max-width: 1500px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5,18,40,0.95) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg,
        rgba(13,207,178,0.10) 0%,
        rgba(10,132,255,0.07) 60%,
        transparent 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🌬️';
    position: absolute;
    right: 1.8rem; top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.15;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--teal), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem;
}
.hero-sub { color: var(--text-lo); font-size: 0.92rem; letter-spacing: 0.5px; }

/* ── Section headers ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 3px;
    color: var(--teal);
    text-transform: uppercase;
    margin: 0 0 0.9rem;
}

/* ── KPI cards ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] p { color: var(--text-lo) !important; font-size: 0.78rem !important; letter-spacing: 1px; }
[data-testid="stMetricValue"]   { color: var(--teal) !important; font-family: 'Syne', sans-serif !important; font-size: 1.5rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(13,207,178,0.15), rgba(10,132,255,0.10));
    border: 1px solid var(--teal);
    color: var(--teal) !important;
    font-family: 'Syne', sans-serif;
    font-size: 0.76rem;
    letter-spacing: 1.5px;
    border-radius: 8px;
    padding: 0.5rem 1.3rem;
    transition: all .2s;
}
.stButton > button:hover { background: rgba(13,207,178,0.28); box-shadow: 0 0 14px rgba(13,207,178,0.35); }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: rgba(35,232,122,0.12);
    border: 1px solid var(--green);
    color: var(--green) !important;
    font-family: 'Syne', sans-serif;
    font-size: 0.76rem;
    letter-spacing: 1px;
    border-radius: 8px;
}

/* ── Sliders ── */
[data-testid="stSlider"] .rc-slider-track { background: var(--teal) !important; }
[data-testid="stSlider"] .rc-slider-handle { border-color: var(--teal) !important; background: var(--teal) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button { font-family: 'Syne', sans-serif !important; font-size: 0.74rem !important; letter-spacing: 1.5px !important; color: var(--text-lo) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-hi) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* ── Prediction result card ── */
.pred-card {
    background: linear-gradient(135deg, rgba(13,207,178,0.12), rgba(10,132,255,0.08));
    border: 1px solid var(--teal);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--teal), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.pred-label { color: var(--text-lo); font-size: 0.85rem; letter-spacing: 2px; margin-top: 0.3rem; }

/* ── Suitability badge ── */
.badge-hi  { color: #23e87a; font-weight: 700; }
.badge-med { color: #ffb830; font-weight: 700; }
.badge-lo  { color: #ff5757; font-weight: 700; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(13,207,178,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOTLY BASE THEME
# ══════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#b0ccd8"),
    title_font=dict(family="Syne, sans-serif", color="#0dcfb2", size=13),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(13,207,178,0.2)", borderwidth=1),
    margin=dict(l=30, r=20, t=48, b=30),
)

# ══════════════════════════════════════════════════════════════
# DATA GENERATION  (cached so it doesn't re-run on interaction)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def generate_historical_data() -> pd.DataFrame:
    """
    Produce 30 days × 24 h = 720 hourly records of realistic
    wind-farm telemetry using seeded random state for reproducibility.
    Power output follows a Weibull-inspired curve clipped to rated capacity.
    """
    rng  = np.random.default_rng(seed=42)
    n    = 30 * 24  # 720 rows

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("H"),
        periods=n,
        freq="H",
    )

    # Wind speed: Weibull-like (shape=2, scale=8) + diurnal sine wave
    hours        = np.arange(n)
    diurnal      = 1.5 * np.sin(2 * np.pi * hours / 24)
    wind_speed   = rng.weibull(2, n) * 8 + diurnal
    wind_speed   = np.clip(wind_speed, 0.5, 28.0)

    # Wind direction: slowly drifting bearing with noise
    bearing_base = np.cumsum(rng.normal(0, 3, n)) % 360
    wind_dir     = np.clip(bearing_base, 0, 359)

    # Ambient temperature: daily sinusoidal + seasonal drift
    day_idx     = hours / 24
    temperature = 15 + 8 * np.sin(2 * np.pi * day_idx / 30) \
                     + 5 * np.sin(2 * np.pi * hours / 24) \
                     + rng.normal(0, 1.0, n)

    # Rated turbine: cut-in=3, rated=12, cut-out=25 m/s, capacity=2000 kW
    def power_curve(ws, cut_in=3.0, rated_spd=12.0, cut_out=25.0, rated_pwr=2000.0):
        pwr = np.where(
            ws < cut_in, 0,
            np.where(
                ws < rated_spd,
                rated_pwr * ((ws - cut_in) / (rated_spd - cut_in)) ** 3,
                np.where(ws <= cut_out, rated_pwr, 0),
            ),
        )
        return pwr

    power_output = power_curve(wind_speed)
    # Add realistic measurement noise
    power_output = np.clip(
        power_output + rng.normal(0, 40, n), 0, 2000
    )

    return pd.DataFrame({
        "Timestamp":           timestamps,
        "Wind_Speed":          np.round(wind_speed,   2),
        "Wind_Direction":      np.round(wind_dir,     1),
        "Ambient_Temperature": np.round(temperature,  1),
        "Actual_Power_Output": np.round(power_output, 1),
    })


@st.cache_data
def generate_site_data() -> pd.DataFrame:
    """Five candidate wind-farm sites with geographic and performance attributes."""
    return pd.DataFrame({
        "Location":        ["Highland Ridge", "Coastal Bluff", "Prairie Flats",
                            "Desert Mesa",    "Offshore Beta"],
        "Latitude":        [53.5,  51.2,  48.9,  36.7,  54.8],
        "Longitude":       [-2.4,  -3.9,  -1.1,  -6.3,   2.1],
        "Avg_Wind_Speed":  [8.4,   9.1,   6.8,   7.5,  10.2],
        "AEP_GWh":         [142.0, 168.0, 98.0, 122.0, 215.0],
        "Suitability":     [78,    88,    55,    67,    95],
    })


@st.cache_resource
def train_ml_model(df: pd.DataFrame):
    """
    Train a GradientBoosting regressor on historical data.
    Features: Wind_Speed, Ambient_Temperature
    Target:   Actual_Power_Output
    Returns the fitted sklearn Pipeline.
    """
    X = df[["Wind_Speed", "Ambient_Temperature"]].values
    y = df["Actual_Power_Output"].values
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=7)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=7)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 1.2rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;
                  background:linear-gradient(90deg,#0dcfb2,#0a84ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🌬️ WIND ANALYTICS
      </div>
      <div style='color:#6b8fa8;font-size:0.72rem;letter-spacing:2px;margin-top:0.2rem'>
        ENERGY INTELLIGENCE SUITE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    module = st.radio(
        "Navigate",
        ["📊  Historical Analysis",
         "🤖  ML Power Prediction",
         "🗺️  Site Evaluation Map"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#6b8fa8;font-size:0.7rem;text-align:center;'
        'letter-spacing:1px;">BUILT WITH STREAMLIT · PLOTLY · SKLEARN</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">Wind Energy Analytics Platform</div>
  <div class="hero-sub">
    Real-time telemetry · Machine learning prediction · Geographic site intelligence
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODULE 1 – HISTORICAL DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
if "Historical" in module:
    df = generate_historical_data()

    # ── Sidebar filters ──────────────────────────────────────
    st.sidebar.markdown("### 📅 Date Range Filter")
    min_date = df["Timestamp"].dt.date.min()
    max_date = df["Timestamp"].dt.date.max()
    date_from = st.sidebar.date_input("From", value=min_date,
                                      min_value=min_date, max_value=max_date)
    date_to   = st.sidebar.date_input("To",   value=max_date,
                                      min_value=min_date, max_value=max_date)

    # Filter dataframe to selected date range
    mask  = (df["Timestamp"].dt.date >= date_from) & (df["Timestamp"].dt.date <= date_to)
    df_f  = df[mask].copy()

    if df_f.empty:
        st.warning("⚠️ No data in selected range. Adjust the date filter.")
        st.stop()

    # CSV download
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️  Download CSV",
        data=csv_bytes,
        file_name="wind_data_export.csv",
        mime="text/csv",
    )

    # ── KPI block ────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    rated_cap_kw = 2000.0
    total_mwh    = df_f["Actual_Power_Output"].sum() / 1000      # kWh → MWh
    peak_kw      = df_f["Actual_Power_Output"].max()
    avg_cf       = (df_f["Actual_Power_Output"].mean() / rated_cap_kw) * 100
    avg_wind     = df_f["Wind_Speed"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⚡ Total Generated",   f"{total_mwh:,.1f} MWh")
    k2.metric("📊 Avg Capacity Factor", f"{avg_cf:.1f} %")
    k3.metric("🔝 Peak Output",        f"{peak_kw:,.0f} kW")
    k4.metric("💨 Avg Wind Speed",     f"{avg_wind:.2f} m/s")

    st.markdown("---")

    # ── Chart 1: Time-series line ─────────────────────────────
    st.markdown('<div class="sec-title">📉 Wind Speed vs Power Output Over Time</div>',
                unsafe_allow_html=True)

    try:
        # Dual-axis: wind speed (left) and power (right)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_f["Timestamp"], y=df_f["Wind_Speed"],
            name="Wind Speed (m/s)",
            line=dict(color="#0dcfb2", width=1.5),
            opacity=0.85,
            yaxis="y1",
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_f["Timestamp"], y=df_f["Actual_Power_Output"],
            name="Power Output (kW)",
            line=dict(color="#ffb830", width=1.5),
            opacity=0.85,
            yaxis="y2",
            fill="tozeroy",
            fillcolor="rgba(255,184,48,0.06)",
        ))
        fig_ts.update_layout(
            **PL,
            height=360,
            title="Hourly Wind Speed & Power Output",
            yaxis=dict(title="Wind Speed (m/s)",   gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Power Output (kW)", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    except Exception as e:
        st.error(f"Time-series chart error: {e}")

    # ── Chart 2 & 3 side by side ─────────────────────────────
    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown('<div class="sec-title">🧭 Wind Rose (Direction × Speed)</div>',
                    unsafe_allow_html=True)
        try:
            # Bin directions into 16 compass sectors
            bins   = np.arange(-11.25, 360, 22.5)
            labels = [f"{int(b+11.25)}°" for b in bins[:-1]]
            df_f["Dir_Bin"] = pd.cut(df_f["Wind_Direction"], bins=bins, labels=labels)
            # Count per bin
            rose_df = (df_f.groupby("Dir_Bin", observed=True)["Wind_Speed"]
                         .mean().reset_index()
                         .rename(columns={"Wind_Speed": "Avg_Speed"}))

            fig_rose = go.Figure(go.Barpolar(
                r=rose_df["Avg_Speed"],
                theta=rose_df["Dir_Bin"].astype(str),
                marker=dict(
                    color=rose_df["Avg_Speed"],
                    colorscale=[[0, "#050f20"], [0.4, "#0dcfb2"], [1, "#0a84ff"]],
                    showscale=True,
                    colorbar=dict(title="m/s", tickfont=dict(color="#b0ccd8")),
                ),
                opacity=0.88,
            ))
            fig_rose.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                   tickfont=dict(color="#b0ccd8")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                     tickfont=dict(color="#b0ccd8")),
                ),
                font=dict(family="DM Sans", color="#b0ccd8"),
                title=dict(text="Avg Wind Speed by Direction",
                           font=dict(family="Syne", color="#0dcfb2", size=13)),
                margin=dict(l=20, r=20, t=50, b=20),
                height=380,
            )
            st.plotly_chart(fig_rose, use_container_width=True)
        except Exception as e:
            st.error(f"Wind rose error: {e}")

    with cc2:
        st.markdown('<div class="sec-title">📊 Wind Speed Distribution</div>',
                    unsafe_allow_html=True)
        try:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_f["Wind_Speed"],
                nbinsx=30,
                name="Frequency",
                marker=dict(color="#0dcfb2", opacity=0.65,
                            line=dict(color="#050f20", width=0.5)),
            ))
            # KDE trend line via numpy
            from scipy.stats import gaussian_kde  # soft import inside try-except
            kde_x = np.linspace(df_f["Wind_Speed"].min(), df_f["Wind_Speed"].max(), 200)
            kde   = gaussian_kde(df_f["Wind_Speed"], bw_method=0.3)
            kde_y = kde(kde_x) * len(df_f) * (df_f["Wind_Speed"].max() - df_f["Wind_Speed"].min()) / 30
            fig_hist.add_trace(go.Scatter(
                x=kde_x, y=kde_y,
                mode="lines", name="KDE",
                line=dict(color="#ffb830", width=2.5),
            ))
            fig_hist.update_layout(
                **PL, height=380,
                title="Wind Speed Histogram with KDE",
                xaxis_title="Wind Speed (m/s)",
                yaxis_title="Count",
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except ImportError:
            # Fallback without scipy
            fig_hist = px.histogram(df_f, x="Wind_Speed", nbins=30,
                                    title="Wind Speed Distribution",
                                    color_discrete_sequence=["#0dcfb2"])
            fig_hist.update_layout(**PL, height=380)
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Histogram error: {e}")

    # ── Raw data expander ─────────────────────────────────────
    with st.expander("🔍 Inspect Raw Data Table"):
        st.dataframe(
            df_f.tail(200).sort_values("Timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════
# MODULE 2 – ML POWER PREDICTION
# ══════════════════════════════════════════════════════════════
elif "ML" in module:
    df  = generate_historical_data()
    mdl = train_ml_model(df)

    st.markdown('<div class="sec-title">⚙️ Turbine Parameter Controls</div>',
                unsafe_allow_html=True)

    col_sl, col_res = st.columns([1.3, 1])

    with col_sl:
        wind_speed   = st.slider("💨 Wind Speed (m/s)",          0.0, 30.0, 9.0,  0.1)
        blade_length = st.slider("⚙️ Blade Length / Rotor R (m)", 20.0, 80.0, 45.0, 0.5)
        efficiency   = st.slider("🔋 Generator Efficiency",       0.70, 0.98, 0.90, 0.01)
        air_density  = st.slider("🌡 Air Density (kg/m³)",        1.10, 1.35, 1.225, 0.005)
        amb_temp     = st.slider("🌡 Ambient Temperature (°C)",   -10.0, 45.0, 15.0, 0.5)

        # ── Physics-based power curve formula ────────────────
        # P = 0.5 × ρ × A × v³ × Cp × η
        # Betz limit Cp ≈ 0.45, η = generator efficiency
        CUT_IN   = 3.0
        RATED_V  = 12.0
        CUT_OUT  = 25.0
        CP       = 0.45    # aerodynamic power coefficient (Betz limit ≈ 0.593)

        rotor_area    = math.pi * blade_length ** 2
        raw_power_w   = 0.5 * air_density * rotor_area * (wind_speed ** 3) * CP * efficiency
        raw_power_kw  = raw_power_w / 1000

        # Apply cut-in / cut-out logic
        if wind_speed < CUT_IN or wind_speed > CUT_OUT:
            physics_kw = 0.0
        elif wind_speed < RATED_V:
            physics_kw = raw_power_kw
        else:
            physics_kw = raw_power_kw  # capped by rated in real units; displayed as-is here

        physics_kw = max(0.0, physics_kw)

        # ML model prediction (temperature-adjusted)
        try:
            ml_kw = float(mdl.predict([[wind_speed, amb_temp]])[0])
            ml_kw = max(0.0, ml_kw)
        except Exception:
            ml_kw = physics_kw  # fallback

    with col_res:
        st.markdown('<div class="sec-title">🔋 Predicted Power Output</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-value">{physics_kw:,.1f}<span style='font-size:1.2rem'> kW</span></div>
          <div class="pred-label">PHYSICS MODEL (Betz + Efficiency)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ML model output
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("🤖 ML Prediction",  f"{ml_kw:,.1f} kW")
        col_m2.metric("📐 Rotor Area",     f"{rotor_area:,.0f} m²")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(13,207,178,0.15);
                    border-radius:10px;padding:1rem 1.2rem;font-size:0.82rem;color:#6b8fa8;'>
        <b style='color:#0dcfb2'>Formula:</b><br>
        P = ½ × ρ × A × v³ × Cₚ × η<br><br>
        ρ = {air_density} kg/m³ &nbsp;|&nbsp; A = {rotor_area:,.0f} m²<br>
        v = {wind_speed} m/s &nbsp;|&nbsp; Cₚ = {CP} &nbsp;|&nbsp; η = {efficiency}
        </div>
        """, unsafe_allow_html=True)

    # ── Power Curve Plot ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-title">📈 Theoretical vs ML-Predicted Power Curve</div>',
                unsafe_allow_html=True)

    try:
        v_range = np.linspace(0, 30, 300)

        # Theoretical curve (physics)
        def physics_curve(ws_arr):
            out = []
            for ws in ws_arr:
                if ws < CUT_IN or ws > CUT_OUT:
                    out.append(0.0)
                else:
                    p = 0.5 * air_density * rotor_area * (ws**3) * CP * efficiency / 1000
                    out.append(max(0.0, p))
            return np.array(out)

        phys_pwr = physics_curve(v_range)

        # ML curve (predict at current amb_temp for varying wind speeds)
        ml_inputs = np.column_stack([v_range, np.full_like(v_range, amb_temp)])
        ml_pwr    = np.clip(mdl.predict(ml_inputs), 0, None)

        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(
            x=v_range, y=phys_pwr,
            name="Physics Model",
            line=dict(color="#0dcfb2", width=3),
            mode="lines",
        ))
        fig_pc.add_trace(go.Scatter(
            x=v_range, y=ml_pwr,
            name="ML Prediction",
            line=dict(color="#ffb830", width=2.5, dash="dot"),
            mode="lines",
        ))
        # Vertical marker at current wind speed
        fig_pc.add_vline(
            x=wind_speed, line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=f"  {wind_speed} m/s",
            annotation_font_color="#0dcfb2",
        )
        fig_pc.add_vrect(x0=0, x1=CUT_IN, fillcolor="rgba(255,87,87,0.05)",
                         line_width=0, annotation_text="Below cut-in",
                         annotation_font_color="#ff5757", annotation_position="top left")
        fig_pc.add_vrect(x0=CUT_OUT, x1=30, fillcolor="rgba(255,87,87,0.05)",
                         line_width=0, annotation_text="Above cut-out",
                         annotation_font_color="#ff5757")

        fig_pc.update_layout(
            **PL, height=400,
            title=f"Power Curve — Blade R={blade_length}m · ρ={air_density} kg/m³ · η={efficiency}",
            xaxis_title="Wind Speed (m/s)",
            yaxis_title="Power Output (kW)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_pc, use_container_width=True)
    except Exception as e:
        st.error(f"Power curve error: {e}")


# ══════════════════════════════════════════════════════════════
# MODULE 3 – GEOGRAPHIC SITE EVALUATION
# ══════════════════════════════════════════════════════════════
elif "Site" in module or "Map" in module:
    sites = generate_site_data()

    # ── Sidebar filter ────────────────────────────────────────
    st.sidebar.markdown("### 🌬️ Site Filter")
    min_wind = st.sidebar.slider(
        "Min Avg Wind Speed (m/s)",
        float(sites["Avg_Wind_Speed"].min()),
        float(sites["Avg_Wind_Speed"].max()),
        6.0, 0.1,
    )
    sites_f = sites[sites["Avg_Wind_Speed"] >= min_wind].copy()

    st.markdown('<div class="sec-title">🗺️ Candidate Wind Farm Locations</div>',
                unsafe_allow_html=True)

    if sites_f.empty:
        st.warning("No sites match the current wind speed filter. Lower the threshold.")
    else:
        # ── Map rendering ─────────────────────────────────────
        if HAS_FOLIUM:
            # Folium interactive map
            m = folium.Map(
                location=[sites_f["Latitude"].mean(), sites_f["Longitude"].mean()],
                zoom_start=5,
                tiles="CartoDB dark_matter",
            )

            def suitability_color(score):
                if score >= 80: return "#23e87a"
                if score >= 60: return "#ffb830"
                return "#ff5757"

            for _, row in sites_f.iterrows():
                color = suitability_color(row["Suitability"])
                radius = 8 + row["Avg_Wind_Speed"] * 1.5  # size by wind speed

                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.75,
                    popup=folium.Popup(
                        f"""<b>{row['Location']}</b><br>
                        Wind: {row['Avg_Wind_Speed']} m/s<br>
                        AEP: {row['AEP_GWh']} GWh/yr<br>
                        Score: {row['Suitability']}/100""",
                        max_width=200,
                    ),
                    tooltip=row["Location"],
                ).add_to(m)

            st_folium(m, width=None, height=480)

        else:
            # PyDeck fallback map
            try:
                # Normalize suitability to 0-255 for colour channels
                sites_f["r"] = sites_f["Suitability"].apply(lambda s: 35 if s >= 80 else 255 if s >= 60 else 255)
                sites_f["g"] = sites_f["Suitability"].apply(lambda s: 232 if s >= 80 else 184 if s >= 60 else 87)
                sites_f["b"] = sites_f["Suitability"].apply(lambda s: 122 if s >= 80 else 48 if s >= 60 else 87)
                sites_f["radius"] = sites_f["Avg_Wind_Speed"] * 18000

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=sites_f,
                    get_position="[Longitude, Latitude]",
                    get_radius="radius",
                    get_fill_color="[r, g, b, 190]",
                    pickable=True,
                )
                view = pdk.ViewState(
                    latitude=sites_f["Latitude"].mean(),
                    longitude=sites_f["Longitude"].mean(),
                    zoom=4, pitch=30,
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[layer], initial_view_state=view,
                    map_style="mapbox://styles/mapbox/dark-v10",
                    tooltip={"text": "{Location}\nWind: {Avg_Wind_Speed} m/s\nScore: {Suitability}"},
                ))
            except Exception as e:
                st.error(f"Map render error: {e}")
                st.map(sites_f.rename(columns={"Latitude": "lat", "Longitude": "lon"}))

        # ── Bubble scatter: AEP vs Wind Speed ─────────────────
        st.markdown("---")
        st.markdown('<div class="sec-title">📊 Site Performance Matrix</div>',
                    unsafe_allow_html=True)

        try:
            fig_bubble = px.scatter(
                sites_f,
                x="Avg_Wind_Speed", y="AEP_GWh",
                size="Suitability", color="Suitability",
                hover_name="Location",
                color_continuous_scale=["#ff5757", "#ffb830", "#23e87a"],
                size_max=55,
                labels={"Avg_Wind_Speed": "Avg Wind Speed (m/s)",
                        "AEP_GWh": "Annual Energy Production (GWh)",
                        "Suitability": "Score"},
                title="AEP vs Wind Speed — bubble size = Suitability Score",
            )
            fig_bubble.update_traces(marker=dict(line=dict(color="#050f20", width=1.5)))
            fig_bubble.update_layout(**PL, height=380,
                                     coloraxis_colorbar=dict(
                                         tickfont=dict(color="#b0ccd8"),
                                         title="Score",
                                     ))
            st.plotly_chart(fig_bubble, use_container_width=True)
        except Exception as e:
            st.error(f"Scatter error: {e}")

        # ── Data table ────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-title">📋 Site Data Table</div>', unsafe_allow_html=True)

        def score_badge(s):
            if s >= 80: return f'<span class="badge-hi">● {s}</span>'
            if s >= 60: return f'<span class="badge-med">● {s}</span>'
            return f'<span class="badge-lo">● {s}</span>'

        display_df = sites_f.copy()
        display_df["Suitability"] = display_df["Suitability"].apply(score_badge)
        st.write(
            display_df[["Location", "Avg_Wind_Speed", "AEP_GWh", "Suitability"]]
            .to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#2a4a60;font-family:Syne,sans-serif;'
    'font-size:0.72rem;letter-spacing:2px;">'
    'WIND ENERGY ANALYTICS PLATFORM  ·  STREAMLIT + PLOTLY + SCIKIT-LEARN'
    '</div>',
    unsafe_allow_html=True,
)
