import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PKT            = timezone(timedelta(hours=5))
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "")
LAT, LON       = 24.8607, 67.0011
MODEL_FILE     = "aqi_model.pkl"
RIDGE_FILE     = "aqi_ridge_model.pkl"
TF_MODEL_FILE  = "aqi_tf_model.keras"
SCALER_FILE    = "scaler.pkl"
CLEAN_FILE     = "karachi_clean_dataset.csv"

FEATURES = [
    "pm25","pm10","no2","co","o3","so2","nh3",
    "hour","day_of_week","month",
    "aqi_lag_1h","aqi_lag_3h","aqi_change"
]

AQI_INFO = {
    1: {"label":"Good",      "color":"#16a34a","dark_color":"#4ade80",
        "light":"#dcfce7","border":"#86efac","icon":"😊",
        "advice":"Air quality is excellent. Great day for outdoor activities!"},
    2: {"label":"Fair",      "color":"#ca8a04","dark_color":"#facc15",
        "light":"#fef9c3","border":"#fde047","icon":"🙂",
        "advice":"Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion."},
    3: {"label":"Moderate",  "color":"#ea580c","dark_color":"#fb923c",
        "light":"#fff7ed","border":"#fdba74","icon":"😐",
        "advice":"Sensitive groups should limit prolonged outdoor exposure. Consider wearing a mask."},
    4: {"label":"Poor",      "color":"#dc2626","dark_color":"#f87171",
        "light":"#fef2f2","border":"#fca5a5","icon":"😷",
        "advice":"Everyone should reduce prolonged outdoor exertion. Wear a mask if going outside."},
    5: {"label":"Very Poor", "color":"#7c3aed","dark_color":"#a78bfa",
        "light":"#f5f3ff","border":"#c4b5fd","icon":"🚨",
        "advice":"HAZARDOUS. Avoid all outdoor activities. Stay indoors with windows closed."},
}

# ── Streamlit theme config (light/dark auto) ──────────────────────────────────
# Create .streamlit/config.toml in your project root with:
#   [theme]
#   base = "light"
# OR set via sidebar toggle below

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #f8fafc !important; }
.block-container { padding: 1.2rem 2rem 2rem !important; max-width: 1400px !important; }

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color: #64748b !important; font-size:0.8rem !important; }
section[data-testid="stSidebar"] h3 { color: #1e293b !important; }
section[data-testid="stSidebar"] input {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
}

.dash-header {
    background: linear-gradient(135deg,#ffffff 0%,#f0f9ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dash-title { font-size:1.8rem; font-weight:700; color:#1e293b; margin:0; }
.dash-sub   { color:#94a3b8; font-size:0.8rem; margin:3px 0 0 0; }
.city-badge {
    background:#f0fdf4; border:1px solid #86efac;
    border-radius:20px; padding:6px 14px;
    color:#16a34a; font-size:0.78rem; font-weight:600;
}

.aqi-big {
    border-radius:16px; padding:26px 20px; text-align:center;
    border:1px solid; position:relative; overflow:hidden;
}
.aqi-icon { font-size:2.2rem; margin-bottom:4px; }
.aqi-num  { font-size:3.8rem; font-weight:700; line-height:1; margin:4px 0; }
.aqi-lbl  { font-size:0.85rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; }
.aqi-src  { font-size:0.68rem; color:#94a3b8; margin-top:6px; }

.metric-card {
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px; text-align:center;
}
.mval { font-size:1.7rem; font-weight:700; color:#1e293b; }
.mlbl { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }
.mico { font-size:1.3rem; margin-bottom:4px; }

.model-compare {
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px;
}
.model-badge {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:0.72rem; font-weight:600; margin-bottom:8px;
}
.rf-badge  { background:#dbeafe; color:#1d4ed8; }
.rid-badge { background:#fce7f3; color:#9d174d; }

.alert-box {
    border-radius:12px; padding:16px 18px; border:1px solid;
}
.alert-title { font-weight:700; font-size:0.9rem; margin-bottom:6px; }
.alert-text  { font-size:0.75rem; line-height:1.5; }

.sec {
    font-size:0.7rem; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:2px;
    margin:20px 0 10px; display:flex; align-items:center; gap:8px;
}
.sec::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,#e2e8f0,transparent);
}

.fc {
    background:#ffffff; border:1px solid #e2e8f0;
    border-radius:14px; padding:20px 14px; text-align:center;
}
.fc-day { font-size:0.7rem; color:#94a3b8; letter-spacing:1px; text-transform:uppercase; }
.fc-ico  { font-size:1.8rem; margin:6px 0; }
.fc-aqi  { font-size:2.6rem; font-weight:700; margin:2px 0; }
.fc-lbl  { font-size:0.72rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; }
.fc-rng  { font-size:0.68rem; color:#cbd5e1; margin-top:8px; }

.hl { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px; text-align:center; }
.hl-v { font-size:1.5rem; font-weight:700; }
.hl-l { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

.srow { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #f1f5f9; }
.srow:last-child { border-bottom:none; }
.sk { font-size:0.78rem; color:#94a3b8; }
.sv { font-size:0.78rem; color:#1e293b; font-weight:500; }

.leg { border-radius:8px; padding:10px; text-align:center; font-size:0.76rem; font-weight:600; }

.stTabs [data-baseweb="tab-list"] {
    background:#ffffff !important; border-radius:10px !important;
    padding:3px !important; border:1px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#94a3b8 !important;
    border-radius:7px !important; font-size:0.8rem !important; padding:7px 14px !important;
}
.stTabs [aria-selected="true"] { background:#f0f9ff !important; color:#0284c7 !important; }

hr { border-color:#e2e8f0 !important; margin:16px 0 !important; }
div[data-testid="stExpander"] {
    border:1px solid #e2e8f0 !important; border-radius:10px !important;
    background:#ffffff !important;
}
.footer { text-align:center; color:#cbd5e1; font-size:0.68rem; padding:14px 0 2px; }
</style>
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #0f172a !important; }
.block-container { padding: 1.2rem 2rem 2rem !important; max-width: 1400px !important; }

section[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size:0.8rem !important; }
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] input {
    background: #0f172a !important; border:1px solid #334155 !important;
    color: #f1f5f9 !important; border-radius:8px !important;
}

.dash-header {
    background: linear-gradient(135deg,#1e293b 0%,#0c1a2e 100%);
    border:1px solid #334155; border-radius:16px;
    padding:22px 28px; margin-bottom:18px;
    display:flex; align-items:center; justify-content:space-between;
}
.dash-title { font-size:1.8rem; font-weight:700; color:#f1f5f9; margin:0; }
.dash-sub   { color:#64748b; font-size:0.8rem; margin:3px 0 0 0; }
.city-badge {
    background:#14532d; border:1px solid #166534;
    border-radius:20px; padding:6px 14px;
    color:#4ade80; font-size:0.78rem; font-weight:600;
}

.aqi-big {
    border-radius:16px; padding:26px 20px; text-align:center;
    border:1px solid; position:relative; overflow:hidden;
}
.aqi-icon { font-size:2.2rem; margin-bottom:4px; }
.aqi-num  { font-size:3.8rem; font-weight:700; line-height:1; margin:4px 0; }
.aqi-lbl  { font-size:0.85rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; }
.aqi-src  { font-size:0.68rem; color:#475569; margin-top:6px; }

.metric-card {
    background:#1e293b; border:1px solid #334155;
    border-radius:12px; padding:16px; text-align:center;
}
.mval { font-size:1.7rem; font-weight:700; color:#f1f5f9; }
.mlbl { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }
.mico { font-size:1.3rem; margin-bottom:4px; }

.model-compare {
    background:#1e293b; border:1px solid #334155;
    border-radius:12px; padding:16px;
}
.model-badge { display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.72rem; font-weight:600; margin-bottom:8px; }
.rf-badge  { background:#1e3a5f; color:#60a5fa; }
.rid-badge { background:#4a1340; color:#f472b6; }

.alert-box { border-radius:12px; padding:16px 18px; border:1px solid; }
.alert-title { font-weight:700; font-size:0.9rem; margin-bottom:6px; }
.alert-text  { font-size:0.75rem; line-height:1.5; }

.sec {
    font-size:0.7rem; font-weight:600; color:#475569;
    text-transform:uppercase; letter-spacing:2px;
    margin:20px 0 10px; display:flex; align-items:center; gap:8px;
}
.sec::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#334155,transparent); }

.fc {
    background:#1e293b; border:1px solid #334155;
    border-radius:14px; padding:20px 14px; text-align:center;
}
.fc-day { font-size:0.7rem; color:#475569; letter-spacing:1px; text-transform:uppercase; }
.fc-ico  { font-size:1.8rem; margin:6px 0; }
.fc-aqi  { font-size:2.6rem; font-weight:700; margin:2px 0; }
.fc-lbl  { font-size:0.72rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; }
.fc-rng  { font-size:0.68rem; color:#334155; margin-top:8px; }

.hl { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; text-align:center; }
.hl-v { font-size:1.5rem; font-weight:700; }
.hl-l { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

.srow { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e293b; }
.srow:last-child { border-bottom:none; }
.sk { font-size:0.78rem; color:#64748b; }
.sv { font-size:0.78rem; color:#e2e8f0; font-weight:500; }

.leg { border-radius:8px; padding:10px; text-align:center; font-size:0.76rem; font-weight:600; }

.stTabs [data-baseweb="tab-list"] {
    background:#1e293b !important; border-radius:10px !important;
    padding:3px !important; border:1px solid #334155 !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#475569 !important;
    border-radius:7px !important; font-size:0.8rem !important; padding:7px 14px !important;
}
.stTabs [aria-selected="true"] { background:#0c2340 !important; color:#38bdf8 !important; }

hr { border-color:#334155 !important; margin:16px 0 !important; }
div[data-testid="stExpander"] {
    border:1px solid #334155 !important; border-radius:10px !important;
    background:#1e293b !important;
}
.footer { text-align:center; color:#334155; font-size:0.68rem; padding:14px 0 2px; }
</style>
"""


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_rf_model():
    return joblib.load(MODEL_FILE)

@st.cache_resource
def load_ridge_model(df):
    """Train Ridge on first 80% only — last 20% reserved for honest evaluation."""
    data  = df[FEATURES + ["aqi"]].dropna().reset_index(drop=True)
    split = int(len(data) * 0.8)
    X_train = data[FEATURES].values[:split]
    y_train = data["aqi"].values[:split]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, RIDGE_FILE)
    return pipe

@st.cache_resource
def load_tf_model():
    if not TF_AVAILABLE:
        return None, None
    try:
        model = tf.keras.models.load_model(TF_MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception:
        return None, None

def predict_tf(tf_model, scaler, inp_df):
    """Run TF model prediction, returns int AQI 1-5."""
    X = scaler.transform(inp_df[FEATURES].values)
    raw = tf_model.predict(X, verbose=0)[0][0]
    return max(1, min(5, int(round(raw))))

def get_tf_metrics():
    """Return hardcoded TF test metrics from train_tf_model.py run."""
    return {"rmse": 0.1700, "mae": 0.1254, "r2": 0.8699}

@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    return df

@st.cache_data
def get_ridge_metrics(df):
    """Evaluate Ridge on unseen 20% hold-out (model was trained on first 80%)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    data  = df[FEATURES + ["aqi"]].dropna().reset_index(drop=True)
    split = int(len(data) * 0.8)
    X_test = data[FEATURES].values[split:]
    y_test = data["aqi"].values[split:]
    ridge  = load_ridge_model(df)
    y_pred_raw   = ridge.predict(X_test)
    y_pred_round = np.clip(np.round(y_pred_raw), 1, 5)
    acc  = np.mean(y_pred_round == y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_raw))
    mae  = mean_absolute_error(y_test, y_pred_raw)
    r2   = r2_score(y_test, y_pred_raw)
    return {"accuracy": acc, "rmse": rmse, "mae": mae, "r2": r2}


# ── API ───────────────────────────────────────────────────────────────────────
def fetch_live(key):
    try:
        r = requests.get(
            f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={key}",
            timeout=5).json()
        if "list" not in r: return None
        d = r["list"][0]
        return {"aqi":d["main"]["aqi"],"pm25":d["components"]["pm2_5"],
                "pm10":d["components"]["pm10"],"no2":d["components"]["no2"],
                "co":d["components"]["co"],"o3":d["components"]["o3"],
                "so2":d["components"]["so2"],"nh3":d["components"]["nh3"]}
    except: return None


# ── Forecast ──────────────────────────────────────────────────────────────────
def forecast_72h(model, df, model_type="rf"):
    rows, w = [], df.tail(10).copy().reset_index(drop=True)
    now = datetime.now(PKT)
    for i in range(1, 73):
        ft = now + timedelta(hours=i)
        l1 = w.iloc[-1]
        l3 = w["aqi"].iloc[-3] if len(w) >= 3 else w["aqi"].iloc[0]
        lp = w["aqi"].iloc[-2] if len(w) >= 2 else w["aqi"].iloc[0]
        inp = pd.DataFrame([{
            "pm25":float(l1["pm25"]),"pm10":float(l1["pm10"]),
            "no2":float(l1["no2"]),"co":float(l1["co"]),
            "o3":float(l1["o3"]),"so2":float(l1["so2"]),"nh3":float(l1["nh3"]),
            "hour":ft.hour,"day_of_week":ft.weekday(),"month":ft.month,
            "aqi_lag_1h":float(l1["aqi"]),"aqi_lag_3h":float(l3),
            "aqi_change":float(l1["aqi"])-float(lp)
        }])
        raw = model.predict(inp)[0]
        pred = max(1, min(5, int(round(raw))))
        rows.append({"datetime":ft,"predicted_aqi":pred,
                     "label":AQI_INFO[pred]["label"],"color":AQI_INFO[pred]["color"]})
        nr = l1.copy(); nr["aqi"] = pred; nr["timestamp"] = ft
        w = pd.concat([w, pd.DataFrame([nr])], ignore_index=True)
    return pd.DataFrame(rows)


# ── Chart helpers ─────────────────────────────────────────────────────────────
def pcfg(h=260, dark=False):
    bg = "#1e293b" if dark else "#ffffff"
    grid = "#334155" if dark else "#f1f5f9"
    line = "#475569" if dark else "#e2e8f0"
    txt  = "#94a3b8"
    return dict(
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="Inter", color=txt, size=11),
        height=h, margin=dict(l=8,r=8,t=8,b=8),
        xaxis=dict(gridcolor=grid, linecolor=line),
        yaxis=dict(gridcolor=grid, linecolor=line),
        showlegend=False, hovermode="x unified",
    )

def chart_forecast(fdf, rf_fdf=None, dark=False):
    fig = go.Figure()
    hrs  = list(range(len(fdf)))
    avals = fdf["predicted_aqi"].tolist()
    dts   = fdf["datetime"].dt.strftime("%a %b %d, %H:%M PKT").tolist()

    # Ridge line
    fig.add_trace(go.Scatter(
        x=hrs, y=avals, mode="lines", name="Ridge Regression",
        line=dict(color="#db2777", width=2, shape="spline", dash="dot"),
        hovertemplate="<b>%{customdata}</b><br>Ridge AQI: <b>%{y}</b><extra></extra>",
        customdata=dts, showlegend=True))

    # RF line
    if rf_fdf is not None:
        rf_vals = rf_fdf["predicted_aqi"].tolist()
        fig.add_trace(go.Scatter(
            x=hrs, y=rf_vals, mode="lines", name="Random Forest",
            line=dict(color="#0284c7", width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(2,132,199,0.07)",
            hovertemplate="RF AQI: <b>%{y}</b><extra></extra>",
            showlegend=True))

    tvals = list(range(0, 73, 12))
    ttext = [fdf["datetime"].iloc[i].strftime("%b %d\n%H:%M") if i < len(fdf) else "" for i in tvals]
    cfg = pcfg(280, dark)
    cfg["xaxis"]["tickvals"]  = tvals
    cfg["xaxis"]["ticktext"]  = ttext
    cfg["yaxis"]["tickvals"]  = [1,2,3,4,5]
    cfg["yaxis"]["ticktext"]  = ["Good","Fair","Moderate","Poor","Very Poor"]
    cfg["yaxis"]["range"]     = [0.5, 5.5]
    cfg["showlegend"]         = True
    cfg["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11, color="#94a3b8")
    )
    fig.update_layout(**cfg)
    return fig

def chart_historical(df, dark=False):
    df7   = df.tail(168).copy()
    avals = df7["aqi"].tolist()
    ts    = list(range(len(avals)))
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=avals, mode="lines",
        line=dict(color="#7c3aed", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.06)",
        hovertemplate="Hour -%{x}<br>AQI: <b>%{y}</b><extra></extra>"))
    cfg = pcfg(250, dark)
    cfg["xaxis"]["title"]    = "Hours ago (0 = most recent)"
    cfg["xaxis"]["autorange"] = "reversed"
    cfg["yaxis"]["tickvals"] = [1,2,3,4,5]
    cfg["yaxis"]["ticktext"] = ["Good","Fair","Moderate","Poor","Very Poor"]
    cfg["yaxis"]["range"]    = [0.5, 5.5]
    fig.update_layout(**cfg)
    return fig

def chart_pollutants(pm25, pm10, no2, co, o3, so2, dark=False):
    cats  = ["PM2.5","PM10","NO₂","CO/100","O₃","SO₂"]
    raw   = [pm25, pm10, no2, co, o3, so2]
    mx    = [75, 150, 0.5, 10000, 180, 0.5]
    norms = [min(1.0, max(0.0, v/m)) for v, m in zip(raw, mx)]
    clrs  = ["#dc2626" if n > 0.7 else "#ea580c" if n > 0.4 else "#16a34a" for n in norms]
    disp  = [pm25, pm10, no2, co, o3, so2]
    fig   = go.Figure(go.Bar(
        x=cats, y=norms, marker_color=clrs,
        text=[f"{v:.1f}" for v in disp],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=10)))
    bg  = "#1e293b" if dark else "#f8fafc"
    cfg = pcfg(210, dark)
    cfg["yaxis"]["title"]      = "Relative vs Safe Threshold"
    cfg["yaxis"]["tickformat"] = ".0%"
    cfg["yaxis"]["range"]      = [0, 1.3]
    cfg["plot_bgcolor"]        = bg
    cfg["paper_bgcolor"]       = bg
    fig.update_layout(**cfg)
    return fig

def chart_monthly(df, dark=False):
    month_names = {11:"Nov",12:"Dec",1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May"}
    monthly = df.groupby("month")["aqi"].mean()
    x    = [month_names.get(m, str(m)) for m in monthly.index]
    clrs = ["#16a34a" if v < 2.5 else "#ea580c" if v < 3.5 else "#dc2626"
            for v in monthly.values]
    fig = go.Figure(go.Bar(
        x=x, y=monthly.values, marker_color=clrs,
        text=[f"{v:.2f}" for v in monthly.values],
        textposition="outside", textfont=dict(color="#94a3b8", size=10)))
    bg  = "#1e293b" if dark else "#f8fafc"
    cfg = pcfg(220, dark)
    cfg["yaxis"]["title"]  = "Avg AQI"
    cfg["yaxis"]["range"]  = [0, 5.5]
    cfg["plot_bgcolor"]    = bg
    cfg["paper_bgcolor"]   = bg
    fig.update_layout(**cfg)
    return fig

def shap_chart(model, df, dark=False):
    sample = df[FEATURES].dropna().tail(100)
    expl   = shap.TreeExplainer(model)
    sv     = expl.shap_values(sample)
    if isinstance(sv, list): ms = np.mean([np.abs(s).mean(0) for s in sv], 0)
    else:
        sv_arr = np.array(sv)
        ms = np.abs(sv_arr).mean(0) if sv_arr.ndim == 2 else np.abs(sv_arr).mean((0,1))
    ms = np.array(ms).flatten()[:len(FEATURES)]
    idx    = np.argsort(ms)
    fnames = [FEATURES[i] for i in idx]
    fvals  = [float(ms[i]) for i in idx]

    bg_clr = "#1e293b" if dark else "#f8fafc"
    txt_clr = "#94a3b8"
    fig, ax = plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor(bg_clr); ax.set_facecolor(bg_clr)
    bclrs = ["#dc2626" if v == max(fvals)
             else "#16a34a" if v >= np.percentile(fvals, 70)
             else "#0284c7" if v >= np.percentile(fvals, 40)
             else "#cbd5e1" for v in fvals]
    bars = ax.barh(fnames, fvals, color=bclrs, height=0.6)
    for bar, val in zip(bars, fvals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", color=txt_clr, fontsize=8)
    ax.set_xlabel("Mean |SHAP Value|", color=txt_clr, fontsize=9)
    ax.set_title("Feature Importance — What Drives AQI?",
                 color="#f1f5f9" if dark else "#1e293b", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=txt_clr, labelsize=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#334155" if dark else "#e2e8f0")
    ax.grid(axis="x", alpha=0.5, color="#334155" if dark else "#f1f5f9")
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        rf_model = load_rf_model()
        df       = load_data()
        ridge_model = load_ridge_model(df)
        tf_model, tf_scaler = load_tf_model()
        tf_available = tf_model is not None
    except Exception as e:
        st.error(f"❌ {e}"); st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Theme toggle
        dark_mode = st.toggle("Dark Mode", value=False)

        api_key = st.text_input("OpenWeather API Key", value=OPENWEATHER_KEY,
                                type="password", placeholder="Paste API key...")
        if api_key: os.environ["OPENWEATHER_KEY"] = api_key

        # Model selector
        st.caption("💡 Delete aqi_ridge_model.pkl to retrain Ridge")
        st.markdown("---")
        model_options = ["Random Forest", "Ridge Regression"]
        if tf_available:
            model_options.append("TensorFlow Neural Net")
        active_model_name = st.radio(
            "🤖 Active Forecast Model",
            model_options,
            index=0,
            help="Switch between ML models for the 72-hour forecast"
        )
        if active_model_name == "Random Forest":
            active_model = rf_model
        elif active_model_name == "Ridge Regression":
            active_model = ridge_model
        else:
            active_model = None  # TF handled separately

        # RF metrics
        st.markdown("---")
        rf_color = "#1e293b" if dark_mode else "#f8fafc"
        st.markdown(f"""
        <div style="background:{rf_color};border:1px solid {'#334155' if dark_mode else '#e2e8f0'};border-radius:10px;padding:14px;">
        <div style="color:{'#64748b'};font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">📊 Model Comparison</div>
        <div style="font-size:0.72rem;font-weight:600;color:#3b82f6;margin-bottom:6px;">🌲 Random Forest</div>
        <div class="srow"><span class="sk">Accuracy</span><span class="sv" style="color:#16a34a;">99.66%</span></div>
        <div class="srow"><span class="sk">RMSE</span><span class="sv">0.0583</span></div>
        <div class="srow"><span class="sk">MAE</span><span class="sv">0.0034</span></div>
        <div class="srow"><span class="sk">R²</span><span class="sv">0.9955</span></div>
        <div style="font-size:0.72rem;font-weight:600;color:#ec4899;margin:10px 0 6px;">📐 Ridge Regression</div>
        <div class="srow"><span class="sk">Trained on</span><span class="sv">same dataset</span></div>
        <div class="srow"><span class="sk">Alpha</span><span class="sv">1.0</span></div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:8px;">Metrics computed on 20% hold-out</div>
        </div>
        """, unsafe_allow_html=True)

        # Ridge live metrics
        with st.spinner("Computing Ridge metrics..."):
            rm = get_ridge_metrics(df)
        st.markdown(f"""
        <div style="margin-top:8px;background:{rf_color};border:1px solid {'#334155' if dark_mode else '#e2e8f0'};border-radius:10px;padding:14px;">
        <div class="srow"><span class="sk">Accuracy</span><span class="sv" style="color:#ec4899;">{rm['accuracy']*100:.1f}%</span></div>
        <div class="srow"><span class="sk">RMSE</span><span class="sv">{rm['rmse']:.4f}</span></div>
        <div class="srow"><span class="sk">MAE</span><span class="sv">{rm['mae']:.4f}</span></div>
        <div class="srow"><span class="sk">R²</span><span class="sv">{rm['r2']:.4f}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # TF metrics
        if tf_available:
            tm = get_tf_metrics()
            st.markdown(f"""
            <div style="margin-top:8px;background:{rf_color};border:1px solid {'#334155' if dark_mode else '#e2e8f0'};border-radius:10px;padding:14px;">
            <div style="font-size:0.72rem;font-weight:600;color:#7c3aed;margin-bottom:6px;">🧠 TensorFlow Neural Net</div>
            <div class="srow"><span class="sk">Architecture</span><span class="sv">Dense 64→32→1</span></div>
            <div class="srow"><span class="sk">RMSE</span><span class="sv">{tm['rmse']:.4f}</span></div>
            <div class="srow"><span class="sk">MAE</span><span class="sv">{tm['mae']:.4f}</span></div>
            <div class="srow"><span class="sk">R²</span><span class="sv" style="color:#7c3aed;">{tm['r2']:.4f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div class="srow"><span class="sk">City</span><span class="sv">Karachi, PK</span></div>
        <div class="srow"><span class="sk">Timezone</span><span class="sv" style="color:#0284c7;">PKT (UTC+5)</span></div>
        <div class="srow"><span class="sk">Samples</span><span class="sv">4,410</span></div>
        <div class="srow"><span class="sk">Training</span><span class="sv">Nov 25–May 26</span></div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<a href="https://openweathermap.org/api/air-pollution" target="_blank" style="color:#94a3b8;font-size:0.75rem;">🔗 OpenWeatherMap API</a>', unsafe_allow_html=True)

    # ── Apply theme CSS ───────────────────────────────────────────────────────
    st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    now_str = datetime.now(PKT).strftime("%a, %d %b %Y  %H:%M PKT")
    st.markdown(f"""
    <div class="dash-header">
        <div>
            <div class="dash-title">🌤️ Karachi AQI Predictor</div>
            <div class="dash-sub">Real-time Air Quality Monitoring & 3-Day Forecast · Random Forest + Ridge Regression + TensorFlow</div>
        </div>
        <div style="text-align:right;">
            <div class="city-badge">📍 Karachi, Pakistan</div>
            <div style="color:{'#475569' if dark_mode else '#cbd5e1'};font-size:0.7rem;margin-top:6px;">{now_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch live data ───────────────────────────────────────────────────────
    live = fetch_live(api_key) if api_key else None
    last = df.iloc[-1]
    if live:
        aqi=int(live["aqi"]); pm25=live["pm25"]; pm10=live["pm10"]
        no2=live["no2"]; co=live["co"]; o3=live["o3"]; so2=live["so2"]; src="🟢 Live API"
    else:
        aqi=int(last["aqi"]); pm25=float(last["pm25"]); pm10=float(last["pm10"])
        no2=float(last.get("no2",0)); co=float(last.get("co",0))
        o3=float(last.get("o3",0)); so2=float(last.get("so2",0)); src="🟡 Cached"

    inf = AQI_INFO.get(aqi, AQI_INFO[3])
    lbl, clr, light, border, icon, advice = (
        inf["label"], inf["color"], inf["light"], inf["border"], inf["icon"], inf["advice"]
    )
    # Use dark-friendly colors if dark mode
    if dark_mode:
        clr = inf["dark_color"]

    # ── Current AQI ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📍 Current Air Quality — Karachi</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1.6])

    # Also get Ridge prediction for current conditions
    inp_now = pd.DataFrame([{
        "pm25": pm25, "pm10": pm10, "no2": no2, "co": co,
        "o3": o3, "so2": so2, "nh3": float(last.get("nh3", 0)),
        "hour": datetime.now(PKT).hour,
        "day_of_week": datetime.now(PKT).weekday(),
        "month": datetime.now(PKT).month,
        "aqi_lag_1h": float(last["aqi"]),
        "aqi_lag_3h": float(df["aqi"].iloc[-3]) if len(df) >= 3 else float(last["aqi"]),
        "aqi_change": float(df["aqi"].iloc[-1]) - float(df["aqi"].iloc[-2]) if len(df) >= 2 else 0
    }])
    ridge_now = max(1, min(5, int(round(ridge_model.predict(inp_now)[0]))))
    ridge_inf = AQI_INFO.get(ridge_now, AQI_INFO[3])

    with c1:
        st.markdown(f"""
        <div class="aqi-big" style="background:{'#1a2a1a' if dark_mode else light};border-color:{border};">
        <div style="position:absolute;top:0;left:0;right:0;height:4px;
                    background:{clr};border-radius:16px 16px 0 0;"></div>
        <div class="aqi-icon">{icon}</div>
        <div class="aqi-num" style="color:{clr};">{aqi}</div>
        <div class="aqi-lbl" style="color:{clr};">{lbl}</div>
        <div class="aqi-src">{src} · {datetime.now(PKT).strftime('%H:%M PKT')}</div>
        </div>""", unsafe_allow_html=True)

    for col, (ico, name, val, unit) in zip([c2, c3, c4, c5], [
        ("💨","PM2.5", f"{pm25:.1f}","µg/m³"),
        ("🌫️","PM10",  f"{pm10:.1f}","µg/m³"),
        ("🔬","NO₂",   f"{no2:.2f}", "µg/m³"),
        ("⚗️","O₃",    f"{o3:.1f}",  "µg/m³")]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
            <div class="mico">{ico}</div>
            <div class="mval">{val}</div>
            <div class="mlbl">{name} {unit}</div>
            </div>""", unsafe_allow_html=True)

    with c6:
        ridge_clr = ridge_inf["dark_color"] if dark_mode else ridge_inf["color"]
        st.markdown(f"""
        <div class="model-compare">
            <div class="model-badge rf-badge">🌲 RF → {lbl}</div>
            <div style="font-size:1.4rem;font-weight:700;color:{clr};margin:2px 0 10px;">{aqi}</div>
            <div class="model-badge rid-badge">📐 Ridge → {ridge_inf['label']}</div>
            <div style="font-size:1.4rem;font-weight:700;color:{ridge_clr};margin:2px 0 8px;">{ridge_now}</div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:6px;">{icon} {advice[:60]}...</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Pollutants ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🧪 Today\'s Highlights — Pollutant Levels</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_pollutants(pm25, pm10, no2, co, o3, so2, dark=dark_mode),
                    use_container_width=True, key="chart_pollutants")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "3-Day Forecast",
        "Model Comparison",
        "Historical Trends",
        "SHAP Explainability"
    ])

    with tab1:
        with st.spinner("Generating 72-hour forecast..."):
            rf_fdf    = forecast_72h(rf_model, df, "rf")
            ridge_fdf = forecast_72h(ridge_model, df, "ridge")
            fdf       = rf_fdf if active_model_name == "Random Forest" else ridge_fdf

        fdf["date"] = fdf["datetime"].dt.date
        daily = fdf.groupby("date")["predicted_aqi"].agg(["mean","min","max"]).reset_index().head(3)
        daily.columns = ["Date","Avg","Min","Max"]

        st.markdown(f'<div class="sec">Daily Summary — {active_model_name}</div>', unsafe_allow_html=True)
        dcols = st.columns(3)
        for i, (_, row) in enumerate(daily.iterrows()):
            avg = max(1, min(5, int(round(row["Avg"]))))
            di  = AQI_INFO[avg]
            dstr = pd.Timestamp(row["Date"]).strftime("%A, %b %d")
            dclr = di["dark_color"] if dark_mode else di["color"]
            with dcols[i]:
                st.markdown(f"""
                <div class="fc" style="border-top:4px solid {dclr};">
                <div class="fc-day">{dstr}</div>
                <div class="fc-ico">{di['icon']}</div>
                <div class="fc-aqi" style="color:{dclr};">{avg}</div>
                <div class="fc-lbl" style="color:{dclr};">{di['label']}</div>
                <div class="fc-rng">Min {int(row['Min'])} &nbsp;/&nbsp; Max {int(row['Max'])}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:16px;">72-Hour Hourly Forecast — Both Models</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_forecast(ridge_fdf, rf_fdf, dark=dark_mode), use_container_width=True, key="chart_forecast_tab1")

        with st.expander("View full hourly forecast table"):
            disp = fdf[["datetime","predicted_aqi","label"]].copy()
            disp["datetime"] = disp["datetime"].dt.strftime("%a %b %d, %H:%M PKT")
            disp.columns = ["Date & Time (PKT)","AQI Level","Category"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<div class="sec">Random Forest vs Ridge Regression vs TensorFlow — Side by Side</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("""
            <div style="background:#dbeafe;border-radius:10px;padding:14px;margin-bottom:12px;">
            <div style="font-weight:700;color:#1d4ed8;margin-bottom:8px;">🌲 Random Forest</div>
            <div style="font-size:0.78rem;color:#1e40af;">
            • Ensemble of 100+ decision trees<br>
            • Handles non-linear relationships<br>
            • Resistant to overfitting<br>
            • Best for complex feature interactions
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Accuracy:** 99.66% &nbsp; **R²:** 0.9955 &nbsp; **RMSE:** 0.0583")
        with m2:
            st.markdown(f"""
            <div style="background:#fce7f3;border-radius:10px;padding:14px;margin-bottom:12px;">
            <div style="font-weight:700;color:#9d174d;margin-bottom:8px;">📐 Ridge Regression</div>
            <div style="font-size:0.78rem;color:#831843;">
            • Regularized linear regression (α=1.0)<br>
            • Fast, interpretable baseline<br>
            • L2 regularization prevents overfitting<br>
            • Good for linear feature relationships
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Accuracy:** {rm['accuracy']*100:.1f}% &nbsp; **R²:** {rm['r2']:.4f} &nbsp; **RMSE:** {rm['rmse']:.4f}")
        with m3:
            tm = get_tf_metrics()
            tf_status = "✅ Loaded" if tf_available else "⚠️ File not found"
            st.markdown(f"""
            <div style="background:#f5f3ff;border-radius:10px;padding:14px;margin-bottom:12px;">
            <div style="font-weight:700;color:#6d28d9;margin-bottom:8px;">🧠 TensorFlow Neural Net</div>
            <div style="font-size:0.78rem;color:#5b21b6;">
            • Dense layers: 64 → 32 → 1<br>
            • ReLU activation, Adam optimizer<br>
            • Feature scaling with StandardScaler<br>
            • 50 epochs, batch size 32
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**R²:** {tm['r2']:.4f} &nbsp; **RMSE:** {tm['rmse']:.4f} &nbsp; **MAE:** {tm['mae']:.4f} &nbsp; *{tf_status}*")

        st.markdown('<div class="sec" style="margin-top:8px;">Forecast Comparison Chart</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_forecast(ridge_fdf, rf_fdf, dark=dark_mode), use_container_width=True, key="chart_forecast_tab2")

        st.info("💡 **Why three models?** Random Forest captures complex non-linear patterns for highest accuracy. Ridge Regression is a fast interpretable baseline. TensorFlow Neural Net demonstrates deep learning on the same data. RF outperforms TF here because lag features favour tree-based models.")

    with tab3:
        col_h1, col_h2 = st.columns([2,1])
        with col_h1:
            st.markdown('<div class="sec">Last 7 Days — Hourly AQI</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_historical(df, dark=dark_mode), use_container_width=True, key="chart_historical")
        with col_h2:
            st.markdown('<div class="sec">Monthly Average AQI</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_monthly(df, dark=dark_mode), use_container_width=True, key="chart_monthly")

        hist  = df["aqi"].tail(168)
        hcols = st.columns(4)
        for col, (lbl2, val, clr2) in zip(hcols, [
            ("Avg AQI",   f"{hist.mean():.1f}", AQI_INFO.get(round(hist.mean()),   AQI_INFO[3])["color"]),
            ("Peak AQI",  f"{int(hist.max())}",  AQI_INFO.get(int(hist.max()),    AQI_INFO[5])["color"]),
            ("Best AQI",  f"{int(hist.min())}",  AQI_INFO.get(int(hist.min()),    AQI_INFO[1])["color"]),
            ("Hazardous", f"{int((hist>=4).sum())}h", "#dc2626"),
        ]):
            with col:
                st.markdown(f"""
                <div class="hl">
                <div class="hl-v" style="color:{clr2};">{val}</div>
                <div class="hl-l">{lbl2}</div>
                </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec">What Drives the AQI Prediction?</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.8rem;margin-bottom:12px;">SHAP quantifies how much each feature influences the Random Forest model output. Longer bar = stronger influence.</p>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP values..."):
            sf = shap_chart(rf_model, df, dark=dark_mode)
        st.pyplot(sf); plt.close()

    st.markdown("---")

    # ── AQI Legend ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📖 AQI Scale Reference</div>', unsafe_allow_html=True)
    lcols = st.columns(5)
    for i, (lvl, inf2) in enumerate(AQI_INFO.items()):
        lc = inf2["dark_color"] if dark_mode else inf2["color"]
        lb = "#334155" if dark_mode else inf2["border"]
        ll = "#1e293b" if dark_mode else inf2["light"]
        with lcols[i]:
            st.markdown(f"""
            <div class="leg" style="background:{ll};border:1px solid {lb};color:{lc};">
            {inf2['icon']} {lvl} — {inf2['label']}
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">
    Built for 10Pearls Data Sciences Internship &nbsp;·&nbsp;
    Data: OpenWeatherMap API &nbsp;·&nbsp;
    Models: Random Forest + Ridge Regression + TensorFlow &nbsp;·&nbsp;
    Timezone: PKT (UTC+5)
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
