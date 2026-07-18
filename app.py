"""Main Streamlit application for AgriDrone crop disease monitoring."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import io

# Import project modules
import config
from grid_utils import load_field_from_json, create_grid_from_config, get_available_fields
from core_logic import run_scan
from llm_handler import generate_report, get_spray_advice, initialize_openai

# Page configuration
st.set_page_config(
    page_title="AgriDrone - Crop Disease Monitor",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Initialization ─────────────────────────────────
if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False
    st.session_state.scanned_grid = None
    st.session_state.path = []
    st.session_state.metrics = {
        "total": 0, "healthy": 0, "early": 0, "severe": 0
    }
    st.session_state.current_field = None
    st.session_state.current_field_name = "flat_farm.json"
    st.session_state.crop_type = "Wheat"

# ── Load Model (cached) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = config.MODELS_DIR / "disease_clf.pkl"
    if not model_path.exists():
        st.warning("⚠️ Model not found. Training new model...")
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from disease_model import generate_synthetic_data
        X, y = generate_synthetic_data(5000)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X, y)
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(clf, model_path)
        return clf
    else:
        import joblib
        return joblib.load(model_path)

def predict_disease(ndvi_value: float, crop_type: str) -> int:
    """Predict disease class from NDVI value using thresholds."""
    thresholds = config.NDVI_THRESHOLDS.get(crop_type, config.NDVI_THRESHOLDS["Wheat"])
    
    if ndvi_value >= thresholds["healthy_min"]:
        return 0  # Healthy
    elif ndvi_value >= thresholds["early_min"]:
        return 1  # Early Disease
    else:
        return 2  # Severe Disease

def create_feature_vector(ndvi: float, red: float, green: float, texture: float, moisture: float) -> np.ndarray:
    """Create feature vector for prediction."""
    return np.array([[ndvi, red, green, texture, moisture]])

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/drone.png", width=80)
    st.title("🚁 AgriDrone")
    st.caption("Crop Disease Monitor v2.0")
    st.divider()
    
    # Controls
    crop_type = st.selectbox(
        "🌾 Crop Type",
        ["Wheat", "Cotton", "Rice", "Sugarcane"],
        help="Select the crop type for disease threshold calibration"
    )
    st.session_state.crop_type = crop_type
    
    field_choice = st.selectbox(
        "🗺️ Field Layout",
        ["flat_farm.json", "pond_farm.json", "dense_field.json"],
        help="Select a pre-configured field layout"
    )
    st.session_state.current_field_name = field_choice
    
    uploaded_file = st.file_uploader(
        "📤 Upload Custom Field (JSON)",
        type=["json"],
        help="Upload your own field configuration JSON file"
    )
    
    spread_steps = st.slider(
        "🔄 Disease Spread Steps",
        min_value=0,
        max_value=10,
        value=5,
        help="Number of cellular automaton iterations for disease spread"
    )
    
    scan_speed = st.select_slider(
        "⚡ Scan Speed",
        options=["🐢 Slow", "🐇 Medium", "🚀 Fast"],
        value="🐇 Medium",
        help="Controls the simulation speed"
    )
    
    enable_seeds = st.toggle(
        "🌱 Manual Disease Seeds",
        value=False,
        help="Enable manual placement of disease seeds"
    )
    
    st.divider()
    
    run_clicked = st.button(
        "▶️ Run Drone Scan",
        type="primary",
        use_container_width=True
    )
    
    st.divider()
    
    # API Status
    api_ok = initialize_openai()
    if api_ok:
        st.success("✅ OpenRouter: Connected")
    else:
        st.warning("⚠️ OpenRouter: Not configured")
        st.caption("Set OPENROUTER_API_KEY in .env file")

# ── Main Page ──────────────────────────────────────────────────

# Header
st.title("🚁 Agricultural Drone — Crop Disease Monitor")
st.caption(f"📍 Field: {field_choice} | 🌾 Crop: {crop_type}")

# Load field data
try:
    field_path = config.DATA_DIR / field_choice
    if not field_path.exists():
        field_path = Path(field_choice)
    
    if uploaded_file is not None:
        field_data = json.load(uploaded_file)
    else:
        field_data = load_field_from_json(str(field_path))
    
    field_grid = create_grid_from_config(field_data)
    field_name = field_data.get('name', field_choice)
    
except Exception as e:
    st.error(f"❌ Error loading field: {e}")
    field_grid = np.zeros((config.GRID_ROWS, config.GRID_COLS))
    field_name = "Default Field"

# Handle Run button
if run_clicked:
    with st.spinner("🔄 Drone scanning field..."):
        seeds = []
        if enable_seeds:
            for seed in field_data.get('disease_seeds', []):
                cell = seed.get('cell', [])
                if len(cell) == 2:
                    dtype = 1 if seed.get('type') == 'early' else 2
                    seeds.append((cell[0], cell[1], dtype))
        
        scanned_grid, path = run_scan(field_grid.copy(), spread_steps, seeds if seeds else None)
        
        total = np.sum(scanned_grid != -1)
        healthy = np.sum(scanned_grid == 0)
        early = np.sum(scanned_grid == 1)
        severe = np.sum(scanned_grid == 2)
        
        st.session_state.scanned_grid = scanned_grid
        st.session_state.path = path
        st.session_state.metrics = {
            "total": int(total),
            "healthy": int(healthy),
            "early": int(early),
            "severe": int(severe)
        }
        st.session_state.scan_complete = True
        st.session_state.current_field = field_data
        
        st.rerun()

# ── Section 1: Metric Cards ──────────────────────────────────────
metrics = st.session_state.metrics

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📊 Cells Scanned",
        f"{metrics['total']:,}",
        delta=f"✅ {metrics['total']:,} cells" if metrics['total'] > 0 else None,
        delta_color="normal"
    )

with col2:
    st.metric(
        "🟢 Healthy",
        f"{metrics['healthy']:,}",
        delta=f"🌿 {metrics['healthy']:,}" if metrics['healthy'] > 0 else None,
        delta_color="normal"
    )

with col3:
    st.metric(
        "🟡 Early Disease",
        f"{metrics['early']:,}",
        delta=f"⚠️ {metrics['early']:,}" if metrics['early'] > 0 else None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "🔴 Severe Disease",
        f"{metrics['severe']:,}",
        delta=f"🚨 {metrics['severe']:,}" if metrics['severe'] > 0 else None,
        delta_color="inverse"
    )

# ── Section 2: Status Banner ────────────────────────────────────
if not st.session_state.scan_complete:
    st.info("📋 Load a field and click '▶️ Run Drone Scan' to start monitoring.")
else:
    st.success("✅ Scan complete! View the results below.")

# ── Section 3: Heatmap ──────────────────────────────────────────
if st.session_state.scan_complete and st.session_state.scanned_grid is not None:
    grid_to_show = st.session_state.scanned_grid.copy()
    heatmap_display = np.where(grid_to_show == -1, np.nan, grid_to_show)
    
    col_grid, col_legend = st.columns([3, 1])
    
    with col_grid:
        colorscale = [
            [0.0, "#2ecc71"],
            [0.33, "#2ecc71"],
            [0.34, "#f1c40f"],
            [0.66, "#f1c40f"],
            [0.67, "#e74c3c"],
            [1.0, "#e74c3c"]
        ]
        
        fig = px.imshow(
            heatmap_display,
            color_continuous_scale=colorscale,
            title="🗺️ Farm Grid - Disease Status",
            labels=dict(x="Column", y="Row", color="Disease Class"),
            aspect="equal",
            width=600,
            height=600
        )
        
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='closest'
        )
        
        fig.update_traces(
            hovertemplate="<b>📍 Cell (%{x}, %{y})</b><br>" +
                          "Disease Class: %{z}<br>" +
                          "<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_legend:
        st.markdown("### 📖 Legend")
        st.markdown("🟢 **Healthy** (0)")
        st.markdown("🟡 **Early Disease** (1)")
        st.markdown("🔴 **Severe Disease** (2)")
        st.markdown("⬛ **Obstacle**")
        
        st.markdown("---")
        st.markdown("### 📊 NDVI Scale")
        st.markdown("🟢 0.8+ : Healthy")
        st.markdown("🟡 0.4-0.6 : Early Disease")
        st.markdown("🔴 0.0-0.3 : Severe Disease")

# ── Section 4: Analytics Charts ──────────────────────────────────
if st.session_state.scan_complete and st.session_state.scanned_grid is not None:
    st.subheader("📊 Disease Analytics")
    
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        labels = ['Healthy', 'Early Disease', 'Severe Disease']
        values = [metrics['healthy'], metrics['early'], metrics['severe']]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=0.3
        )])
        
        fig_pie.update_layout(
            title="📊 Disease Distribution",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_bar:
        fig_bar = px.bar(
            x=['Healthy', 'Early Disease', 'Severe Disease'],
            y=values,
            color=['Healthy', 'Early Disease', 'Severe Disease'],
            color_discrete_map={
                'Healthy': '#2ecc71',
                'Early Disease': '#f1c40f',
                'Severe Disease': '#e74c3c'
            },
            title="📊 Disease Severity Counts",
            labels={'x': 'Disease Class', 'y': 'Number of Cells'}
        )
        
        fig_bar.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        fig_bar.update_traces(text=values, textposition='outside')
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed metrics expander
    with st.expander("📋 Detailed Metrics Table"):
        df_metrics = pd.DataFrame({
            "Metric": ["📊 Total Cells", "🟢 Healthy", "🟡 Early Disease", "🔴 Severe Disease", 
                      "📈 Affected %", "📈 Healthy %"],
            "Value": [
                metrics['total'],
                metrics['healthy'],
                metrics['early'],
                metrics['severe'],
                f"{(metrics['early'] + metrics['severe']) / max(metrics['total'], 1) * 100:.1f}%",
                f"{metrics['healthy'] / max(metrics['total'], 1) * 100:.1f}%"
            ]
        })
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

# ── Section 5: AI Field Report ──────────────────────────────────
st.subheader("🤖 AI Field Report")

col_report1, col_report2 = st.columns(2)

with col_report1:
    if st.button("📄 Generate Field Report", use_container_width=True):
        with st.spinner("🤖 Gemini is generating the field report..."):
            report = generate_report(
                metrics,
                field_name,
                crop_type
            )
            st.session_state.report = report
    
    if "report" in st.session_state and st.session_state.report:
        st.text_area(
            "📝 Field Health Report",
            st.session_state.report,
            height=200,
            key="report_text"
        )

with col_report2:
    if st.button("💊 Get Spray Advice", use_container_width=True):
        with st.spinner("🤖 Gemini is generating spray advice..."):
            advice = get_spray_advice(metrics, crop_type)
            st.session_state.spray_advice = advice
    
    if "spray_advice" in st.session_state and st.session_state.spray_advice:
        st.text_area(
            "🧪 Spray Recommendation",
            st.session_state.spray_advice,
            height=200,
            key="spray_text"
        )

# ── Section 6: Export ───────────────────────────────────────────
if st.session_state.scan_complete:
    st.divider()
    st.subheader("📤 Export Data")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        if st.session_state.scanned_grid is not None:
            df_export = pd.DataFrame(st.session_state.scanned_grid)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"scan_{field_choice.replace('.json', '')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_download2:
        if st.session_state.scanned_grid is not None:
            json_data = {
                "field": field_choice,
                "crop_type": crop_type,
                "grid": st.session_state.scanned_grid.tolist(),
                "metrics": metrics
            }
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"scan_{field_choice.replace('.json', '')}.json",
                mime="application/json",
                use_container_width=True
            )

# ── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption("🚁 AgriDrone v2.0 | Powered by Streamlit, Plotly, and AI")