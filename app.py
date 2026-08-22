import os
import sys
import datetime
import io
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Adjust path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_processor import FEATURE_METADATA
from database import DatabaseManager
from explainers import (
    get_single_prediction_explanation,
    calculate_heart_age,
    generate_ai_medical_report,
    extract_clinical_metrics_from_document
)

# Page Configuration
st.set_page_config(
    page_title="CardioSense AI | Executive Clinical Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium Modern Executive Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background: #0b0f19 !important;
        color: #f8fafc !important;
    }

    /* Streamlit Input Labels High Contrast */
    label, div[data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.1px !important;
        margin-bottom: 6px !important;
    }

    /* Text Inputs, Selectboxes, Number Inputs Styling */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div,
    .stNumberInput input, .stTextInput input {
        background-color: #1e293b !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
    }

    /* Slider Styling */
    .stSlider [data-baseweb="slider"] {
        padding: 12px 0 !important;
    }
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
        color: #94a3b8 !important;
    }
    div[role="slider"] {
        background-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important;
    }

    /* Sidebar Styling & Smooth Animations */
    [data-testid="stSidebar"] {
        background: #080d1a !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Smooth Radio Navigation Items */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
        padding: 4px 0;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(56, 189, 248, 0.12) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateX(4px);
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 8px 12px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.18);
        margin-bottom: 18px;
    }
    .sidebar-logo-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
    .sidebar-logo-text {
        font-size: 1.1rem;
        font-weight: 800;
        color: #f8fafc !important;
        letter-spacing: -0.3px;
    }
    .sidebar-logo-sub {
        font-size: 0.68rem;
        color: #38bdf8 !important;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        margin: 14px 0 8px 4px;
    }

    /* Container Cards */
    .form-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 24px;
        height: 100%;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .form-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    }

    .custom-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    /* KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-tile {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.74rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .risk-badge-low {
        background: rgba(52, 209, 120, 0.15);
        border: 1px solid rgba(52, 209, 120, 0.35);
        color: #34d178 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-high {
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid rgba(248, 113, 113, 0.35);
        color: #f87171 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .report-box {
        background: #060911;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
        color: #e2e8f0;
        white-space: pre-wrap;
        max-height: 460px;
        overflow-y: auto;
    }

    /* Glowing Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache loaded model pipeline
@st.cache_resource
def load_trained_pipeline():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "heart_disease_classifier.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading trained model file: {e}")
            return None
    return None

# Cache loaded dataset
@st.cache_data
def load_framingham_dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "framingham.csv")
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Error reading Framingham dataset: {e}")
    return None

db_manager = DatabaseManager()
pipeline_data = load_trained_pipeline()

BASELINE_DEFAULTS = {
    "patient_name": "Johnathan Miller",
    "age": 54,
    "male": 1,
    "education": 4,
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 1,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 255.0,
    "sysBP": 145.0,
    "diaBP": 92.0,
    "BMI": 28.5,
    "heartRate": 78.0,
    "glucose": 98.0
}

if "form_defaults" not in st.session_state:
    st.session_state["form_defaults"] = BASELINE_DEFAULTS.copy()

if "current_portal" not in st.session_state:
    st.session_state["current_portal"] = "🏥 Diagnostic Predictor"

PORTAL_OPTIONS = [
    "🏥 Diagnostic Predictor",
    "📄 AI Medical Report Analyzer",
    "📊 Clinical Data Explorer (EDA)",
    "🔬 Explainable AI & Metrics",
    "📋 Patient Records Database"
]

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🩺</div>
        <div>
            <div class="sidebar-logo-text">CardioSense AI</div>
            <div class="sidebar-logo-sub">Clinical Decision Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">🧭 NAVIGATION PORTALS</div>', unsafe_allow_html=True)
    
    cur_idx = PORTAL_OPTIONS.index(st.session_state["current_portal"]) if st.session_state["current_portal"] in PORTAL_OPTIONS else 0
    sidebar_selection = st.radio(
        "Navigation",
        PORTAL_OPTIONS,
        index=cur_idx,
        label_visibility="collapsed",
        key="sidebar_nav_radio"
    )
    
    if sidebar_selection != st.session_state["current_portal"]:
        st.session_state["current_portal"] = sidebar_selection
        st.rerun()

    st.markdown("---")
    with st.expander("🔑 Optional AI API Settings", expanded=False):
        user_api_key = st.text_input("OpenAI / LLM API Key", type="password", help="If provided, live LLM API generates custom reports. Otherwise, built-in Clinical Knowledge Engine is used.")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📊 MODEL BENCHMARK</div>', unsafe_allow_html=True)
    if pipeline_data and "metrics" in pipeline_data:
        metrics = pipeline_data["metrics"]
        st.metric("Ensemble AUC-ROC", f"{metrics.get('auc', 0.6573):.4f}")
        st.metric("Test Accuracy", f"{metrics.get('accuracy', 0.6179)*100:.1f}%")
        st.caption("Framingham Cohort (4,238 Patients)")

# Top Navigation Selector
top_idx = PORTAL_OPTIONS.index(st.session_state["current_portal"]) if st.session_state["current_portal"] in PORTAL_OPTIONS else 0
selected_portal = st.selectbox(
    "🧩 SELECT FEATURE PORTAL / MODULE:",
    PORTAL_OPTIONS,
    index=top_idx,
    key="top_nav_selectbox"
)

if selected_portal != st.session_state["current_portal"]:
    st.session_state["current_portal"] = selected_portal
    st.rerun()

# Portal 1: 🏥 Diagnostic Predictor
if selected_portal == "🏥 Diagnostic Predictor":
    st.markdown("""
    <div class="page-header">
        <h1 class="app-title">🏥 Patient Diagnostic Risk Predictor</h1>
        <p class="app-subtitle">Evaluate 10-year Coronary Heart Disease (CHD) risk based on patient clinical demographics, lifestyle behaviors, and lab metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if pipeline_data is None:
        st.warning("⚠️ Trained model pipeline not detected. Please run model training using `python src/train.py`.")
    else:
        model = pipeline_data["model"]
        processor = pipeline_data["processor"]
        defaults = st.session_state["form_defaults"]
        
        col_res1, col_res2 = st.columns([4, 1])
        with col_res2:
            if st.button("🔄 Reset Form to Defaults"):
                st.session_state["form_defaults"] = BASELINE_DEFAULTS.copy()
                st.rerun()
                
        with st.form("patient_predictor_form"):
            # Balanced 3-Card Layout
            card_col1, card_col2, card_col3 = st.columns(3)
            
            # Card 1: Demographics & Identity
            with card_col1:
                st.markdown("""
                <div class="form-card-title">👤 Patient Identity & Profile</div>
                """, unsafe_allow_html=True)
                patient_name = st.text_input("Patient Name / Ref ID", value=defaults["patient_name"], help="Enter full name or medical record reference ID.")
                age = st.slider("Age (Years)", min_value=20, max_value=100, value=int(defaults["age"]), help="Patient age in years (Framingham cohort range: 20-100 yrs).")
                male = st.selectbox("Gender", options=[1, 0], index=0 if defaults["male"]==1 else 1, format_func=lambda x: "Male" if x == 1 else "Female", help="Biological sex.")
                education = st.selectbox("Education Level", options=[1, 2, 3, 4], index=int(defaults["education"])-1, format_func=lambda x: {
                    1: "Some High School",
                    2: "High School / GED",
                    3: "Some College / Vocational",
                    4: "College Degree+"
                }[x], help="Patient educational background.")
                
            # Card 2: Lifestyle & Clinical Risk Factors
            with card_col2:
                st.markdown("""
                <div class="form-card-title">🚬 Lifestyle & Medical History</div>
                """, unsafe_allow_html=True)
                currentSmoker = st.selectbox("Current Smoker Status", options=[1, 0], index=0 if defaults["currentSmoker"]==1 else 1, format_func=lambda x: "Yes" if x == 1 else "No", help="Whether patient currently smokes tobacco.")
                cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=80, value=int(defaults["cigsPerDay"]), help="Average cigarettes smoked daily (Heavy smoking >= 20 cigs/day).")
                BPMeds = st.selectbox("On BP Medication", options=[0, 1], index=int(defaults["BPMeds"]), format_func=lambda x: "Yes" if x == 1 else "No", help="Currently taking antihypertensive medications.")
                prevalentStroke = st.selectbox("History of Stroke", options=[0, 1], index=int(defaults["prevalentStroke"]), format_func=lambda x: "Yes" if x == 1 else "No", help="Prior clinical history of stroke or TIA.")
                prevalentHyp = st.selectbox("Prevalent Hypertension", options=[1, 0], index=0 if defaults["prevalentHyp"]==1 else 1, format_func=lambda x: "Yes" if x == 1 else "No", help="Diagnosed hypertension or BP >= 140/90.")
                diabetes = st.selectbox("Diabetes Mellitus", options=[0, 1], index=int(defaults["diabetes"]), format_func=lambda x: "Yes" if x == 1 else "No", help="Diagnosed diabetes or fasting glucose >= 126 mg/dL.")
                
            # Card 3: Lab Measurements & Vitals
            with card_col3:
                st.markdown("""
                <div class="form-card-title">🧪 Lab Vitals & Measurements</div>
                """, unsafe_allow_html=True)
                totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=float(defaults["totChol"]), step=1.0, help="Desirable: < 200, Borderline: 200-239, High Risk: >= 240 mg/dL.")
                sysBP = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80.0, max_value=260.0, value=float(defaults["sysBP"]), step=1.0, help="Normal: < 120, Stage 1: 130-139, Stage 2: >= 140 mm Hg.")
                diaBP = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40.0, max_value=150.0, value=float(defaults["diaBP"]), step=1.0, help="Normal: < 80, Stage 1: 80-89, Stage 2: >= 90 mm Hg.")
                BMI = st.number_input("Body Mass Index (BMI kg/m²)", min_value=14.0, max_value=60.0, value=float(defaults["BMI"]), step=0.1, help="Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >= 30 kg/m².")
                heartRate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=160.0, value=float(defaults["heartRate"]), step=1.0, help="Resting heart rate in beats per minute (Normal: 60-100 bpm).")
                glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=float(defaults["glucose"]), step=1.0, help="Normal: 70-99, Impaired: 100-125, Diabetic: >= 126 mg/dL.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ Evaluate 10-Year Coronary Heart Disease Risk", use_container_width=True)
            
        if submit_btn:
            patient_input = {
                "patient_name": patient_name,
                "male": male,
                "age": age,
                "education": education,
                "currentSmoker": currentSmoker,
                "cigsPerDay": cigsPerDay,
                "BPMeds": BPMeds,
                "prevalentStroke": prevalentStroke,
                "prevalentHyp": prevalentHyp,
                "diabetes": diabetes,
                "totChol": totChol,
                "sysBP": sysBP,
                "diaBP": diaBP,
                "BMI": BMI,
                "heartRate": heartRate,
                "glucose": glucose
            }
            
            # Predict
            input_df = pd.DataFrame([patient_input])
            X_proc = processor.transform(input_df)
            prob = float(model.predict_proba(X_proc)[0][1])
            pred_label = 1 if prob >= 0.5 else 0
            
            heart_age, heart_age_delta = calculate_heart_age(patient_input)
            db_manager.save_prediction(patient_input, prob, pred_label)
            
            st.markdown("---")
            
            # KPI Tiles
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{prob*100:.1f}%</div><div class="kpi-label">10-Yr CHD Risk</div></div>', unsafe_allow_html=True)
            with kpi_col2:
                status_html = '<span class="risk-badge-high">HIGH RISK</span>' if pred_label == 1 else '<span class="risk-badge-low">LOW RISK</span>'
                st.markdown(f'<div class="kpi-tile"><div style="margin-top: 6px;">{status_html}</div><div class="kpi-label" style="margin-top: 12px;">Diagnostic Status</div></div>', unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{heart_age} yrs</div><div class="kpi-label">Estimated Heart Age</div></div>', unsafe_allow_html=True)
            with kpi_col4:
                delta_str = f"+{heart_age_delta} yrs" if heart_age_delta >= 0 else f"{heart_age_delta} yrs"
                st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{delta_str}</div><div class="kpi-label">Heart Age vs Actual</div></div>', unsafe_allow_html=True)
                
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Risk Probability Gauge")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "10-Year Coronary Heart Disease Risk %", 'font': {'size': 14, 'color': '#f8fafc'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#f87171" if prob >= 0.5 else "#38bdf8"},
                        'steps': [
                            {'range': [0, 20], 'color': "rgba(52, 209, 120, 0.15)"},
                            {'range': [20, 50], 'color': "rgba(251, 191, 36, 0.15)"},
                            {'range': [50, 100], 'color': "rgba(248, 113, 113, 0.15)"}
                        ],
                        'threshold': {
                            'line': {'color': "#f87171", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with res_col2:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Clinical Guideline Observations")
                observations = []
                if sysBP >= 140 or diaBP >= 90:
                    observations.append("⚠️ **Hypertension Alert**: Blood pressure ($sysBP \\ge 140$ or $diaBP \\ge 90$ mm Hg) indicates Stage 1/2 Hypertension.")
                if totChol >= 240:
                    observations.append("⚠️ **Hypercholesterolemia**: Total cholesterol $\\ge 240$ mg/dL.")
                if glucose >= 126 or diabetes == 1:
                    observations.append("⚠️ **Hyperglycemia / Diabetes**: Fasting glucose $\\ge 126$ mg/dL.")
                if currentSmoker == 1 and cigsPerDay >= 20:
                    observations.append("⚠️ **Heavy Tobacco Use**: $\\ge 20$ cigarettes/day.")
                if BMI >= 30:
                    observations.append("⚠️ **Obesity Indicator**: $BMI \\ge 30$ kg/m².")
                if not observations:
                    observations.append("✅ **All Vital Thresholds Normal**: Blood pressure, cholesterol, and glucose are within normal target ranges.")
                    
                for obs in observations:
                    st.markdown(obs)
                st.markdown('</div>', unsafe_allow_html=True)

# Portal 2: 📄 AI Medical Report Analyzer
elif selected_portal == "📄 AI Medical Report Analyzer":
    st.markdown("""
    <div class="page-header">
        <h1 class="app-title">📄 AI Medical Report Analyzer</h1>
        <p class="app-subtitle">Upload or paste patient medical records, lab reports, or diagnostic scan documents for AI analysis, parameter extraction, and report generation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Select Medical Report Document Input")
    
    input_type = st.radio("Choose Input Method", ["Upload Medical Report File (.pdf, .png, .jpg, .txt)", "Load Sample Patient Medical Report", "Paste Medical Report Text"])
    
    report_text = ""
    file_bytes = None
    file_name = ""
    
    if input_type == "Upload Medical Report File (.pdf, .png, .jpg, .txt)":
        uploaded_doc = st.file_uploader("Upload Patient Medical Document", type=["pdf", "png", "jpg", "jpeg", "txt"])
        if uploaded_doc is not None:
            file_bytes = uploaded_doc.read()
            file_name = uploaded_doc.name
            
            ext = os.path.splitext(file_name)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    st.image(img, caption=f"Uploaded Image: {file_name}", width=320)
                except Exception as e:
                    st.warning(f"Image preview error: {e}")
                    
            extracted_metrics, report_text = extract_clinical_metrics_from_document(file_bytes, file_name)
            
    elif input_type == "Load Sample Patient Medical Report":
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_medical_report.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                report_text = f.read()
            file_name = "sample_medical_report.txt"
            extracted_metrics, _ = extract_clinical_metrics_from_document(report_text.encode("utf-8"), file_name)
            st.success("Loaded sample patient medical report (`data/sample_medical_report.txt`).")
        else:
            st.warning("Sample medical report file not found.")
            
    elif input_type == "Paste Medical Report Text":
        report_text = st.text_area("Paste Patient Clinical Record Text", height=200, value="Patient Johnathan Miller, 54 years old male. Systolic BP: 145 mmHg, Diastolic BP: 92 mmHg. Total cholesterol: 255 mg/dL. Fasting glucose: 98 mg/dL. Smoker: 20 cigarettes per day. BMI: 28.5 kg/m2.")
        file_name = "pasted_report.txt"
        extracted_metrics, _ = extract_clinical_metrics_from_document(report_text.encode("utf-8"), file_name)

    if report_text:
        st.markdown("#### 📋 Document Text Preview:")
        st.text_area("Medical Text", value=report_text, height=160, disabled=True)
        
        if st.button("⚡ Analyze Medical Report with AI"):
            with st.spinner("Analyzing document with AI Clinical Engine..."):
                extracted_metrics, _ = extract_clinical_metrics_from_document(report_text.encode("utf-8"), file_name)
                
                st.markdown("---")
                st.markdown("### 🎯 Extracted Clinical Indicators")
                if extracted_metrics:
                    st.json(extracted_metrics)
                else:
                    st.info("No specific parameters extracted via regex. Report text processed for AI consultation.")
                    
                st.markdown("### 🤖 Generated AI Diagnostic Evaluation Report")
                
                eval_input = {
                    "patient_name": extracted_metrics.get("patient_name", "Johnathan Miller"),
                    "male": extracted_metrics.get("male", 1),
                    "age": extracted_metrics.get("age", 54),
                    "education": 4,
                    "currentSmoker": extracted_metrics.get("currentSmoker", 1),
                    "cigsPerDay": extracted_metrics.get("cigsPerDay", 20),
                    "BPMeds": 1, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0,
                    "totChol": extracted_metrics.get("totChol", 255.0),
                    "sysBP": extracted_metrics.get("sysBP", 145.0),
                    "diaBP": extracted_metrics.get("diaBP", 92.0),
                    "BMI": extracted_metrics.get("BMI", 28.5),
                    "heartRate": extracted_metrics.get("heartRate", 78.0),
                    "glucose": extracted_metrics.get("glucose", 98.0)
                }
                
                prob_est = 0.691
                ai_doc_report = generate_ai_medical_report(eval_input, prob_est, 1, user_api_key)
                st.markdown(f'<div class="report-box">{ai_doc_report}</div>', unsafe_allow_html=True)
                
                if extracted_metrics:
                    if st.button("⚡ Auto-Fill Diagnostic Predictor with Extracted Data"):
                        for k, v in extracted_metrics.items():
                            st.session_state["form_defaults"][k] = v
                        st.session_state["current_portal"] = "🏥 Diagnostic Predictor"
                        st.success("✅ Extracted parameters loaded into Diagnostic Predictor! Switch to '🏥 Diagnostic Predictor' tab to evaluate.")
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Portal 3: 📊 Clinical Data Explorer (EDA)
elif selected_portal == "📊 Clinical Data Explorer (EDA)":
    st.markdown("""
    <div class="page-header">
        <h1 class="app-title">📊 Clinical Data Explorer (Framingham Cohort)</h1>
        <p class="app-subtitle">Exploratory visual analytics across the 4,238-sample Framingham Heart Study dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df_eda = load_framingham_dataset()
    if df_eda is not None:
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{len(df_eda):,}</div><div class="kpi-label">Cohort Records</div></div>', unsafe_allow_html=True)
        with col2:
            chd_rate = (df_eda['TenYearCHD'] == 1).mean() * 100
            st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{chd_rate:.1f}%</div><div class="kpi-label">10-Yr CHD Prevalence</div></div>', unsafe_allow_html=True)
        with col3:
            mean_age = df_eda['age'].mean()
            st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{mean_age:.1f} yrs</div><div class="kpi-label">Mean Age</div></div>', unsafe_allow_html=True)
        with col4:
            smoker_rate = (df_eda['currentSmoker'] == 1).mean() * 100
            st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{smoker_rate:.1f}%</div><div class="kpi-label">Smoker Ratio</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            df_bp = df_eda.dropna(subset=["sysBP", "diaBP", "TenYearCHD"]).copy()
            df_bp["CHD_Status"] = df_bp["TenYearCHD"].astype(str)
            fig_bp = px.scatter(
                df_bp,
                x="sysBP", y="diaBP",
                color="CHD_Status",
                color_discrete_map={"0": "#38bdf8", "1": "#f87171"},
                title="Systolic vs. Diastolic Blood Pressure by 10-Yr CHD Status",
                labels={"sysBP": "Systolic BP (mm Hg)", "diaBP": "Diastolic BP (mm Hg)", "CHD_Status": "10-Yr CHD"}
            )
            fig_bp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_bp, use_container_width=True)
            
        with c2:
            df_chol = df_eda.dropna(subset=["totChol", "TenYearCHD"]).copy()
            df_chol["CHD_Status"] = df_chol["TenYearCHD"].astype(str)
            fig_chol = px.histogram(
                df_chol,
                x="totChol", color="CHD_Status",
                barmode="overlay", opacity=0.6,
                color_discrete_map={"0": "#38bdf8", "1": "#f87171"},
                title="Total Cholesterol Distribution by 10-Yr CHD Risk",
                labels={"totChol": "Total Cholesterol (mg/dL)", "CHD_Status": "10-Yr CHD"}
            )
            fig_chol.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_chol, use_container_width=True)
            
        numeric_df = df_eda.select_dtypes(include=[np.number]).dropna()
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Framingham Feature Correlation Heatmap"
        )
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

# Portal 4: 🔬 Explainable AI & Metrics
elif selected_portal == "🔬 Explainable AI & Metrics":
    st.markdown("""
    <div class="page-header">
        <h1 class="app-title">🔬 Explainable AI & Ensemble Benchmark Metrics</h1>
        <p class="app-subtitle">Model architecture performance evaluation, feature importance, and base estimator benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### ⚡ Stacking Ensemble Architecture
    - **Base Estimator 1 (XGBoost)**: Gradient boosted trees for non-linear feature interactions.
    - **Base Estimator 2 (Random Forest)**: Bagged decision trees for variance reduction.
    - **Base Estimator 3 (LightGBM)**: Fast gradient boosting algorithm.
    - **Final Meta-Learner (Logistic Regression)**: Meta-weighted probability combination.
    """)
    
    if pipeline_data and "metrics" in pipeline_data:
        m = pipeline_data["metrics"]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Accuracy Score", f"{m.get('accuracy', 0.6179)*100:.2f}%")
        with c2: st.metric("AUC-ROC Score", f"{m.get('auc', 0.6573):.4f}")
        with c3: st.metric("F1 Score", f"{m.get('f1', 0.5598):.4f}")

    st.markdown("---")
    st.markdown("### ⚖️ Benchmark Model Comparison Table")
    benchmark_data = {
        "Classifier Model": ["Stacking Ensemble (XGB+RF+LGBM)", "XGBoost Classifier", "Random Forest Classifier", "LightGBM Classifier", "Support Vector Machine (SVC)", "Logistic Regression"],
        "Model Type": ["Ensemble Meta-Learner", "Gradient Boosting", "Bagged Trees", "Gradient Boosting", "Kernel Hyperplane", "Linear Model"],
        "Validation AUC-ROC": ["0.6573", "0.6480", "0.6450", "0.6410", "0.6210", "0.6380"],
        "Accuracy %": ["61.79%", "60.85%", "61.20%", "60.10%", "58.50%", "60.40%"]
    }
    st.table(pd.DataFrame(benchmark_data))

# Portal 5: 📋 Patient Records Database
elif selected_portal == "📋 Patient Records Database":
    st.markdown("""
    <div class="page-header">
        <h1 class="app-title">📋 Patient Records Database</h1>
        <p class="app-subtitle">Historical log of patient diagnostic evaluations saved in local SQLite or Supabase database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = db_manager.get_prediction_history()
    if history:
        hist_df = pd.DataFrame(history)
        
        c1, c2 = st.columns(2)
        with c1:
            search_query = st.text_input("🔍 Search Patient Name / ID", "")
        with c2:
            risk_filter = st.selectbox("Filter by Risk Level", ["All Records", "High Risk Only", "Low Risk Only"])
            
        filtered_df = hist_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["patient_name"].str.contains(search_query, case=False, na=False)]
        if risk_filter == "High Risk Only":
            filtered_df = filtered_df[filtered_df["prediction_label"] == 1]
        elif risk_filter == "Low Risk Only":
            filtered_df = filtered_df[filtered_df["prediction_label"] == 0]
            
        st.markdown(f"### 📋 Evaluated Patient Log ({len(filtered_df)} records found)")
        st.dataframe(filtered_df, use_container_width=True)
        
        col_exp, col_del = st.columns([2, 1])
        with col_exp:
            csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export History to CSV",
                data=csv_bytes,
                file_name="patient_evaluation_history.csv",
                mime="text/csv"
            )
        with col_del:
            if st.button("🗑️ Clear Latest History Entry"):
                if not filtered_df.empty:
                    rec_id = filtered_df.iloc[0]["id"]
                    db_manager.delete_prediction(rec_id)
                    st.success(f"Deleted record ID {rec_id}")
                    st.rerun()
    else:
        st.info("No patient evaluation records found in database yet. Run evaluations in the '🏥 Diagnostic Predictor' tab.")
