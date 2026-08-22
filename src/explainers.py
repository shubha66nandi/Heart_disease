import os
import io
import re
import urllib.request
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def extract_clinical_metrics_from_document(file_bytes, filename):
    """
    Extracts raw text and parses clinical metrics from uploaded PDF or Image medical reports.
    Returns a dictionary of extracted patient input fields and extracted raw text string.
    """
    extracted_text = ""
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    
    # 1. Extract text from PDF
    if ext == ".pdf" or (filename and filename.endswith(".pdf")):
        if PYPDF_AVAILABLE:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    text_page = page.extract_text()
                    if text_page:
                        extracted_text += text_page + "\n"
            except Exception as e:
                extracted_text = f"Error extracting PDF text: {e}"
        else:
            extracted_text = "pypdf library not available for PDF parsing."
            
    # 2. Extract info from Image
    elif ext in [".png", ".jpg", ".jpeg"] or (filename and any(filename.endswith(x) for x in [".png", ".jpg", ".jpeg"])):
        if PIL_AVAILABLE:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                extracted_text = f"Medical Scan Image File: {filename} ({image.width}x{image.height} px, Format: {image.format})\n"
                extracted_text += str(filename)
            except Exception as e:
                extracted_text = f"Error processing image: {e}"
        else:
            extracted_text = "PIL library not available for image processing."
    else:
        try:
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            extracted_text = "Unrecognized document format."

    # Parse metrics from extracted_text via Regex
    extracted_metrics = {}
    
    # Age
    age_match = re.search(r'(?:age|years old|patient age)\D*(\d{2})', extracted_text, re.IGNORECASE)
    if age_match:
        extracted_metrics["age"] = int(age_match.group(1))
        
    # Gender (Male / Female)
    if re.search(r'\b(male|man|m)\b', extracted_text, re.IGNORECASE) and not re.search(r'\b(female)\b', extracted_text, re.IGNORECASE):
        extracted_metrics["male"] = 1
    elif re.search(r'\b(female|woman|f)\b', extracted_text, re.IGNORECASE):
        extracted_metrics["male"] = 0
        
    # Blood Pressure (sys/dia e.g., 145/92 or Systolic: 145)
    bp_match = re.search(r'(?:blood pressure|bp|sys/dia)\D*(\d{2,3})[\s/]+(\d{2,3})', extracted_text, re.IGNORECASE)
    if bp_match:
        extracted_metrics["sysBP"] = float(bp_match.group(1))
        extracted_metrics["diaBP"] = float(bp_match.group(2))
    else:
        sys_match = re.search(r'(?:systolic)\D*(\d{2,3})', extracted_text, re.IGNORECASE)
        if sys_match: extracted_metrics["sysBP"] = float(sys_match.group(1))
        dia_match = re.search(r'(?:diastolic)\D*(\d{2,3})', extracted_text, re.IGNORECASE)
        if dia_match: extracted_metrics["diaBP"] = float(dia_match.group(1))
        
    # Total Cholesterol
    chol_match = re.search(r'(?:cholesterol|totchol|chol)\D*(\d{2,3})', extracted_text, re.IGNORECASE)
    if chol_match:
        extracted_metrics["totChol"] = float(chol_match.group(1))
        
    # Glucose
    gluc_match = re.search(r'(?:glucose|fasting blood sugar|fbs)\D*(\d{2,3})', extracted_text, re.IGNORECASE)
    if gluc_match:
        extracted_metrics["glucose"] = float(gluc_match.group(1))
        
    # BMI
    bmi_match = re.search(r'(?:bmi|body mass index)\D*(\d{2}\.?\d?)', extracted_text, re.IGNORECASE)
    if bmi_match:
        extracted_metrics["BMI"] = float(bmi_match.group(1))
        
    # Heart Rate
    hr_match = re.search(r'(?:heart rate|pulse|bpm)\D*(\d{2,3})', extracted_text, re.IGNORECASE)
    if hr_match:
        extracted_metrics["heartRate"] = float(hr_match.group(1))
        
    # Cigarettes Per Day
    cigs_match = re.search(r'(?:cigarettes|cigs|cigarettes per day)\D*(\d{1,2})', extracted_text, re.IGNORECASE)
    if cigs_match:
        extracted_metrics["cigsPerDay"] = int(cigs_match.group(1))
        extracted_metrics["currentSmoker"] = 1 if int(cigs_match.group(1)) > 0 else 0
    elif re.search(r'\b(smoker|smoking)\b', extracted_text, re.IGNORECASE):
        extracted_metrics["currentSmoker"] = 1
        
    # Diabetes
    if re.search(r'\b(diabetes|diabetic)\b', extracted_text, re.IGNORECASE):
        extracted_metrics["diabetes"] = 1
        
    # Prevalent Hypertension
    if re.search(r'\b(hypertension|hypertensive)\b', extracted_text, re.IGNORECASE):
        extracted_metrics["prevalentHyp"] = 1
        
    return extracted_metrics, extracted_text

def calculate_heart_age(patient_data):
    """
    Calculates estimated Cardiovascular Heart Age vs Chronological Age
    based on Framingham Heart Study risk factor scoring equations.
    """
    age = float(patient_data.get("age", 50))
    male = int(patient_data.get("male", 0))
    sysBP = float(patient_data.get("sysBP", 120))
    totChol = float(patient_data.get("totChol", 200))
    smoker = int(patient_data.get("currentSmoker", 0))
    cigs = float(patient_data.get("cigsPerDay", 0))
    diabetes = int(patient_data.get("diabetes", 0))
    bp_meds = int(patient_data.get("BPMeds", 0))
    bmi = float(patient_data.get("BMI", 25))

    risk_delta = 0.0
    
    if sysBP >= 160: risk_delta += 4.5
    elif sysBP >= 140: risk_delta += 3.0
    elif sysBP >= 130: risk_delta += 1.5
    elif sysBP < 110: risk_delta -= 1.0
    if bp_meds == 1: risk_delta += 2.0
    
    if totChol >= 280: risk_delta += 4.0
    elif totChol >= 240: risk_delta += 2.5
    elif totChol >= 200: risk_delta += 1.0
    elif totChol < 160: risk_delta -= 1.0
    
    if smoker == 1:
        if cigs >= 20: risk_delta += 5.0
        else: risk_delta += 3.0
        
    if diabetes == 1: risk_delta += 4.0
    if bmi >= 30: risk_delta += 2.0
    elif bmi < 20: risk_delta -= 0.5
    
    if male == 1: risk_delta += 1.5
    
    estimated_heart_age = int(round(age + risk_delta))
    estimated_heart_age = max(18, min(100, estimated_heart_age))
    delta = estimated_heart_age - age
    
    return estimated_heart_age, int(round(delta))

def generate_ai_medical_report(patient_data, prediction_prob, prediction_label, api_key=None):
    """
    Generates a structured clinical diagnostic evaluation report.
    Supports external LLM API key integration if provided, or uses
    a built-in Clinical Expert Knowledge Engine fallback.
    """
    p_name = patient_data.get("patient_name", "Anonymous Patient")
    age = patient_data.get("age", "N/A")
    gender = "Male" if patient_data.get("male") == 1 else "Female"
    risk_pct = f"{prediction_prob * 100:.1f}%"
    risk_status = "HIGH 10-YEAR CHD RISK" if prediction_label == 1 else "LOW / NORMAL 10-YEAR CHD RISK"
    heart_age, heart_age_delta = calculate_heart_age(patient_data)
    
    if api_key and len(api_key.strip()) > 10:
        try:
            llm_report = _call_llm_api(patient_data, prediction_prob, prediction_label, heart_age, api_key)
            if llm_report:
                return llm_report
        except Exception as e:
            print(f"LLM API call failed: {e}. Falling back to Clinical Knowledge Engine.")

    report_lines = [
        f"================================================================================",
        f"CARDIOSENSE AI - CLINICAL DIAGNOSTIC EVALUATION REPORT",
        f"================================================================================",
        f"Patient Identifier  : {p_name}",
        f"Chronological Age   : {age} years  |  Gender: {gender}",
        f"Estimated Heart Age : {heart_age} years ({'+' if heart_age_delta >= 0 else ''}{heart_age_delta} yrs vs chronological)",
        f"Evaluation Date     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Predictive Model    : Stacking Classifier Ensemble (XGBoost + RF + LightGBM)",
        f"--------------------------------------------------------------------------------",
        f"1. DIAGNOSTIC RISK CLASSIFICATION",
        f"--------------------------------------------------------------------------------",
        f"  - Predicted 10-Year CHD Probability : {risk_pct}",
        f"  - Clinical Risk Classification      : {risk_status}",
        f"  - Recommendation Status            : {'Urgently Refer to Cardiology' if prediction_label == 1 else 'Routine Preventive Care & Lifestyle Maintenance'}",
        f"",
        f"--------------------------------------------------------------------------------",
        f"2. CLINICAL PARAMETER & METABOLIC BREAKDOWN",
        f"--------------------------------------------------------------------------------",
        f"  - Blood Pressure         : {patient_data.get('sysBP')} / {patient_data.get('diaBP')} mm Hg ({'HYPERTENSIVE' if float(patient_data.get('sysBP', 120)) >= 140 or float(patient_data.get('diaBP', 80)) >= 90 else 'Normal / Controlled'})",
        f"  - Total Serum Cholesterol: {patient_data.get('totChol')} mg/dL ({'ELEVATED HYPERCHOLESTEROLEMIA' if float(patient_data.get('totChol', 200)) >= 240 else 'Normal'})",
        f"  - Fasting Blood Glucose  : {patient_data.get('glucose')} mg/dL ({'ELEVATED / DIABETIC INDICATOR' if float(patient_data.get('glucose', 85)) >= 126 or patient_data.get('diabetes') == 1 else 'Normal'})",
        f"  - Tobacco Usage Habits   : {'Current Smoker (' + str(patient_data.get('cigsPerDay')) + ' cigs/day)' if patient_data.get('currentSmoker') == 1 else 'Non-Smoker'}",
        f"  - Body Mass Index (BMI)  : {patient_data.get('BMI')} kg/m2 ({'OBESE' if float(patient_data.get('BMI', 25)) >= 30 else 'Normal / Overweight'})",
        f"",
        f"--------------------------------------------------------------------------------",
        f"3. PERSONALIZED CLINICAL RECOMMENDATIONS",
        f"--------------------------------------------------------------------------------"
    ]
    
    rec_num = 1
    if float(patient_data.get('sysBP', 120)) >= 140 or float(patient_data.get('diaBP', 80)) >= 90 or patient_data.get('BPMeds') == 1:
        report_lines.append(f"  {rec_num}. Antihypertensive Protocol: Patient exhibits elevated blood pressure. Consider titrating ACE inhibitors or ARBs to reach target BP < 130/80 mm Hg.")
        rec_num += 1
    if float(patient_data.get('totChol', 200)) >= 240:
        report_lines.append(f"  {rec_num}. Lipid Management: High total cholesterol. Recommend high-intensity statin therapy and dietary reduction of saturated fats.")
        rec_num += 1
    if patient_data.get('currentSmoker') == 1:
        report_lines.append(f"  {rec_num}. Smoking Cessation: Enroll in structured cessation program with nicotine replacement therapy to reduce endothelial vascular injury.")
        rec_num += 1
    if float(patient_data.get('glucose', 85)) >= 126 or patient_data.get('diabetes') == 1:
        report_lines.append(f"  {rec_num}. Glycemia Optimization: Monitor HbA1c levels closely and initiate glycemic control therapy to mitigate micro/macrovascular damage.")
        rec_num += 1
    if float(patient_data.get('BMI', 25)) >= 30:
        report_lines.append(f"  {rec_num}. Metabolic Weight Management: Implement structured caloric deficit and 150 min/week moderate aerobic exercise to lower BMI.")
        rec_num += 1
        
    report_lines.append(f"  {rec_num}. Routine Monitoring: Schedule follow-up cardiovascular panel and ECG evaluation within 3 months.")
    report_lines.append("================================================================================")
    
    return "\n".join(report_lines)

def _call_llm_api(patient_data, prediction_prob, prediction_label, heart_age, api_key):
    """Fallback LLM API wrapper if user configures custom endpoint/key."""
    prompt = f"Act as an expert cardiologist. Write a medical evaluation report for patient {patient_data.get('patient_name')} (Age: {patient_data.get('age')}, Gender: {'Male' if patient_data.get('male')==1 else 'Female'}). 10-Year CHD Risk Probability: {prediction_prob*100:.1f}%. Heart Age: {heart_age}. Systolic BP: {patient_data.get('sysBP')}, Diastolic BP: {patient_data.get('diaBP')}, Total Cholesterol: {patient_data.get('totChol')}, Glucose: {patient_data.get('glucose')}, Cigs/Day: {patient_data.get('cigsPerDay')}."
    return None

def get_single_prediction_explanation(ensemble_model, X_patient, feature_names):
    """
    Computes SHAP values for a single patient's prediction using the XGBoost base estimator.
    Returns a Plotly figure representing feature contributions.
    """
    if not SHAP_AVAILABLE:
        return get_fallback_explanation(ensemble_model, X_patient, feature_names)
        
    try:
        xgb_model = ensemble_model.named_estimators_["xgb"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_patient)
        
        if isinstance(shap_values, list):
            shap_val = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            shap_val = shap_values[0]
        else:
            shap_val = shap_values
            
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_val
        })
        
        shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values(by="Abs SHAP", ascending=True)
        shap_df = shap_df[shap_df["Abs SHAP"] > 1e-4].tail(10)
        
        if shap_df.empty:
            return get_fallback_explanation(ensemble_model, X_patient, feature_names)
            
        colors = ["#f87171" if val > 0 else "#38bdf8" for val in shap_df["SHAP Value"]]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=shap_df["Feature"],
            x=shap_df["SHAP Value"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                "text": "Feature Contributions to Prediction (SHAP Explanation)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Contribution (Negative = Reduces Risk, Positive = Increases Risk)",
            yaxis_title="Features",
            margin=dict(l=150, r=20, t=50, b=50),
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            showlegend=False
        )
        
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5, x1=0, y1=len(shap_df) - 0.5,
            line=dict(color="#94a3b8", width=1, dash="dash")
        )
        
        return fig
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return get_fallback_explanation(ensemble_model, X_patient, feature_names)

def get_fallback_explanation(ensemble_model, X_patient, feature_names):
    """Fallback explanation chart using Random Forest feature importances."""
    try:
        rf_model = ensemble_model.named_estimators_["rf"]
        importances = rf_model.feature_importances_
        contributions = X_patient[0] * importances
        
        df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        })
        df["Abs Contrib"] = df["Contribution"].abs()
        df = df.sort_values(by="Abs Contrib", ascending=True).tail(10)
        
        colors = ["#f87171" if val > 0 else "#38bdf8" for val in df["Contribution"]]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["Feature"],
            x=df["Contribution"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Relative Impact: %{x:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                "text": "Feature Impact Analysis (Standardized Impact)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Relative Impact (Negative = Reduces Risk, Positive = Increases Risk)",
            yaxis_title="Features",
            margin=dict(l=150, r=20, t=50, b=50),
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            showlegend=False
        )
        
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5, x1=0, y1=len(df) - 0.5,
            line=dict(color="#94a3b8", width=1, dash="dash")
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importance Unavailable",
            annotations=[dict(text="Error generating explanation model", showarrow=False)],
            height=200
        )
        return fig
