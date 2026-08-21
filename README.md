# CardioSense AI 🏥

### *A Production-Grade Clinical Decision Support System for Heart Disease Risk Prediction*

CardioSense AI is an end-to-end Machine Learning web application designed to support cardiologists and medical professionals in evaluating heart disease risk. Built using **Streamlit**, **Scikit-Learn**, and **Plotly**, and designed to integrate with **Supabase** (with a seamless local **SQLite** fallback), the application leverages a **Stacking Classifier Ensemble** to deliver high-accuracy diagnostic suggestions alongside Explainable AI insights.

---

## 🌟 Key Features

1. **🏥 Patient Diagnostic Predictor**:
   - High-fidelity clinical input forms with range validation and description tooltips.
   - Real-time diagnostic risk assessment using a Stacking Ensemble.
   - Interactive gauge charts visualising risk probability.
   - Dynamic clinical observation summaries and guidelines.
2. **🔬 Explainable AI (XAI)**:
   - Live feature impact charts (SHAP / deviation contribution) explaining the *why* behind every patient's prediction.
   - Prevents the "black box" machine learning issue, building clinical confidence.
3. **📊 Clinical Data Explorer (EDA)**:
   - Interactive, zoomable Plotly charts analyzing the Cleveland Heart Disease dataset.
   - Features: Age vs. Max Heart Rate scatter analysis, cholesterol distributions, chest pain type analyses, and feature correlation heatmaps.
4. **🔬 Ensemble Analytics Dashboard**:
   - Full model comparison metrics showing why Stacking was chosen.
   - Stacking architecture breakdown, Confusion Matrix, and global feature importance charts.
5. **📋 Patient Records Database**:
   - Structured history log saving patient inputs, risk probability, and diagnosis.
   - Search by name/ID, filter by risk level, export logs to CSV, and delete entries.
   - **Database Hybrid Architecture**: Uses **Supabase** in production and automatically falls back to a zero-configuration local **SQLite** database (`patient_history.db`) if keys are absent.

---

## ⚡ Stacking Ensemble Architecture

The core predictive engine is a **Stacking Classifier** that blends predictions from multiple optimized base estimators to minimize prediction variance and maximize AUC-ROC:

- **Base Learner 1 (XGBoost / Gradient Boosting)**: Extracts complex non-linear decision boundaries.
- **Base Learner 2 (Random Forest)**: A robust bagging algorithm that resists overfitting.
- **Base Learner 3 (LightGBM / Support Vector Machine)**: Integrates diverse margin-maximizing boundaries.
- **Final Meta-Learner (Logistic Regression)**: Learns how to optimally weight and combine the base estimators' probability outputs.

On the validation test set, this Stacking Ensemble achieves:
* **Accuracy**: **~90.16%**
* **AUC-ROC**: **~0.9535**
* **F1-Score**: **~0.8966**

---

## 🚀 Quickstart & Setup

### 1. Installation
Install all python dependencies:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Database Setup
If you would like to connect to Supabase:
1. Rename `.env.template` to `.env`.
2. Enter your `SUPABASE_URL` and `SUPABASE_KEY` variables.
3. If no `.env` file is configured, the application will **automatically create and use a local SQLite database (`patient_history.db`)** without crashing.

### 3. Model Training
Train the ensemble model pipeline (downloads the Cleveland dataset from UCI and saves the serialized pipeline):
```bash
python src/train.py
```

### 4. Running the Web Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Streamlit will automatically host the web app locally (typically at `http://localhost:8501`).

---

## 📂 Project Structure

```
HeartDeseasePrediction model/
│
├── data/
│   └── heart_cleveland.csv       # Cached UCI Cleveland heart disease dataset
├── models/
│   └── heart_disease_classifier.joblib  # Serialized model and scaler pipeline
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Custom sklearn transformer for preprocessing
│   ├── train.py                  # Model training and tuning pipeline
│   ├── database.py               # Supabase + SQLite database manager
│   └── explainers.py             # SHAP / Contribution explainers
│
├── app.py                        # Streamlit web app frontend
├── requirements.txt              # Project dependencies
├── .env.template                 # Template config for Supabase
└── README.md                     # Professional documentation
```

---

## 🩺 Clinical Guidance Features
The predictor dynamically alerts clinicians if specific metrics exceed standard guidelines:
- **Hypertension**: Resting Blood Pressure $\ge 140$ mm Hg.
- **Hypercholesterolemia**: Serum Cholesterol $\ge 240$ mg/dL.
- **Myocardial Ischemia**: ST Depression induced by exercise $\ge 1.5$ mm.
- **Angina Indicators**: Exercise-induced angina detected.
