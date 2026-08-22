# CardioSense AI 🏥

### *A Production-Grade Clinical Decision Support System for 10-Year Heart Disease Risk Prediction*
*(Final Year Capstone & Portfolio-Grade Data Science Project)*

CardioSense AI is a state-of-the-art Machine Learning web application engineered to support cardiologists and medical researchers in evaluating 10-year risk of **Coronary Heart Disease (CHD)**. Powered by the **Framingham Heart Study dataset (4,238 patient cohort records)**, **Streamlit**, **Scikit-Learn**, **XGBoost**, **LightGBM**, and **Plotly**, the application integrates a **Stacking Classifier Ensemble**, **AI Clinical Consultation Engine**, **Heart Age Estimator**, **Batch CSV Patient Screening**, and **Explainable AI (SHAP)**.

---

## 🌟 Key Capstone Features

1. **🏥 Diagnostic Risk Predictor & AI Clinical Consultation**:
   - Comprehensive intake form covering Patient Demographics, Lifestyle Behaviors, Medical Conditions, Vitals, and Blood Glucose.
   - Real-time 10-year CHD risk probability gauge and diagnostic risk classification (Low vs. High Risk).
   - **❤️ Heart Age Estimator**: Calculates estimated cardiovascular Heart Age vs. actual chronological age ($+X$ years older/younger).
   - **🤖 AI Medical Evaluation Report Generator**: Generates structured clinical evaluation reports with individualized prescriptions (supports OpenAI/LLM API key integration + built-in Clinical Knowledge Engine fallback). One-click `.txt` report download.
2. **📁 Batch Patient CSV Screening**:
   - Drag-and-drop CSV file uploader for bulk patient screening.
   - Automated batch prediction, high-risk ratio KPI tiles, probability distribution charts, and exportable bulk prediction CSV.
3. **🔬 Explainable AI (XAI)**:
   - Live feature contribution charts powered by **SHAP (SHapley Additive exPlanations)**.
   - Transparently highlights which clinical metrics increase or decrease a patient's risk.
4. **📊 Clinical Data Explorer (EDA)**:
   - Interactive Plotly analytics across 4 tabs: Blood Pressure & Cholesterol, Smoking & Diabetes Impact, Age & Metabolic Profile, and Correlation Matrix.
5. **🔬 Base Models vs. Stacking Benchmark Comparator**:
   - Performance comparison table (**XGBoost**, **Random Forest**, **LightGBM**, **Support Vector Classifier**, **Logistic Regression** vs. **Stacking Ensemble**).
   - Interactive Model Switcher: Test patient inputs against individual base estimators vs. the Stacking Ensemble.
6. **📋 Patient Records & Database Manager**:
   - Search patient records by name/ID and filter by risk level.
   - Dual-mode architecture: Automatic zero-config local **SQLite** (`patient_history.db`) with automatic schema migration and **Supabase** cloud sync.

---

## ⚡ Stacking Ensemble Architecture

The core predictive engine is a **Stacking Classifier** that blends probability outputs from diverse base estimators using a Logistic Regression meta-learner:

- **Base Estimator 1 (XGBoost)**: Gradient boosted decision trees for non-linear feature interactions.
- **Base Estimator 2 (Random Forest)**: Bagged decision trees for robust variance reduction.
- **Base Estimator 3 (LightGBM / SVC)**: High-speed gradient boosting or margin-maximizing decision boundaries.
- **Final Meta-Learner (Logistic Regression)**: Optimal meta-weighted probability assignment.

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
3. If no `.env` file is configured, the application automatically creates and uses a local SQLite database (`patient_history.db`).

### 3. Model Training
Train the ensemble model pipeline on the Framingham dataset:
```bash
python src/train.py
```

To run optional Optuna hyperparameter tuning during training:
```bash
python src/train.py --tune
```

### 4. Running the Web Application
Launch the Streamlit clinical dashboard:
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
│   ├── framingham.csv             # Framingham Heart Study dataset (4,238 records, 16 features)
│   └── sample_medical_report.txt  # Sample medical report template
├── models/
│   └── heart_disease_classifier.joblib  # Serialized Stacking Ensemble pipeline & scaler
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Custom sklearn transformer for Framingham preprocessing & feature engineering
│   ├── train.py                  # Model training, SMOTE balancing, and tuning pipeline
│   ├── database.py               # Supabase + SQLite database manager (with auto-migration)
│   └── explainers.py             # SHAP feature explainer, Heart Age calculator & AI Medical Report generator
│
├── app.py                        # Streamlit web app frontend (5 Executive Modules)
├── requirements.txt              # Project dependencies
├── .env.template                 # Template config for Supabase
└── README.md                     # Documentation
```
