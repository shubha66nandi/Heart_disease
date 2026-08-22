import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Mapping of raw value representations for clinical UI and explainability (Framingham Dataset)
FEATURE_METADATA = {
    "male": {"label": "Gender", "type": "categorical", "options": {1: "Male", 0: "Female"}},
    "age": {"label": "Age", "type": "numeric", "unit": "years"},
    "education": {
        "label": "Education Level",
        "type": "categorical",
        "options": {
            1: "Some High School",
            2: "High School / GED",
            3: "Some College / Vocational",
            4: "College Degree or Higher"
        }
    },
    "currentSmoker": {"label": "Current Smoker", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "cigsPerDay": {"label": "Cigarettes Per Day", "type": "numeric", "unit": "cigs/day"},
    "BPMeds": {"label": "On Blood Pressure Medication", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "prevalentStroke": {"label": "History of Stroke", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "prevalentHyp": {"label": "Prevalent Hypertension", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "diabetes": {"label": "Diabetes Mellitus", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "totChol": {"label": "Total Cholesterol", "type": "numeric", "unit": "mg/dL"},
    "sysBP": {"label": "Systolic Blood Pressure", "type": "numeric", "unit": "mm Hg"},
    "diaBP": {"label": "Diastolic Blood Pressure", "type": "numeric", "unit": "mm Hg"},
    "BMI": {"label": "Body Mass Index (BMI)", "type": "numeric", "unit": "kg/m²"},
    "heartRate": {"label": "Heart Rate", "type": "numeric", "unit": "bpm"},
    "glucose": {"label": "Fasting Blood Glucose", "type": "numeric", "unit": "mg/dL"}
}

class HeartDiseaseDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns_ = None
        self.raw_features_ = [
            "male", "age", "education", "currentSmoker", "cigsPerDay", 
            "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
            "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
        ]
        
    def _clean_raw_inputs(self, df):
        df_clean = df.copy()
        for col in self.raw_features_:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean

    def fit(self, X, y=None):
        X_copy = self._clean_raw_inputs(X)
        
        # 1. Impute missing values for numeric and categorical features
        self.imputer.fit(X_copy[self.raw_features_])
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_copy[self.raw_features_]),
            columns=self.raw_features_,
            index=X_copy.index
        )
        
        # 2. Engineer features
        X_engineered = self._engineer_features(X_imputed)
        
        # 3. One-hot encode categorical columns if multi-category (education)
        cat_cols = ["education"]
        X_encoded = pd.get_dummies(X_engineered, columns=cat_cols, drop_first=True)
        
        # Store column order after encoding to align during transform
        self.feature_columns_ = list(X_encoded.columns)
        
        # 4. Fit scaler
        self.scaler.fit(X_encoded)
        
        return self
        
    def transform(self, X):
        X_copy = self._clean_raw_inputs(X)
        
        # 1. Impute
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_copy[self.raw_features_]),
            columns=self.raw_features_,
            index=X_copy.index
        )
        
        # 2. Engineer
        X_engineered = self._engineer_features(X_imputed)
        
        # 3. One-hot encode
        cat_cols = ["education"]
        X_encoded = pd.get_dummies(X_engineered, columns=cat_cols, drop_first=True)
        
        # 4. Align columns to match what was learned during fit
        for col in self.feature_columns_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Keep only the columns present in fit, in the exact same order
        X_encoded = X_encoded[self.feature_columns_]
        
        # 5. Scale
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled

    def _engineer_features(self, df):
        df_eng = df.copy()
        
        # Ensure numeric types
        for col in ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "cigsPerDay"]:
            df_eng[col] = pd.to_numeric(df_eng[col], errors="coerce")
            
        # Clinical Blood Pressure metrics
        df_eng["sys_dia_bp_ratio"] = df_eng["sysBP"] / (df_eng["diaBP"] + 1.0)
        df_eng["pulse_pressure"]   = df_eng["sysBP"] - df_eng["diaBP"]
        df_eng["mean_arterial_bp"] = df_eng["diaBP"] + (df_eng["pulse_pressure"] / 3.0)
        
        # Metabolic & Lipid ratios
        df_eng["chol_bmi_ratio"]   = df_eng["totChol"] / (df_eng["BMI"] + 1.0)
        df_eng["pack_years_proxy"]  = df_eng["age"] * df_eng["cigsPerDay"] / 20.0
        
        # Age group bins: [0, 40, 55, 70, 120] -> [0, 1, 2, 3]
        df_eng["age_group"] = pd.cut(
            df_eng["age"], 
            bins=[0, 40, 55, 70, 120],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
        
        return df_eng
