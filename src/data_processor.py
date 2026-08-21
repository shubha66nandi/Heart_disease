import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Mapping of raw value representations for clinical UI and explainability
FEATURE_METADATA = {
    "age": {"label": "Age", "type": "numeric", "unit": "years"},
    "sex": {"label": "Gender", "type": "categorical", "options": {1: "Male", 0: "Female"}},
    "cp": {
        "label": "Chest Pain Type",
        "type": "categorical",
        "options": {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-anginal Pain",
            4: "Asymptomatic"
        }
    },
    "trestbps": {"label": "Resting Blood Pressure", "type": "numeric", "unit": "mm Hg"},
    "chol": {"label": "Serum Cholesterol", "type": "numeric", "unit": "mg/dl"},
    "fbs": {"label": "Fasting Blood Sugar > 120 mg/dl", "type": "categorical", "options": {1: "True", 0: "False"}},
    "restecg": {
        "label": "Resting Electrocardiographic Results",
        "type": "categorical",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }
    },
    "thalach": {"label": "Maximum Heart Rate Achieved", "type": "numeric", "unit": "bpm"},
    "exang": {"label": "Exercise Induced Angina", "type": "categorical", "options": {1: "Yes", 0: "No"}},
    "oldpeak": {"label": "ST Depression Induced by Exercise", "type": "numeric", "unit": "mm"},
    "slope": {
        "label": "Slope of the Peak Exercise ST Segment",
        "type": "categorical",
        "options": {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        }
    },
    "ca": {"label": "Number of Major Vessels Colored by Fluoroscopy", "type": "numeric", "unit": "vessels (0-3)"},
    "thal": {
        "label": "Thalassemia type",
        "type": "categorical",
        "options": {
            3: "Normal",
            6: "Fixed Defect",
            7: "Reversible Defect"
        }
    }
}

class HeartDiseaseDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns_ = None
        self.raw_features_ = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", 
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        
    def _clean_raw_inputs(self, df):
        df_clean = df.copy()
        for col in self.raw_features_:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean

    def fit(self, X, y=None):
        X_copy = self._clean_raw_inputs(X)
        
        # 1. Impute missing values for raw numeric and numerical-categorical features
        # For Cleveland: 'ca' and 'thal' have missing values
        self.imputer.fit(X_copy[self.raw_features_])
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_copy[self.raw_features_]),
            columns=self.raw_features_,
            index=X_copy.index
        )
        
        # 2. Engineer features
        X_engineered = self._engineer_features(X_imputed)
        
        # 3. One-hot encode categorical columns
        cat_cols = ["cp", "restecg", "slope", "thal"]
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
        cat_cols = ["cp", "restecg", "slope", "thal"]
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
        
        # Convert numeric types just in case
        for col in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]:
            df_eng[col] = pd.to_numeric(df_eng[col], errors="coerce")
            
        # Interaction features
        df_eng["age_thalach"]   = df_eng["age"] * df_eng["thalach"]          # age × max heart rate
        df_eng["bp_chol_ratio"] = df_eng["trestbps"] / (df_eng["chol"] + 1)  # blood pressure / cholesterol
        
        # Age group bins: [0, 40, 55, 70, 100] -> [0, 1, 2, 3]
        df_eng["age_group"] = pd.cut(
            df_eng["age"], 
            bins=[0, 40, 55, 70, 120],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
        
        return df_eng
