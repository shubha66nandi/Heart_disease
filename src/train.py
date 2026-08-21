import os
import argparse
import datetime
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Fallback imports
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Class balancing will fall back to class weighting.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. Falling back to Scikit-Learn classifiers.")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: lightgbm not installed. Falling back to Scikit-Learn classifiers.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Tuning will be skipped.")

try:
    from data_processor import HeartDiseaseDataProcessor
except ImportError:
    from src.data_processor import HeartDiseaseDataProcessor

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(DATA_DIR, "heart_cleveland.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_classifier.joblib")

def download_dataset():
    """Downloads the Cleveland dataset from UCI repository and caches it locally."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {UCI_URL}...")
        try:
            urllib.request.urlretrieve(UCI_URL, DATA_PATH)
            print(f"Dataset cached successfully at {DATA_PATH}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Attempting to load synthetic backup data for offline testing...")
            return get_synthetic_data()
    else:
        print(f"Using cached dataset at {DATA_PATH}")
        
    try:
        column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(DATA_PATH, names=column_names, na_values="?")
        # Binary target: 0 = no disease, 1 = disease (target > 0 represents disease presence)
        df["target"] = (df["target"] > 0).astype(int)
        return df
    except Exception as e:
        print(f"Error reading dataset: {e}. Falling back to synthetic data.")
        return get_synthetic_data()

def get_synthetic_data():
    """Generates synthetic heart disease data if network is unavailable."""
    np.random.seed(42)
    n_samples = 300
    
    # Generate realistic ranges matching Cleveland dataset
    age = np.random.normal(54, 9, n_samples).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n_samples)
    cp = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.15, 0.25, 0.5])
    trestbps = np.random.normal(131, 17, n_samples).clip(94, 200).astype(int)
    chol = np.random.normal(246, 50, n_samples).clip(126, 564).astype(int)
    fbs = np.random.binomial(1, 0.15, n_samples)
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.1, 0.4])
    thalach = np.random.normal(149, 22, n_samples).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.32, n_samples)
    oldpeak = np.random.exponential(1.0, n_samples).clip(0.0, 6.2).round(1)
    slope = np.random.choice([1, 2, 3], n_samples, p=[0.45, 0.45, 0.1])
    ca = np.random.choice([0.0, 1.0, 2.0, 3.0], n_samples, p=[0.6, 0.2, 0.12, 0.08])
    thal = np.random.choice([3.0, 6.0, 7.0], n_samples, p=[0.55, 0.05, 0.4])
    
    # Simple probability model for disease target
    score = (
        (age - 54) / 9.0 * 0.2 +
        (sex - 0.5) * 0.3 +
        (cp - 2.5) * 0.4 +
        (trestbps - 130) / 17.0 * 0.2 +
        (chol - 240) / 50.0 * 0.1 -
        (thalach - 150) / 22.0 * 0.3 +
        (exang - 0.3) * 0.4 +
        (oldpeak - 1.0) * 0.5 +
        (ca - 0.7) * 0.6 +
        (thal - 4.5) * 0.5
    )
    median_score = np.median(score)
    target = (score > median_score).astype(int)
    
    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df

def run_optuna_tuning(X_train, y_train):
    """Optional tuning of models if requested by user."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Skipping tuning phase.")
        return get_default_tuned_params()
        
    print("=" * 60)
    print("RUNNING OPTUNA HYPERPARAMETER TUNING...")
    print("=" * 60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. XGBoost Tuning if available
    best_xgb_params = {}
    if XGB_AVAILABLE:
        print("Tuning XGBoost...")
        def xgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "eval_metric": "logloss",
                "random_state": 42
            }
            model = xgb.XGBClassifier(**params)
            return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
            
        xgb_study = optuna.create_study(direction="maximize")
        xgb_study.optimize(xgb_objective, n_trials=15)
        best_xgb_params = xgb_study.best_params
        best_xgb_params.update({"eval_metric": "logloss", "random_state": 42})
        print(f"Best XGBoost AUC: {xgb_study.best_value:.4f}")
    
    # 2. Random Forest Tuning
    print("Tuning Random Forest...")
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        
    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_objective, n_trials=15)
    best_rf_params = rf_study.best_params
    best_rf_params.update({"random_state": 42})
    print(f"Best RF AUC: {rf_study.best_value:.4f}")
    
    return best_xgb_params, best_rf_params

def get_default_tuned_params():
    """Default optimized hyperparameters."""
    xgb_params = {
        "n_estimators": 312,
        "max_depth": 7,
        "learning_rate": 0.0195,
        "subsample": 0.90,
        "colsample_bytree": 0.74,
        "reg_alpha": 0.23,
        "reg_lambda": 0.0013,
        "min_child_weight": 7,
        "eval_metric": "logloss",
        "random_state": 42
    }
    rf_params = {
        "n_estimators": 250,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42
    }
    return xgb_params, rf_params

def build_stacking_classifier(xgb_params, rf_params):
    """Creates a Stacking Classifier, falling back gracefully if estimators are missing."""
    base_estimators = []
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        base_estimators.append(("xgb", xgb.XGBClassifier(**xgb_params)))
    else:
        # Fallback to Gradient Boosting
        base_estimators.append(("gb", GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)))
        
    # Add Random Forest (always available)
    base_estimators.append(("rf", RandomForestClassifier(**rf_params)))
    
    # Add LightGBM if available
    if LGBM_AVAILABLE:
        base_estimators.append(("lgbm", lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)))
    else:
        # Fallback to Support Vector Classifier
        base_estimators.append(("svc", SVC(probability=True, C=1.0, kernel="rbf", random_state=42)))
        
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    print(f"Constructing Stacking Ensemble with estimators: {[name for name, _ in base_estimators]}")
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )
    return stacking_clf

def train_model(tune_hyperparams=False):
    # 1. Download & load data
    df = download_dataset()
    
    # 2. Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    
    X_train_raw = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test_raw = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    
    # 3. Fit Data Processor
    print("Fitting data processor and engineering features...")
    processor = HeartDiseaseDataProcessor()
    X_train_processed = processor.fit_transform(X_train_raw)
    X_test_processed = processor.transform(X_test_raw)
    
    # 4. Handle Imbalance
    if SMOTE_AVAILABLE:
        print(f"Applying SMOTE... Original target counts: {np.bincount(y_train)}")
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_processed, y_train)
        print(f"Balanced target counts: {np.bincount(y_train_bal)}")
    else:
        print("SMOTE unavailable. Proceeding with standard training set...")
        X_train_bal, y_train_bal = X_train_processed, y_train
        
    # 5. Obtain parameters
    xgb_params, rf_params = get_default_tuned_params()
    if tune_hyperparams and OPTUNA_AVAILABLE:
        xgb_params, rf_params = run_optuna_tuning(X_train_bal, y_train_bal)
        
    # 6. Fit Stacking Ensemble
    print("Training Stacking Ensemble Classifier...")
    ensemble = build_stacking_classifier(xgb_params, rf_params)
    ensemble.fit(X_train_bal, y_train_bal)
    
    # 7. Evaluate
    y_pred = ensemble.predict(X_test_processed)
    y_prob = ensemble.predict_proba(X_test_processed)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Save Pipeline
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Saving trained pipeline to {MODEL_PATH}...")
    pipeline_data = {
        "model": ensemble,
        "processor": processor,
        "metrics": {
            "accuracy": acc,
            "auc": auc,
            "f1": f1
        },
        "metadata": {
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_out": processor.feature_columns_,
            "xgb_params": xgb_params,
            "rf_params": rf_params
        }
    }
    joblib.dump(pipeline_data, MODEL_PATH)
    print("Pipeline saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Heart Disease Prediction Model")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()
    
    train_model(tune_hyperparams=args.tune)
