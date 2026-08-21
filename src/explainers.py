import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def get_single_prediction_explanation(ensemble_model, X_patient, feature_names):
    """
    Computes SHAP values for a single patient's prediction using the XGBoost base estimator.
    Returns a Plotly figure representing the feature contributions.
    """
    # Fallback if SHAP is not available
    if not SHAP_AVAILABLE:
        return get_fallback_explanation(ensemble_model, X_patient, feature_names)
        
    try:
        # Extract the trained XGBoost estimator from the stacking classifier
        xgb_model = ensemble_model.named_estimators_["xgb"]
        
        # Initialize explainer and calculate SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_patient)
        
        # Handle shape differences (binary classification shap values might be a 1D array or 2D)
        if isinstance(shap_values, list):
            # For some shap versions, binary classification returns a list [shap_values_class0, shap_values_class1]
            shap_val = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            shap_val = shap_values[0]
        else:
            shap_val = shap_values
            
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
        # Create a DataFrame of feature names and shap values
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_val
        })
        
        # Sort by absolute SHAP value to get most influential features
        shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values(by="Abs SHAP", ascending=True)
        
        # Filter out features with virtually 0 contribution
        shap_df = shap_df[shap_df["Abs SHAP"] > 1e-4]
        
        # Take top 10 features for display
        shap_df = shap_df.tail(10)
        
        if shap_df.empty:
            return get_fallback_explanation(ensemble_model, X_patient, feature_names)
            
        # Create Plotly Horizontal Bar Chart
        colors = ["#E8593C" if val > 0 else "#3B8BD4" for val in shap_df["SHAP Value"]]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=shap_df["Feature"],
            x=shap_df["SHAP Value"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>"
        ))
        
        # Update layout for premium look
        fig.update_layout(
            title={
                "text": "Feature Contributions to Prediction (SHAP)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Contribution (Negative = Reduces Risk, Positive = Increases Risk)",
            yaxis_title="Features",
            margin=dict(l=150, r=20, t=50, b=50),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        # Add a zero line
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5, x1=0, y1=len(shap_df) - 0.5,
            line=dict(color="black", width=1, dash="dash")
        )
        
        return fig
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return get_fallback_explanation(ensemble_model, X_patient, feature_names)

def get_fallback_explanation(ensemble_model, X_patient, feature_names):
    """
    Alternative explanation chart if SHAP is not installed or errors out.
    Uses the coefficients or feature importances to build a mock explanation.
    """
    try:
        # Access Random Forest estimator which always has feature_importances_
        rf_model = ensemble_model.named_estimators_["rf"]
        importances = rf_model.feature_importances_
        
        # Multiply importance by patient feature sign/magnitude relative to average (which is 0 since standard scaled!)
        # Since standard scaled, X_patient values represent standard deviations from mean
        # Positive value means patient is higher than mean, negative means lower.
        contributions = X_patient[0] * importances
        
        df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        })
        df["Abs Contrib"] = df["Contribution"].abs()
        df = df.sort_values(by="Abs Contrib", ascending=True).tail(10)
        
        colors = ["#E8593C" if val > 0 else "#3B8BD4" for val in df["Contribution"]]
        
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
                "text": "Feature Impact Analysis (Standardized Deviation)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Relative Impact (Negative = Reduces Risk, Positive = Increases Risk)",
            yaxis_title="Features",
            margin=dict(l=150, r=20, t=50, b=50),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        # Add a zero line
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5, x1=0, y1=len(df) - 0.5,
            line=dict(color="black", width=1, dash="dash")
        )
        
        return fig
    except Exception as e:
        # Absolute fallback: empty figure with text annotation
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importance Unavailable",
            annotations=[dict(text="Error generating explanation model", showarrow=False)],
            height=200
        )
        return fig
