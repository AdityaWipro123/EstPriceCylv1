import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Paths to the saved models
model_paths = {
    "XGBoost": "./Model/xgboost_model.pkl",
    
}
#"Random Forest": "D:/CostEstv/Model/random_forest_model.pkl",
# Load models
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Extract model columns from XGBoost as a reference for all models
model_columns = list(models["XGBoost"].get_booster().feature_names)

# Metrics for models (replace with actual values from training phase)
model_metrics = {
    "XGBoost": {"R2": 0.96, "MAE": 991},  # Example values
    "Random Forest": {"R2": 0.95, "MAE": 1220},
    "Linear Regression": {"R2": 0.93, "MAE": 1213}
}

# Streamlit app UI
st.set_page_config(layout="wide")
st.title("Cylinder Cost Prediction")

# Layout setup
with st.sidebar:
    st.header("Input Features")

    # Model selection
    selected_model_name = st.selectbox("Select a Model", list(models.keys()))
    selected_model = models[selected_model_name]

    # Categorical inputs
    st.subheader("Categorical Features")
    cushioning_options = ["CC", "NC", "CH"]  # Update based on your data
    mounting_options = ["CM+RE", "TM+RE", "CM+RM", "CC+RE", "SPL"]  # Update based on your data

    cushioning = st.selectbox("Cushioning", cushioning_options)
    mounting = st.selectbox("Mounting", mounting_options)

    # Numerical inputs with sliders
    st.subheader("Numerical Features")
    pressure = st.slider("Pressure", min_value=50.0, max_value=300.0, value=200.0, step=10.0)
    bore = st.slider("Bore", min_value=50.0, max_value=500.0, value=70.0, step=1.0)
    rod_diameter = st.slider("Rod Diameter", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    stroke = st.slider("Stroke", min_value=50.0, max_value=1500.0, value=800.0, step=1.0)

# Main content area
col1, col2 = st.columns([10, 2])

with col1:
    st.subheader("Dynamic Prediction")

    # Predict dynamically when sliders change
    def make_prediction(pressure, bore, rod_diameter, stroke, cushioning, mounting):
        input_data = pd.DataFrame({
            "Pressure": [pressure],
            "Bore": [bore],
            "Rod diameter": [rod_diameter],
            "Stroke": [stroke],
            f"Cushioning_{cushioning}": [1],
            f"Mounting_{mounting}": [1]
        })

        # Ensure all columns required by the model are present
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match model
        input_data = input_data[model_columns]
        prediction = selected_model.predict(input_data)[0]
        return prediction

    prediction = make_prediction(pressure, bore, rod_diameter, stroke, cushioning, mounting)/1.2
    
    lower_bound = prediction * 0.95
    upper_bound = prediction * 1.05

    st.metric("Predicted Price", f"INR {prediction:.2f}")
    st.metric("Prediction Range", f"INR {lower_bound:.2f} - INR {upper_bound:.2f}")

with col2:
    st.subheader("Model Metrics")
    st.write(f"R2 Score: **{model_metrics[selected_model_name]['R2']:.2f}**")
    st.write(f"Mean Absolute Error (MAE): **{model_metrics[selected_model_name]['MAE']:.2f}**")

# Expandable visualization for feature impact
with st.expander("Visualize Feature Trends"):
    numeric_features = ["Pressure", "Bore", "Rod Diameter", "Stroke"]

    plt.figure(figsize=(8,2))

    for feature in numeric_features:
        feature_values = (
            np.linspace(50, 300, 50) if feature == "Pressure" else
            np.linspace(50, 500, 50) if feature == "Bore" else
            np.linspace(10, 500, 50) if feature == "Rod Diameter" else
            np.linspace(50, 1500, 50)
        )
        feature_predictions = []

        for value in feature_values:
            temp_prediction = make_prediction(
                pressure=value if feature == "Pressure" else pressure,
                bore=value if feature == "Bore" else bore,
                rod_diameter=value if feature == "Rod Diameter" else rod_diameter,
                stroke=value if feature == "Stroke" else stroke,
                cushioning=cushioning,
                mounting=mounting
            )
            feature_predictions.append(temp_prediction)

        # Smooth curve using interpolation
        smooth_values = np.linspace(feature_values.min(), feature_values.max(), 300)
        spline = make_interp_spline(feature_values, feature_predictions, k=3)
        smooth_predictions = spline(smooth_values)

        plt.plot(smooth_values, smooth_predictions, label=f"Trend of {feature}")

    plt.xlabel("Feature Value")
    plt.ylabel("Predicted Price")
    plt.title("Trends of Numeric Features on Predicted Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
