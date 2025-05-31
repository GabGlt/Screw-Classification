import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("Torque_Single_WorkpieceResult.pkl", "rb") as f1:
        torque_model = pickle.load(f1)
    with open("TorqueAngleGradientStep_Multi_WorkpieceResult.pkl", "rb") as f2:
        full_model = pickle.load(f2)
    return torque_model, full_model

torque_model, full_model = load_models()

# --- Helper: Convert user input string to pd.Series ---
def parse_input_series(time_str, value_str):
    try:
        time = list(map(float, time_str.strip().split(',')))
        values = list(map(float, value_str.strip().split(',')))
        return pd.Series(data=values, index=time)
    except:
        return pd.Series([])

# --- Page Config ---
st.set_page_config(page_title="Screw Classification", page_icon="🔩", layout="centered")
st.title("🔩 Screw Classification Inference App")

menu = ["🏠 Home", "🔧 Torque-Only Classification", "🛠️ Custom Feature Classification"]
choice = st.sidebar.radio("Choose Mode", menu)

# --- Home Page ---
if choice == "🏠 Home":
    st.subheader("Welcome!")
    st.write("This app predicts the screw's classification result using either torque alone or multiple sensor features.")
    st.info("Select an option from the sidebar to begin.")

# --- Torque-Only Mode ---
elif choice == "🔧 Torque-Only Classification":
    st.subheader("🔧 Predict Using Torque Only")

    time_input = st.text_input("Time Values (comma-separated)", "0.0,0.001,0.002,0.003")
    torque_input = st.text_input("Torque Values (comma-separated)", "0.1,0.2,0.15,0.25")

    if st.button("🔍 Predict with Torque"):
        torque_series = parse_input_series(time_input, torque_input)

        if torque_series.empty:
            st.error("⚠️ Invalid time or torque input.")
        else:
            input_df = pd.DataFrame({"torque_values": [torque_series]})
            prediction = torque_model.predict(input_df)[0]
            st.success(f"🎯 Predicted Workpiece Result: **{prediction}**")

# --- Custom Feature Mode ---
elif choice == "🛠️ Custom Feature Classification":
    st.subheader("🛠️ Predict Using Multiple Features")

    st.markdown("**✅ Select features to use:**")
    use_torque = st.checkbox("Torque", True)
    use_angle = st.checkbox("Angle", True)
    use_gradient = st.checkbox("Gradient", True)
    use_step = st.checkbox("Step", True)
    use_metadata = st.checkbox("Metadata (optional)", True)

    st.markdown("### 📈 Time Series Input")
    time_input = st.text_input("Time Values (shared)", "0.0,0.001,0.002,0.003")

    features = {}

    if use_torque:
        torque_input = st.text_input("Torque Values", "0.1,0.2,0.15,0.25")
        features["torque_values"] = parse_input_series(time_input, torque_input)

    if use_angle:
        angle_input = st.text_input("Angle Values", "2.5,5.25,6.25,7.0")
        features["angle_values"] = parse_input_series(time_input, angle_input)

    if use_gradient:
        gradient_input = st.text_input("Gradient Values", "0.01,0.02,0.03,0.04")
        features["gradient_values"] = parse_input_series(time_input, gradient_input)

    if use_step:
        step_input = st.text_input("Step Values", "0,0,1,1")
        features["step_values"] = parse_input_series(time_input, step_input)

    if use_metadata:
        st.markdown("### 🧾 Metadata Input")
        features["workpiece_location"] = st.selectbox("Workpiece Location", ["left", "middle", "right"])
        features["workpiece_usage"] = st.selectbox("Workpiece Usage", [0, 1])
        features["workpiece_result"] = st.selectbox("Workpiece Result", ["OK", "NOK"])
        features["scenario_condition"] = st.selectbox("Scenario Condition", ["normal", "abnormal"])
        features["scenario_exception"] = st.selectbox("Scenario Exception", [0, 1])

    if st.button("🔍 Predict with Selected Features"):
        # Check for invalid time series
        invalid_series = [k for k, v in features.items() if isinstance(v, pd.Series) and v.empty]
        if invalid_series:
            st.error(f"⚠️ Invalid input in: {', '.join(invalid_series)}")
        else:
            input_df = pd.DataFrame({k: [v] for k, v in features.items()})
            prediction = full_model.predict(input_df)[0]
            st.success(f"🎯 Predicted Workpiece Result: **{prediction}**")
