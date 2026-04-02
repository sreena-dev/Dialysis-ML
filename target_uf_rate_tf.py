"""
app.py — Clinical UI Prototype for Target UF Prediction
====================================================================
Run this using: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

# Import your existing pipeline functions
from preprocessing import build_inference_dataframe
from train_predict import load_saved_model, load_model_and_predict, DEFAULT_MODEL_DIR
from target_uf_rate import calculate_uf_rate

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Target UF Predictor - Internal Testing",
    page_icon="🩸",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Model Loading (Cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Loads the YDF model once and caches it in memory."""
    if not os.path.exists(DEFAULT_MODEL_DIR):
        st.error(f"Model directory '{DEFAULT_MODEL_DIR}' not found. Please run train_predict.py first.")
        st.stop()
    return load_saved_model(DEFAULT_MODEL_DIR)

model = get_model()

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
st.title("Hemodialysis Target UF Predictor")
st.markdown("**Internal Clinical Testing Prototype v1.0**")
st.markdown("---")

st.header("Patient Presentation")

# Create neat columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Weight")
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1)
    
    sex_str = st.selectbox("Sex", options=["Male", "Female"])
    sex_int = 0 if sex_str == "Male" else 1
    
    pre_weight = st.number_input("Pre-Dialysis Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
    dry_weight = st.number_input("Target Dry Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.1)
    treatment_time = st.number_input("Treatment Time (hours)", min_value=1.0, max_value=12.0, value=4.0, step=0.5)

with col2:
    st.subheader("Pre-Dialysis Vitals")
    sbp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=130, step=1)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
    hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75, step=1)

st.markdown("---")

# ---------------------------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------------------------
if st.button("Calculate Target UF", type="primary", use_container_width=True):
    try:
        # 1. Build the dataframe using your preprocessing script
        input_df = build_inference_dataframe(
            age=age, 
            sex=sex_int, 
            pre_weight=pre_weight, 
            dry_weight=dry_weight, 
            sbp=sbp, 
            dbp=dbp, 
            hr=hr
        )
        
        # 2. Run inference with clinical guardrails
        result = load_model_and_predict(model, input_df)
        
        # 3. Display Results
        st.header("Prediction Results")
        
        # Display the primary metric cleanly
        st.metric(
            label="Recommended Target UF", 
            value=f"{result['predicted_target_uf']:.2f} L",
            delta=f"{result['predicted_deviation']:.2f} L (Clinician Adjustment)" if not result['short_circuited'] else "N/A",
            delta_color="off"
        )
        
        # Show clinical messages based on the guardrail outcomes
        if result['short_circuited']:
            st.info(f"🛡️ **Guardrail Activated:** {result['clinical_message']}")
        elif result['capped']:
            st.warning(f"⚠️ **Safety Cap Applied:** {result['clinical_message']}")
        else:
            st.success(f"✅ **Normal Prediction:** {result['clinical_message']}")
            
        # Calculate and Display UF Rate
        uf_rate = calculate_uf_rate(pre_weight, dry_weight, treatment_time)

        st.metric(label="Calculated UF Rate (Current Parameters)", value=f"{uf_rate:.2f} mL/hr/kg")

        if uf_rate > 13:
            st.warning(f"⚠️ **Safety Cap Applied:** please stick the flow rate to < 13 mL/hr/kg ")

        # Display raw weight difference for clinician context
        st.caption(f"Raw Weight Difference: {result['weight_difference']:.2f} kg")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")