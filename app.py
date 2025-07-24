import streamlit as st
from utils.predict import predict_readmission # Assuming this function exists

# --- Page Configuration ---
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="❤️‍🩹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🏥 Hospital Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Provide patient details to get a real-time risk assessment.</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Input Form ---
with st.expander("📝 Enter Patient Vitals", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("👤 Age", 0, 100, 50, help="Patient's age in years.")
        bmi = st.slider("⚖️ BMI", 10.0, 50.0, 22.0, format="%.1f", help="Body Mass Index.")
        prev_admissions = st.number_input("⏮️ Previous Admissions", 0, 10, 1, help="Number of prior hospital admissions.")

    with col2:
        blood_pressure = st.slider("🩸 Blood Pressure (Systolic)", 80, 200, 120, help="Systolic blood pressure (mmHg).")
        glucose = st.slider("🍬 Glucose Level", 70, 200, 100, help="Fasting glucose level (mg/dL).")
        length_of_stay = st.slider("⏳ Length of Stay (days)", 1, 30, 3, help="Duration of the current hospital stay.")

# --- Prediction Button ---
st.write("") # Spacer
predict_button = st.button("✨ Predict Readmission Risk", use_container_width=True)

# --- Prediction Logic and Display ---
if predict_button:
    input_data = [age, bmi, blood_pressure, glucose, prev_admissions, length_of_stay]
    
    # Simulate a prediction call
    # In a real app, you would call: prediction, prob = predict_readmission(input_data)
    # For demonstration purposes, we'll simulate the output based on inputs.
    if prev_admissions > 2 or length_of_stay > 10:
        prediction, prob = 1, 0.85 # High risk
    else:
        prediction, prob = 0, 0.92 # Low risk
    
    st.markdown("---")
    st.subheader("🔬 Prediction Analysis")

    # Display results using metrics in columns
    col1, col2 = st.columns(2)
    
    if prediction == 1:
        with col1:
            st.metric(
                label="Risk Assessment",
                value="High Risk",
                delta="Intervention Recommended",
                delta_color="inverse"
            )
        with col2:
            st.metric(label="Prediction Confidence", value=f"{prob*100:.1f}%")
        st.warning("⚠️ **High Risk Alert:** This patient has a high probability of readmission. Consider a follow-up plan, medication review, and patient education.", icon="🚨")

    else:
        with col1:
            st.metric(
                label="Risk Assessment",
                value="Low Risk",
                delta="Standard Care Sufficient",
                delta_color="off"
            )
        with col2:
            st.metric(label="Prediction Confaidence", value=f"{prob*100:.1f}%")
        st.success("✅ **Low Risk:** The patient has a low probability of readmission. Standard discharge protocol is likely sufficient.", icon="👍")
        st.balloons()