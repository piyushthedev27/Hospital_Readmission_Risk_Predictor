import streamlit as st
from utils.predict import predict_readmission

st.set_page_config(page_title="Hospital Readmission Risk", layout="centered")

st.title("📊 Hospital Readmission Risk Predictor")
st.subheader("Enter patient details to predict readmission risk")

age = st.slider("Age", 0, 100, 50)
bmi = st.slider("BMI", 10.0, 50.0, 22.0)
blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
glucose = st.slider("Glucose Level", 70, 200, 100)
prev_admissions = st.number_input("Previous Admissions", 0, 10, 1)
length_of_stay = st.slider("Length of Stay (days)", 1, 30, 3)

if st.button("🔍 Predict Risk"):
    input_data = [age, bmi, blood_pressure, glucose, prev_admissions, length_of_stay]
    prediction, prob = predict_readmission(input_data)

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.success(f"**{'High Risk' if prediction == 1 else 'Low Risk'}** of readmission")
    st.info(f"Model Confidence: **{prob*100:.2f}%**")

    if prediction == 1:
        st.warning("⚠️ Follow-up or extra care recommended.")
    else:
        st.balloons()