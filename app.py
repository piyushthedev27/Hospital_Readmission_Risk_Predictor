import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# from utils.predict import predict_readmission # Assuming this function exists

# --- Helper Function for Charting ---
def get_feature_contributions(input_data, prediction):
    """
    Simulates feature contributions for a bar chart.
    In a real-world scenario, this would come from a model explainability
    library like SHAP.
    """
    features = ["Age", "BMI", "Blood Pressure", "Glucose", "Previous Admissions", "Length of Stay"]
    base_contributions = np.random.rand(len(features))
    
    # Make contributions logical based on prediction
    if prediction == 1: # High Risk
        base_contributions[4] *= 2.5 # High impact from prev_admissions
        base_contributions[5] *= 2.0 # High impact from length_of_stay
    else: # Low Risk
        base_contributions[4] *= -2.0 # Negative (good) impact
        base_contributions[5] *= -2.5 # Negative (good) impact

    # Normalize to sum to 1
    contributions = base_contributions / np.sum(np.abs(base_contributions)) * 100
    
    return pd.DataFrame({'feature': features, 'contribution': contributions})


# --- Page Configuration ---
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="❤️‍🩹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🏥 Hospital Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Provide patient details to get a real-time risk assessment with visual insights.</p>", unsafe_allow_html=True)

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
    if prev_admissions > 2 or length_of_stay > 10 or age > 75:
        prediction, prob = 1, np.random.uniform(0.75, 0.95) # High risk
    else:
        prediction, prob = 0, np.random.uniform(0.80, 0.98) # Low risk
    
    st.markdown("---")
    st.subheader("🔬 Prediction Analysis")

    # Display results using metrics
    res_col1, res_col2 = st.columns(2)
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    
    if prediction == 1:
        res_col1.metric("Risk Assessment", "High Risk", "Intervention Recommended", delta_color="inverse")
        st.warning("⚠️ **High Risk Alert:** This patient has a high probability of readmission.", icon="🚨")
    else:
        res_col1.metric("Risk Assessment", "Low Risk", "Standard Care", delta_color="off")
        st.success("✅ **Low Risk:** The patient has a low probability of readmission.", icon="👍")
        st.balloons()
    
    # This logic determines the probability of being "High Risk" for the chart
    prob_high_risk = prob if prediction == 1 else (1 - prob)
    res_col2.metric("Confidence in Assessment", f"{prob*100:.1f}%")

    # --- Charts ---
    st.markdown("---")
    st.subheader("📊 Visual Insights")
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Gauge Chart for Confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Confidence Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#2a9d8f"},
                   'steps': [
                       {'range': [0, 50], 'color': "#e76f51"},
                       {'range': [50, 80], 'color': "#f4a261"},
                   ]}))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        # Donut Chart for Risk Probability
        fig_donut = go.Figure(go.Pie(
            labels=['High Risk', 'Low Risk'],
            values=[prob_high_risk, 1 - prob_high_risk],
            hole=.4,
            marker_colors=['#e76f51', '#2a9d8f']
        ))
        fig_donut.update_layout(title_text='Risk Probability', height=250, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Feature Contribution Chart
    st.markdown("<h5 style='text-align: center;'>Key Factors in Prediction</h5>", unsafe_allow_html=True)
    contributions_df = get_feature_contributions(input_data, prediction)
    contributions_df['color'] = contributions_df['contribution'].apply(lambda x: '#e76f51' if x > 0 else '#2a9d8f')
    
    fig_bar = go.Figure(go.Bar(
        x=contributions_df['contribution'],
        y=contributions_df['feature'],
        orientation='h',
        marker_color=contributions_df['color']
    ))
    fig_bar.update_layout(
        xaxis_title="Contribution to Risk (Red = Increases Risk)",
        yaxis={'categoryorder':'total ascending'},
        height=300,
        margin=dict(l=120, r=20, t=20, b=50)
    )
    st.plotly_chart(fig_bar, use_container_width=True)