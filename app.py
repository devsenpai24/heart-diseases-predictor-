import streamlit as st
import joblib
import pandas as pd

# Load saved files
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Provide the following details to predict heart disease risk.")

age = st.slider("Age", min_value=18, max_value=77, value=30)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=94, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=126, max_value=564, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=120)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.2, value=0.0)
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

if st.button("Predict"):
    # Create raw input dataframe with proper column names
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Convert categorical columns into dummy variables
    input_data = pd.get_dummies(input_data)

    # Add missing columns from training
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 0:
        st.success("Prediction: No heart disease detected")
    else:
        st.error("Prediction: Heart disease detected")