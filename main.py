import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model & scaler
# ----------------------------
model_svm = joblib.load("model_svm.pkl")
model_rf = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction — Head-to-Head Models")
st.write("Pilih model (SVM atau Random Forest), lalu isi data pasien untuk prediksi.")

# ----------------------------
# Pilihan model
# ----------------------------
model_option = st.selectbox("Pilih Model", ["SVM", "Random Forest"])

# ----------------------------
# Form input
# ----------------------------
age = st.number_input("Age (contoh 57)", 1, 120, 57)
sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type",
                  ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

trestbps = st.number_input("Resting Blood Pressure (contoh 130)", 80, 200, 130)
chol = st.number_input("Cholesterol (contoh 246)", 100, 600, 246)

fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate (contoh 150)", 60, 220, 150)

exang = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression (contoh 1.0)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of vessels (0–4)", [0,1,2,3,4])
thal = st.selectbox("Thal", [1,2,3])

# ----------------------------
# Prediksi
# ----------------------------
if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    data_scaled = scaler.transform(data)

    if model_option == "SVM":
        pred = model_svm.predict(data_scaled)[0]
    else:
        pred = model_rf.predict(data_scaled)[0]

    if pred == 1:
        st.error("Hasil: Risiko TINGGI penyakit jantung")
    else:
        st.success("Hasil: Risiko RENDAH penyakit jantung")
