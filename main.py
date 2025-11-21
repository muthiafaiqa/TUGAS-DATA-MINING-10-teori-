import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="centered" # Layout 'centered' lebih fokus dan rapi daripada 'wide'
)

# --- 2. Custom CSS (Tema Merah Elegan & Tabs) ---
st.markdown("""
    <style>
    /* Background halaman */
    .stApp {
        background-color: #fff5f5;
    }
    
    /* Style Tab yang lebih jelas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px;
        color: #d90429;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #d90429;
        color: white;
    }

    /* Tombol Prediksi */
    div.stButton > button:first-child {
        background-color: #d90429;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 25px;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #b30000;
        color: white;
    }
    
    /* Judul */
    h1 { color: #8a0000; text-align: center; }
    h3 { color: #b30000; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Model & Scaler (JANGAN DIHAPUS) ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model_svm = joblib.load('model_svm.pkl')
        model_rf = joblib.load('model_rf.pkl')
        return scaler, model_svm, model_rf
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan! Pastikan scaler.pkl, model_svm.pkl, dan model_rf.pkl ada di folder yang sama.")
        return None, None, None

scaler, model_svm, model_rf = load_models()

if not scaler:
    st.stop()

# --- 4. Sidebar (Navigasi) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2503/2503506.png", width=80)
st.sidebar.title("Panel Kontrol")
model_choice = st.sidebar.radio("Pilih Model AI:", ["Random Forest (Akurasi 100%)", "SVM (Akurasi 92.2%)"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Isi data secara berurutan dari Tab 1 sampai Tab 3.")

if "SVM" in model_choice:
    selected_model = model_svm
else:
    selected_model = model_rf

# --- 5. Judul Utama ---
st.title("🫀 Deteksi Risiko Jantung")
st.markdown("Silakan masukkan data pasien pada kategori di bawah ini:")

# --- 6. Input Form dengan TABS (Biar Rapi) ---
tab1, tab2, tab3 = st.tabs(["👤 1. Profil Pasien", "🩸 2. Tanda Vital", "📈 3. Rekam Jantung"])

with tab1:
    st.header("Profil Dasar")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur (Tahun)", 1, 100, 55)
        sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    with col2:
        cp = st.selectbox("Jenis Nyeri Dada", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                          help="Keluhan nyeri dada yang dirasakan pasien")

with tab2:
    st.header("Pemeriksaan Vital & Darah")
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120)
        chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Gula Darah Puasa > 120?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with col2:
        thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
        exang = st.selectbox("Nyeri Saat Olahraga?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with tab3:
    st.header("Hasil EKG & Medis Lanjutan")
    col1, col2 = st.columns(2)
    with col1:
        restecg = st.selectbox("Hasil EKG Istirahat", [0, 1, 2], format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi"][x])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with col2:
        ca = st.selectbox("Jumlah Pembuluh Utama (CA)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}.get(x, "Unknown"))

# --- 7. Tombol & Hasil ---
st.markdown("---")
if st.button("🔍 ANALISIS SEKARANG"):
    
    # Setup data
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    try:
        # Proses
        input_scaled = scaler.transform(input_data)
        prediction = selected_model.predict(input_scaled)[0]
        prob = selected_model.predict_proba(input_scaled)[0]
        
        risk_score = prob[1]

        # Tampilan Hasil
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"### ⚠️ HASIL: POSITIF BERISIKO")
            st.write(f"Pasien memiliki probabilitas **{risk_score*100:.1f}%** menderita penyakit jantung.")
            st.progress(risk_score)
            st.warning("Saran: Segera rujuk ke dokter spesialis kardiologi.")
        else:
            st.success(f"### ✅ HASIL: NEGATIF (SEHAT)")
            st.write(f"Probabilitas penyakit jantung sangat rendah (**{risk_score*100:.1f}%**).")
            st.progress(risk_score)
            st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")
