import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="centered" 
)

# --- 2. Custom CSS (Tema Merah, Tabs, & Info Box) ---
st.markdown("""
    <style>
    /* Background halaman */
    .stApp {
        background-color: #fff5f5;
    }
    
    /* Style Tab */
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
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
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
        border: 2px solid #b30000;
    }
    div.stButton > button:first-child:hover {
        background-color: #b30000;
        color: white;
        border-color: #8a0000;
    }
    
    /* Judul & Header */
    h1 { color: #8a0000; text-align: center; font-family: sans-serif; }
    h3 { color: #b30000; }
    
    /* Kotak Deskripsi Proyek */
    .info-box {
        background-color: #ffffff;
        padding: 20px;
        border-left: 6px solid #d90429;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .info-box h4 { color: #d90429; margin-top: 0; }
    .info-box p { color: #333; margin-bottom: 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Model & Scaler ---
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
st.sidebar.info("💡 **Tips:** Pilih model Random Forest untuk hasil paling akurat.")

if "SVM" in model_choice:
    selected_model = model_svm
else:
    selected_model = model_rf

# --- 5. Judul & Deskripsi Proyek ---
st.title("🫀 Sistem Deteksi Dini Penyakit Jantung")

st.markdown("""
<div class="info-box">
    <h4>Tentang Proyek Ini</h4>
    <p>Selamat datang di aplikasi <strong>Heart Disease Prediction System</strong>. 
    Aplikasi ini menggunakan algoritma <em>Machine Learning</em> canggih untuk menganalisis risiko penyakit jantung berdasarkan data medis Anda. 
    Tujuannya adalah untuk memberikan deteksi dini yang akurat dan cepat.</p>
</div>
""", unsafe_allow_html=True)

# --- 6. Input Form dengan TABS ---
tab1, tab2, tab3 = st.tabs(["👤 1. Profil Pasien", "🩸 2. Tanda Vital", "📈 3. Rekam Jantung"])

with tab1:
    st.header("Profil Dasar")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur (Tahun)", 1, 100, 55, help="Usia pasien dalam tahun")
        sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    with col2:
        cp = st.selectbox("Jenis Nyeri Dada", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                          help="Keluhan nyeri dada yang dirasakan pasien")

with tab2:
    st.header("Pemeriksaan Vital & Darah")
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 200, 120, help="Tekanan darah saat istirahat")
        chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200, help="Kadar kolesterol serum")
        fbs = st.selectbox("Gula Darah Puasa > 120?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with col2:
        thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150, help="Detak jantung tertinggi saat tes")
        exang = st.selectbox("Nyeri Saat Olahraga?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with tab3:
    st.header("Hasil EKG & Medis Lanjutan")
    col1, col2 = st.columns(2)
    with col1:
        restecg = st.selectbox("Hasil EKG Istirahat", [0, 1, 2], format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi"][x])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1, help="Depresi ST relative terhadap istirahat")
        slope = st.selectbox("Slope ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with col2:
        ca = st.selectbox("Jumlah Pembuluh Utama (CA)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}.get(x, "Unknown"))

# --- 7. Tombol & Hasil ---
st.markdown("---")

# === BAGIAN BARU: PESAN PENGINGAT ===
st.info("👉 **Silahkan isi lengkap 3 kategori di atas (Profil, Tanda Vital, & Rekam Jantung) untuk mendapatkan hasilnya.**")
# ====================================

if st.button("🔍 ANALISIS RISIKO JANTUNG"):
    
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
            st.markdown("""
            <div style="background-color: #d90429; color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h2 style="color: white; margin:0;">⚠️ HASIL: POSITIF</h2>
                <p style="font-size: 18px;">Terindikasi Penyakit Jantung</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"### Probabilitas Risiko: {risk_score*100:.1f}%")
            st.progress(risk_score)
            st.warning("⚠️ **Saran:** Segera konsultasikan hasil ini dengan dokter spesialis jantung.")
            
        else:
            st.markdown("""
            <div style="background-color: #2b9348; color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h2 style="color: white; margin:0;">✅ HASIL: NEGATIF</h2>
                <p style="font-size: 18px;">Jantung Sehat</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"### Probabilitas Risiko: {risk_score*100:.1f}%")
            st.progress(risk_score)
            st.success("🎉 **Saran:** Tetap jaga pola hidup sehat!")
            st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")
