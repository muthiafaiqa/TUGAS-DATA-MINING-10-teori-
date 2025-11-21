import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# --- 2. Custom CSS (Tema Merah/Darah) ---
st.markdown("""
    <style>
    /* Background halaman agak kemerahan */
    .stApp {
        background-color: #fff0f0;
    }
    
    /* Judul Utama */
    h1 {
        color: #8a0000;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Sub-header */
    h2, h3 {
        color: #b30000;
    }

    /* Tombol Prediksi (Merah Darah) */
    div.stButton > button:first-child {
        background-color: #d90429;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #8a0000;
        padding: 10px 24px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ef233c;
        border-color: #d90429;
    }

    /* Highlight Metric di Sidebar */
    [data-testid="stMetricValue"] {
        color: #d90429;
        font-weight: bold;
    }
    
    /* Card Style untuk deskripsi */
    .info-box {
        background-color: #ffffff;
        padding: 20px;
        border-left: 5px solid #d90429;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
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
        st.error("⚠️ File model tidak ditemukan! Pastikan scaler.pkl, model_svm.pkl, dan model_rf.pkl ada di satu folder.")
        return None, None, None

scaler, model_svm, model_rf = load_models()

if not scaler:
    st.stop()

# --- 4. Sidebar (Navigasi & Akurasi) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2503/2503506.png", width=100)
st.sidebar.title("⚙️ Panel Kontrol")
model_choice = st.sidebar.radio("Pilih Model AI:", ["Random Forest (Terbaik)", "Support Vector Machine"])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performa Model")

if model_choice == "Support Vector Machine":
    selected_model = model_svm
    acc_text = "92.20%"
    st.sidebar.metric("Akurasi SVM", acc_text, "High Precision")
    st.sidebar.info("Model ini menggunakan garis hyper-plane untuk memisahkan data pasien sakit dan sehat.")
else:
    selected_model = model_rf
    acc_text = "100%"
    st.sidebar.metric("Akurasi Random Forest", acc_text, "Perfect Score")
    st.sidebar.success("Model ini menggunakan ratusan pohon keputusan (decision trees) untuk hasil paling akurat.")

# --- 5. Judul & Penjelasan Proyek ---
st.title("🫀 Sistem Deteksi Dini Penyakit Jantung")

st.markdown("""
<div class="info-box">
    <h4>Tentang Proyek Ini</h4>
    <p>Selamat datang di aplikasi <strong>Heart Disease Prediction System</strong>. 
    Aplikasi ini dikembangkan untuk membantu tenaga medis dan pengguna umum dalam mendeteksi risiko penyakit jantung lebih awal.</p>
    <p><strong>Cara Kerja:</strong> Anda cukup memasukkan data klinis di bawah ini (seperti umur, tekanan darah, kolesterol). 
    Sistem <em>Machine Learning</em> kami akan menganalisis pola data tersebut dan memberikan prediksi probabilitas risiko penyakit jantung.</p>
</div>
""", unsafe_allow_html=True)

# --- 6. Input Form (User Friendly) ---
st.subheader("📝 Masukkan Data Klinis Pasien")
st.markdown("_Data input sudah diisi dengan **nilai contoh (default)** agar Anda mendapat gambaran format yang benar._")

# Membagi form menjadi 3 bagian logis
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Data Diri")
    age = st.number_input("Umur (Tahun)", min_value=1, max_value=100, value=55, help="Contoh: 55 tahun")
    sex = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    
    st.markdown("### 🩸 Tanda Vital")
    trestbps = st.number_input("Tekanan Darah (mmHg)", min_value=80, max_value=200, value=120, help="Tekanan darah istirahat. Normalnya sekitar 120.")
    chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Kadar kolesterol serum. Normal < 200 mg/dl.")
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=[0, 1], format_func=lambda x: "Ya (Tinggi)" if x == 1 else "Tidak (Normal)", help="Apakah gula darah puasa pasien di atas 120?")

with col2:
    st.markdown("### 💓 Gejala Jantung")
    cp = st.selectbox("Tipe Nyeri Dada (Chest Pain)", options=[0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                      help="0: Nyeri dada tipikal, 1: Nyeri atipikal, 2: Bukan nyeri angina, 3: Tanpa gejala nyeri")
    thalach = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=150, help="Detak jantung tertinggi yang dicapai saat tes.")
    exang = st.selectbox("Nyeri Saat Olahraga?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", help="Apakah timbul nyeri dada saat beraktivitas fisik?")

with col3:
    st.markdown("### 🔬 Hasil EKG & Lainnya")
    restecg = st.selectbox("Hasil EKG Istirahat", options=[0, 1, 2], 
                           format_func=lambda x: ["Normal", "Ada kelainan gelombang ST-T", "Hipertrofi Ventrikel Kiri"][x])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.")
    slope = st.selectbox("Slope ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping (Naik)", "Flat (Datar)", "Downsloping (Turun)"][x])
    ca = st.selectbox("Jml Pembuluh Utama (0-4)", options=[0, 1, 2, 3, 4], help="Jumlah pembuluh darah utama yang diwarnai dengan fluoroskopi.")
    thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}.get(x, "Unknown"))

# --- 7. Logika Prediksi ---
st.markdown("<br>", unsafe_allow_html=True) # Spasi

# Tombol di tengah (menggunakan columns trick)
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    predict_btn = st.button("🔍 ANALISIS RISIKO JANTUNG", use_container_width=True)

if predict_btn:
    # Mapping input ke DataFrame
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    try:
        # Scaling
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("📋 Hasil Diagnosa AI")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if prediction == 1:
                # Tampilan Merah Menyala untuk Positif
                st.markdown("""
                <div style="background-color: #d90429; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white; margin:0;">⚠️ POSITIF</h2>
                    <p style="font-size: 18px;">Terindikasi Penyakit Jantung</p>
                </div>
                """, unsafe_allow_html=True)
                risk_percent = probability[1]
            else:
                # Tampilan Hijau untuk Negatif
                st.markdown("""
                <div style="background-color: #2b9348; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white; margin:0;">✅ NEGATIF</h2>
                    <p style="font-size: 18px;">Jantung Sehat</p>
                </div>
                """, unsafe_allow_html=True)
                risk_percent = probability[1]

        with col_res2:
            st.write("### Tingkat Keyakinan Model:")
            
            # Custom Progress Bar Logic
            st.progress(risk_percent, text=f"Probabilitas Sakit Jantung: {risk_percent*100:.2f}%")
            
            if risk_percent > 0.5:
                st.warning("⚠️ **Saran:** Segera konsultasikan hasil ini dengan dokter spesialis jantung untuk pemeriksaan lebih lanjut.")
            else:
                st.success("🎉 **Saran:** Tetap jaga pola hidup sehat, olahraga teratur, dan makan makanan bergizi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
