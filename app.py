import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# 1️⃣ Load Model Langsung (tanpa API / tanpa Railway)
# ==============================

from services import predict_sleep
from schemas import PredictionRequest

# ==============================
# 2️⃣ Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Deteksi Gangguan Tidur",
    page_icon="😴",
    layout="wide"
)

# ==============================
# 3️⃣ Tema Modern Fresh (Teal / Cyan)
# ==============================
# Nuansa warna: teal + cyan — segar dan tenang, berbeda dari indigo portofolio.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Latar aplikasi */
.stApp {
    background: linear-gradient(180deg, #f0fdfa 0%, #f8fafc 60%, #ecfeff 100%);
}

/* Batasi lebar konten agar rapi */
.block-container {
    max-width: 1080px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar: gradasi teal */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f766e 0%, #0d9488 55%, #0891b2 100%);
}
[data-testid="stSidebar"] * {
    color: #f0fdfa !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
}

/* Judul halaman */
h1, h2, h3 {
    color: #134e4a !important;
    letter-spacing: -0.02em;
}

/* Tombol utama: gradasi teal -> cyan, membulat */
.stButton > button {
    background: linear-gradient(90deg, #0d9488, #06b6d4) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(13, 148, 136, 0.45);
}

/* Input membulat */
.stTextInput input, .stNumberInput input {
    border-radius: 12px !important;
}
.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
}

/* Kartu putih */
.card {
    background: #ffffff;
    border: 1px solid #ccfbf1;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 10px 30px rgba(13, 148, 136, 0.08);
    margin-bottom: 1.1rem;
}

/* Hero beranda */
.hero {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 60%, #0284c7 100%);
    border-radius: 24px;
    padding: 2.2rem 2.4rem;
    color: #ffffff;
    box-shadow: 0 18px 40px rgba(8, 145, 178, 0.30);
    margin-bottom: 1.4rem;
}
.hero .eyebrow {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    color: #a7f3d0;
    text-transform: uppercase;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0.4rem 0 0.6rem 0;
}
.hero p {
    color: #ecfeff;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0;
}

/* Kartu statistik kecil */
.stat {
    background: #ffffff;
    border: 1px solid #ccfbf1;
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.07);
}
.stat .num {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0d9488;
    line-height: 1.1;
}
.stat .lbl {
    font-size: 0.85rem;
    font-weight: 600;
    color: #475569;
    margin-top: 0.25rem;
}

/* Kartu hasil */
.result {
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    color: #ffffff;
    margin-bottom: 1.2rem;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.12);
}
.result .r-label {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.9;
}
.result .r-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0.3rem 0 0.5rem 0;
}
.result .r-desc {
    font-size: 0.98rem;
    line-height: 1.6;
    opacity: 0.95;
    margin: 0;
}
.result-normal { background: linear-gradient(135deg, #059669, #10b981); }
.result-insomnia { background: linear-gradient(135deg, #b45309, #f59e0b); }
.result-apnea { background: linear-gradient(135deg, #be123c, #f43f5e); }
.result-unknown { background: linear-gradient(135deg, #475569, #64748b); }

/* Bar probabilitas */
.prob-row { margin-bottom: 0.9rem; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-weight: 700;
    color: #134e4a;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}
.prob-track {
    background: #e2e8f0;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}

/* Chip kecil */
.chip {
    display: inline-block;
    background: #ccfbf1;
    color: #0f766e;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 700;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Warna hasil & probabilitas
WARNA_HASIL = {
    "Normal": "result-normal",
    "Insomnia": "result-insomnia",
    "Sleep Apnea": "result-apnea",
}
WARNA_PROB = {
    "Normal": "#10b981",
    "Insomnia": "#f59e0b",
    "Sleep Apnea": "#f43f5e",
}

# ==============================
# 4️⃣ Fungsi Bantu (logika tidak berubah)
# ==============================
def hitung_bmi(tinggi_cm, berat_kg):
    tinggi_m = tinggi_cm / 100
    return berat_kg / (tinggi_m ** 2)

def kategori_bmi_text(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def keterangan_tidur(label):
    if label == "Normal":
        return (
            "Kondisi tidur tergolong normal. Kualitas dan durasi tidur sudah cukup baik. "
            "Disarankan untuk mempertahankan pola tidur yang teratur, mengelola stres, "
            "serta menjaga aktivitas fisik agar kesehatan tetap optimal."
        )
    elif label == "Insomnia":
        return (
            "Terdapat indikasi insomnia, yaitu kesulitan untuk memulai atau mempertahankan tidur. "
            "Hal ini dapat dipengaruhi oleh tingkat stres, pola tidur yang tidak teratur, atau gaya hidup. "
            "Disarankan untuk memperbaiki rutinitas tidur, mengurangi penggunaan gadget sebelum tidur, "
            "serta melakukan relaksasi. Jika berlanjut, pertimbangkan konsultasi dengan tenaga medis."
        )
    elif label == "Sleep Apnea":
        return (
            "Terdapat indikasi sleep apnea, yaitu gangguan tidur yang ditandai dengan gangguan pernapasan saat tidur. "
            "Kondisi ini dapat berdampak pada kualitas tidur dan kesehatan jantung. "
            "Disarankan untuk segera melakukan pemeriksaan lebih lanjut ke tenaga medis, "
            "serta menjaga berat badan dan pola hidup sehat."
        )
    else:
        return (
            "Hasil tidak dapat diklasifikasikan secara jelas. "
            "Disarankan untuk melakukan observasi lebih lanjut atau berkonsultasi dengan tenaga medis."
        )

# ==============================
# 5️⃣ Sidebar Navigasi
# ==============================
st.sidebar.markdown(
    """
    <div style="padding: 0.4rem 0 1rem 0;">
        <div style="font-size: 2rem;">😴</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; line-height: 1.2;">Deteksi Tidur</div>
        <div style="font-size: 0.82rem; color: #ccfbf1;">SVM &bull; Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)
menu = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "🧮 Prediksi Tidur",
        "📊 Hasil Prediksi",
        "🕓 Riwayat Prediksi",
        "💤 Tips Tidur Sehat",
        "ℹ️ Tentang"
    ],
    key="menu"
)

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# ==============================
# 6️⃣ Beranda
# ==============================
if menu == "🏠 Beranda":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">😴 Sistem Deteksi Gangguan Tidur</div>
            <h1>Tidur Nyenyak,<br>Hidup Lebih Sehat.</h1>
            <p>Deteksi gangguan tidur berbasis <b>Support Vector Machine (SVM)</b> — isi datamu,
            dan ketahui kondisi tidurmu dalam hitungan detik.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<div class="stat"><div class="num">99,83%</div><div class="lbl">Akurasi Pengujian</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat"><div class="num">SVM</div><div class="lbl">Metode Klasifikasi</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat"><div class="num">3 Kelas</div><div class="lbl">Normal &bull; Insomnia &bull; Sleep Apnea</div></div>', unsafe_allow_html=True)
    </div>
    if st.button("🔮 Mulai Prediksi Sekarang"):
        st.session_state["menu"] = "🧮 Prediksi Tidur"
        st.rerun()

# ==============================
# 7️⃣ Form Prediksi
# ==============================
elif menu == "🧮 Prediksi Tidur":
    st.markdown("## 🧮 Formulir Prediksi Tidur")
    st.markdown("Isi data di bawah ini — model akan mengklasifikasikan kondisi tidurmu.")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👤 Data Diri")
        nama = st.text_input("Nama Lengkap")
        c1, c2 = st.columns(2)
        with c1:
            umur = st.number_input("Usia (tahun)", 1, 100, 30)
        with c2:
            st.markdown('<div style="height: 1.55rem;"></div>', unsafe_allow_html=True)
            st.markdown('<span class="chip">SVM</span><span class="chip">Streamlit</span><span class="chip">JOINS SINTA 4</span>', unsafe_allow_html=True)

        st.markdown("### ⚖️ Fisik & BMI")
        c1, c2 = st.columns(2)
        with c1:
            tinggi = st.number_input("Tinggi Badan (cm)", 140, 220, 170)
        with c2:
            berat = st.number_input("Berat Badan (kg)", 40, 200, 65)
        bmi = hitung_bmi(tinggi, berat)
        kategori_bmi = kategori_bmi_text(bmi)
        st.metric(label="Body Mass Index (BMI)", value=f"{bmi:.2f}", delta=kategori_bmi)

        st.markdown("### 🌙 Pola Tidur")
        c1, c2 = st.columns(2)
        with c1:
            durasi_tidur = st.number_input("Durasi Tidur (jam/hari)", 0.0, 24.0, 6.0)
            kualitas_tidur = st.slider("Kualitas Tidur (1–10)", 1, 10, 5)
        with c2:
            aktivitas_fisik = st.slider("Aktivitas Fisik (1–100)", 1, 100, 40)
            tingkat_stres = st.slider("Tingkat Stres (1–10)", 1, 10, 7)

        st.markdown("### ❤️ Kesehatan")
        c1, c2, c3 = st.columns(3)
        with c1:
            heart_rate = st.number_input("Detak Jantung (bpm)", 40, 180, 72)
        with c2:
            daily_steps = st.number_input("Langkah Harian", 0, 50000, 5000)
        with c3:
            systolic = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
        diastolic = st.number_input("Tekanan Darah Diastolik", 40, 130, 80)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Prediksi Sekarang"):
        if nama.strip() == "":
            st.warning("Nama wajib diisi.")
        else:
            try:
                data_input = PredictionRequest(
                    Age=umur,
                    Sleep_Duration=durasi_tidur,
                    Quality_of_Sleep=kualitas_tidur,
                    Physical_Activity_Level=aktivitas_fisik,
                    Stress_Level=tingkat_stres,
                    Heart_Rate=heart_rate,
                    Daily_Steps=daily_steps,
                    Systolic_BP=systolic,
                    Diastolic_BP=diastolic
                )
                hasil_api = predict_sleep(data_input)
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
                st.stop()

            if "prediction" not in hasil_api or "probability" not in hasil_api:
                st.error(f"Respons prediksi tidak sesuai format: {hasil_api}")
                st.stop()

            hasil = hasil_api["prediction"]
            prob_dict = hasil_api["probability"]

            st.session_state.last_pred = {
                "BMI": bmi,
                "Kategori BMI": kategori_bmi,
                "Hasil": hasil,
                "Probabilitas": prob_dict
            }

            st.session_state.riwayat.append({
                "Nama": nama,
                "Usia": umur,
                "BMI": round(bmi, 2),
                "Kategori BMI": kategori_bmi,
                "Hasil": hasil
            })

            st.success("✅ Prediksi berhasil! Lihat menu **Hasil Prediksi**.")

# ==============================
# 8️⃣ Hasil Prediksi
# ==============================
elif menu == "📊 Hasil Prediksi":
    st.markdown("## 📊 Hasil Prediksi")
    if "last_pred" not in st.session_state:
        st.info("Belum ada prediksi. Isi formulir di menu **Prediksi Tidur** terlebih dahulu.")
    else:
        data = st.session_state.last_pred
        kelas = WARNA_HASIL.get(data["Hasil"], "result-unknown")
        st.markdown(
            f"""
            <div class="result {kelas}">
                <div class="r-label">Hasil Deteksi</div>
                <div class="r-value">{data['Hasil']}</div>
                <p class="r-desc">{keterangan_tidur(data['Hasil'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="stat"><div class="num">{data["BMI"]:.2f}</div><div class="lbl">BMI — {data["Kategori BMI"]}</div></div>', unsafe_allow_html=True)
        with m2:
            top = max(data["Probabilitas"], key=lambda k: data["Probabilitas"][k])
            st.markdown(f'<div class="stat"><div class="num">{data["Probabilitas"][top]:.1f}%</div><div class="lbl">Keyakinan Tertinggi — {top}</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
        st.markdown("### Tingkat Probabilitas")
        for label, value in data["Probabilitas"].items():
            warna = WARNA_PROB.get(label, "#0d9488")
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label"><span>{label}</span><span>{value:.2f}%</span></div>
                    <div class="prob-track"><div class="prob-fill" style="width:{value}%;background:{warna};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================
# 9️⃣ Riwayat Prediksi
# ==============================
elif menu == "🕓 Riwayat Prediksi":
    st.markdown("## 🕓 Riwayat Prediksi")
    if len(st.session_state.riwayat) == 0:
        st.info("Belum ada data riwayat.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.riwayat), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.riwayat = []
            st.rerun()

# ==============================
# 🔟 Tips Tidur
# ==============================
elif menu == "💤 Tips Tidur Sehat":
    st.markdown("## 💤 Tips Tidur Sehat")
    tips = [
        ("🕒", "Konsisten Waktu Tidur", "Tidur dan bangun pada jam yang sama setiap hari membantu menjaga ritme sirkadian tubuh sehingga kualitas tidur lebih stabil."),
        ("☕", "Batasi Kafein & Gadget", "Hindari kopi, teh, dan minuman energi serta penggunaan gadget minimal 1–2 jam sebelum tidur karena dapat mengganggu produksi hormon melatonin."),
        ("😌", "Kelola Stres dengan Baik", "Lakukan relaksasi seperti meditasi, pernapasan dalam, atau membaca buku untuk menenangkan pikiran sebelum tidur."),
        ("🏃", "Rutin Beraktivitas Fisik", "Olahraga ringan hingga sedang secara rutin dapat meningkatkan kualitas tidur, namun hindari olahraga berat menjelang waktu tidur."),
        ("🛏️", "Ciptakan Lingkungan Nyaman", "Pastikan kamar tidur dalam kondisi gelap, sejuk, dan tenang agar tubuh lebih mudah untuk beristirahat."),
        ("🍽️", "Perhatikan Pola Makan", "Hindari makan berat menjelang tidur dan batasi konsumsi gula berlebih di malam hari."),
        ("📊", "Perhatikan Tanda Gangguan", "Jika sulit tidur, sering terbangun, atau merasa lelah saat bangun, pertimbangkan untuk melakukan pemeriksaan lebih lanjut."),
    ]
    k1, k2 = st.columns(2)
    for i, (ikon, judul, isi) in enumerate(tips):
        with (k1 if i % 2 == 0 else k2):
            st.markdown(
                f'<div class="card"><div style="font-size: 1.6rem;">{ikon}</div>'
                f'<h3 style="margin: 0.3rem 0;">{judul}</h3>'
                f'<p style="color: #475569; line-height: 1.6; margin: 0;">{isi}</p></div>',
                unsafe_allow_html=True,
            )

# ==============================
# 1️⃣1️⃣ Tentang
# ==============================
elif menu == "ℹ️ Tentang":
    st.markdown("## ℹ️ Tentang")
    st.markdown(
        """
        <div class="card">
            <p><b>Aplikasi:</b> Sistem Deteksi Gangguan Tidur</p>
            <p><b>Metode:</b> Support Vector Machine (SVM)</p>
            <p><b>Akurasi Pengujian:</b> 99,83%</p>
            <p><b>Pengembang:</b> Satria Dava Riansa (G.211.22.0006) – Universitas Semarang</p>
            <p><b>Publikasi:</b> JOINS (SINTA 4) — “Deteksi Gangguan Tidur Menggunakan Support Vector Machine pada Aplikasi Web Streamlit”</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
