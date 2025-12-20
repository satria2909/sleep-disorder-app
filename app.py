import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3
import uuid
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from fpdf import FPDF

# ==============================
# 1️⃣ SESSION ID (USER TEMPORARY)
# ==============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ==============================
# 2️⃣ DATABASE (SQLITE)
# ==============================
def get_conn():
    return sqlite3.connect("riwayat_temp.db", check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            nama TEXT,
            usia INTEGER,
            bmi REAL,
            kategori_bmi TEXT,
            hasil TEXT,
            prob_normal REAL,
            prob_insomnia REAL,
            prob_apnea REAL,
            last_active TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def cleanup_old_data():
    conn = get_conn()
    c = conn.cursor()
    batas = datetime.now() - timedelta(minutes=10)
    c.execute("DELETE FROM riwayat WHERE last_active < ?", (batas,))
    conn.commit()
    conn.close()

init_db()
cleanup_old_data()

# ==============================
# 3️⃣ LOAD MODEL
# ==============================
bundle = joblib.load("best_sleep_disorder_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
label_encoder = bundle["label_encoder"]
label_classes = label_encoder.classes_

# ==============================
# 4️⃣ PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Deteksi Gangguan Tidur",
    page_icon="😴",
    layout="centered"
)

# ==============================
# 5️⃣ FUNGSI BANTU
# ==============================
def hitung_bmi(tb, bb):
    return bb / ((tb / 100) ** 2)

def kategori_bmi_text(bmi):
    if bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def keterangan_tidur(label):
    if label == "Normal":
        return "Tidur Anda tergolong normal."
    elif label == "Insomnia":
        return "Terdapat indikasi insomnia."
    elif label == "Sleep Apnea":
        return "Terdapat indikasi sleep apnea."
    else:
        return "Perlu observasi lanjutan."

# ==============================
# 6️⃣ SIDEBAR
# ==============================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🧮 Prediksi Tidur", "📊 Hasil Prediksi",
     "🕓 Riwayat Prediksi", "💤 Tips Tidur Sehat", "ℹ️ Tentang"]
)

# ==============================
# 7️⃣ BERANDA
# ==============================
if menu == "🏠 Beranda":
    st.title("😴 Sistem Deteksi Gangguan Tidur")
    st.write("Berbasis **Support Vector Machine (SVM)**")

# ==============================
# 8️⃣ FORM PREDIKSI
# ==============================
elif menu == "🧮 Prediksi Tidur":
    st.title("Form Prediksi")

    nama = st.text_input("Nama")
    umur = st.number_input("Usia", 1, 100, 30)
    tinggi = st.number_input("Tinggi (cm)", 140, 220, 170)
    berat = st.number_input("Berat (kg)", 40, 200, 65)

    durasi_tidur = st.number_input("Durasi Tidur (jam)", 0.0, 24.0, 6.0)
    kualitas_tidur = st.slider("Kualitas Tidur", 1, 10, 5)
    aktivitas = st.slider("Aktivitas Fisik", 1, 100, 40)
    stres = st.slider("Tingkat Stres", 1, 10, 7)
    heart_rate = st.number_input("Detak Jantung", 40, 180, 72)
    steps = st.number_input("Langkah Harian", 0, 50000, 5000)
    sys = st.number_input("Sistolik", 80, 200, 120)
    dia = st.number_input("Diastolik", 40, 130, 80)

    if st.button("Prediksi"):
        bmi = hitung_bmi(tinggi, berat)
        kategori_bmi = kategori_bmi_text(bmi)

        X = pd.DataFrame([[umur, durasi_tidur, kualitas_tidur, aktivitas,
                           stres, heart_rate, steps, sys, dia]],
                         columns=feature_columns)

        prob = model.predict_proba(scaler.transform(X))[0]
        probs = dict(zip(label_classes, prob))

        prob_display = {}
        for k, v in probs.items():
            if pd.isna(k):
                prob_display["Normal"] = v
            else:
                prob_display[k] = v

        hasil = max(prob_display, key=prob_display.get)

        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO riwayat
            (session_id, nama, usia, bmi, kategori_bmi, hasil,
             prob_normal, prob_insomnia, prob_apnea, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.session_id,
            nama, umur, bmi, kategori_bmi, hasil,
            prob_display.get("Normal", 0)*100,
            prob_display.get("Insomnia", 0)*100,
            prob_display.get("Sleep Apnea", 0)*100,
            datetime.now()
        ))
        conn.commit()
        conn.close()

        st.success("Prediksi berhasil")

# ==============================
# 9️⃣ HASIL PREDIKSI TERAKHIR
# ==============================
elif menu == "📊 Hasil Prediksi":
    conn = get_conn()
    df = pd.read_sql("""
        SELECT * FROM riwayat
        WHERE session_id = ?
        ORDER BY last_active DESC LIMIT 1
    """, conn, params=(st.session_state.session_id,))
    conn.close()

    if df.empty:
        st.warning("Belum ada prediksi")
    else:
        row = df.iloc[0]
        st.subheader(f"Hasil: {row['hasil']}")
        st.info(keterangan_tidur(row["hasil"]))

# ==============================
# 🔟 RIWAYAT + EXPORT
# ==============================
elif menu == "🕓 Riwayat Prediksi":
    conn = get_conn()
    df = pd.read_sql("""
        SELECT nama, usia, bmi, kategori_bmi, hasil,
               prob_normal, prob_insomnia, prob_apnea, last_active
        FROM riwayat
        WHERE session_id = ?
        ORDER BY last_active DESC
    """, conn, params=(st.session_state.session_id,))
    conn.close()

    if df.empty:
        st.write("Belum ada data")
    else:
        st.dataframe(df)

        st.download_button(
            "⬇️ Unduh CSV",
            df.to_csv(index=False),
            "riwayat_prediksi.csv",
            "text/csv"
        )

        def export_pdf(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=9)
            for col in df.columns:
                pdf.cell(30, 8, col, border=1)
            pdf.ln()
            for _, row in df.iterrows():
                for item in row:
                    pdf.cell(30, 8, str(item), border=1)
                pdf.ln()
            return pdf.output(dest="S").encode("latin1")

        st.download_button(
            "⬇️ Unduh PDF",
            export_pdf(df),
            "riwayat_prediksi.pdf",
            "application/pdf"
        )

# ==============================
# 11️⃣ TIPS
# ==============================
elif menu == "💤 Tips Tidur Sehat":
    st.markdown("- Tidur teratur\n- Kurangi kafein\n- Kelola stres")

# ==============================
# 12️⃣ TENTANG
# ==============================
elif menu == "ℹ️ Tentang":
    st.markdown("""
    **Metode:** Support Vector Machine (SVM)  
    **Pengembang:** Satria Dava Riansa – Universitas Semarang
    """)
