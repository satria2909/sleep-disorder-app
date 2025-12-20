import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3
import uuid
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==============================
# 🔐 SESSION ID (USER TEMP)
# ==============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ==============================
# 🗄️ DATABASE SQLITE
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
    batas = datetime.now() - timedelta(minutes=10)
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM riwayat WHERE last_active < ?", (batas,))
    conn.commit()
    conn.close()

init_db()
cleanup_old_data()

# ==============================
# 1️⃣ LOAD MODEL
# ==============================
bundle = joblib.load("best_sleep_disorder_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
label_encoder = bundle["label_encoder"]
label_classes = label_encoder.classes_

# ==============================
# 2️⃣ PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Deteksi Gangguan Tidur",
    page_icon="😴",
    layout="centered"
)

# ==============================
# 3️⃣ FUNGSI BANTU (TETAP)
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
        return "Tidur Anda tergolong normal. Pertahankan pola hidup sehat."
    elif label == "Insomnia":
        return "Terdapat indikasi insomnia. Perhatikan durasi tidur, stres, dan kualitas tidur."
    elif label == "Sleep Apnea":
        return "Terdapat indikasi sleep apnea. Disarankan konsultasi medis."
    else:
        return "Hasil berada pada kondisi lain. Perlu observasi lanjutan."

# ==============================
# 4️⃣ SIDEBAR (TETAP)
# ==============================
menu = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "🧮 Prediksi Tidur",
        "📊 Hasil Prediksi",
        "🕓 Riwayat Prediksi",
        "💤 Tips Tidur Sehat",
        "ℹ️ Tentang"
    ]
)

# ==============================
# 5️⃣ BERANDA (TETAP)
# ==============================
if menu == "🏠 Beranda":
    st.title("😴 Sistem Deteksi Gangguan Tidur")
    st.write("Aplikasi deteksi gangguan tidur berbasis **Support Vector Machine (SVM)**.")
    st.info("Gunakan menu di kiri untuk melakukan prediksi.")
    st.image(
        "sleep.png",
        caption="Sistem Deteksi Gangguan Tidur Berbasis SVM",
        use_container_width=True
    )

# ==============================
# 6️⃣ FORM PREDIKSI (UI TETAP)
# ==============================
elif menu == "🧮 Prediksi Tidur":
    st.title("Formulir Prediksi Tidur")

    nama = st.text_input("Nama Lengkap")
    umur = st.number_input("Usia (tahun)", 1, 100, 30)

    tinggi = st.number_input("Tinggi Badan (cm)", 140, 220, 170)
    berat = st.number_input("Berat Badan (kg)", 40, 200, 65)

    bmi = hitung_bmi(tinggi, berat)
    kategori_bmi = kategori_bmi_text(bmi)
    st.write(f"**BMI:** {bmi:.2f} — *{kategori_bmi}*")

    durasi_tidur = st.number_input("Durasi Tidur (jam/hari)", 0.0, 24.0, 6.0)
    kualitas_tidur = st.slider("Kualitas Tidur (1–10)", 1, 10, 5)
    aktivitas_fisik = st.slider("Aktivitas Fisik (1–100)", 1, 100, 40)
    tingkat_stres = st.slider("Tingkat Stres (1–10)", 1, 10, 7)
    heart_rate = st.number_input("Detak Jantung (bpm)", 40, 180, 72)
    daily_steps = st.number_input("Langkah Harian", 0, 50000, 5000)

    systolic = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
    diastolic = st.number_input("Tekanan Darah Diastolik", 40, 130, 80)

    if st.button("Prediksi Sekarang"):
        if nama.strip() == "":
            st.warning("Nama wajib diisi.")
        else:
            # ==============================
            # INPUT MODEL (9 FITUR)
            # ==============================
            X_input = pd.DataFrame([[
                    umur,
                    durasi_tidur,
                    kualitas_tidur,
                    aktivitas_fisik,
                    tingkat_stres,
                    heart_rate,
                    daily_steps,
                    systolic,
                    diastolic
                ]], columns=feature_columns)

            X_scaled = scaler.transform(X_input)
            prob = model.predict_proba(X_scaled)[0]
            probs = dict(zip(label_classes, prob))

            # ==============================
            # 🔥 MAPPING LABEL (NaN → Normal)
            # ==============================
            prob_display = {}
            for k, v in probs.items():
                if pd.isna(k):
                    prob_display["Normal"] = v
                else:
                    prob_display[k] = v

            # ==============================
            # 🔥 PREDIKSI FINAL (ARGMAX)
            # ==============================
            hasil = max(prob_display, key=prob_display.get)

            prob_dict = {
                k: round(v * 100, 2)
                for k, v in prob_display.items()
            }

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
    
            st.success("Prediksi berhasil. Lihat menu **Hasil Prediksi**.")

# ==============================
# 7️⃣ HASIL PREDIKSI (UI TETAP)
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
        st.warning("Belum ada prediksi.")
    else:
        row = df.iloc[0]
        st.subheader(f"Hasil: **{row['hasil']}**")
        st.info(keterangan_tidur(row["hasil"]))

# ==============================
# 8️⃣ RIWAYAT + CSV
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
        st.write("Belum ada data.")
    else:
        st.dataframe(df)

        st.download_button(
            "⬇️ Unduh CSV",
            df.to_csv(index=False),
            "riwayat_prediksi.csv",
            "text/csv"
        )

# ==============================
# 9️⃣ TIPS & TENTANG (TETAP)
# ==============================
elif menu == "💤 Tips Tidur Sehat":
    st.title("💤 Tips Tidur Sehat")
    st.markdown("""
    - Tidur dan bangun di jam yang sama
    - Kurangi kafein dan gadget sebelum tidur
    - Kelola stres dengan relaksasi
    - Rutin berolahraga ringan
    """)

elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang")
    st.markdown("""
    - **Aplikasi:** Sistem Deteksi Gangguan Tidur
    - **Metode:** Support Vector Machine (SVM)
    - **Pengembang:** Satria Dava Riansa (G.211.22.0006) – Universitas Semarang
    """)
