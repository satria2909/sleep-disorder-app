import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Load Model Bundle
# ==============================
bundle = joblib.load("best_sleep_disorder_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]   # 9 fitur
label_encoder = bundle["label_encoder"]
label_classes = label_encoder.classes_

# ==============================
# 2️⃣ Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Deteksi Gangguan Tidur",
    page_icon="😴",
    layout="centered"
)

# ==============================
# 3️⃣ Fungsi Bantu
# ==============================
def hitung_bmi(tinggi_cm, berat_kg):
    tinggi_m = tinggi_cm / 100
    return berat_kg / (tinggi_m ** 2)

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
# 4️⃣ Sidebar Navigasi
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

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
    
# ==============================
# 5️⃣ Beranda
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
# 6️⃣ Form Prediksi
# ==============================
elif menu == "🧮 Prediksi Tidur":
    st.title("Formulir Prediksi Tidur")

    nama = st.text_input("Nama Lengkap")
    umur = st.number_input("Usia (tahun)", 1, 100, 25)

    tinggi = st.number_input("Tinggi Badan (cm)", 140, 220, 170)
    berat = st.number_input("Berat Badan (kg)", 40, 200, 65)

    bmi = hitung_bmi(tinggi, berat)
    kategori_bmi = kategori_bmi_text(bmi)
    st.write(f"**BMI:** {bmi:.2f} — *{kategori_bmi}*")

    durasi_tidur = st.number_input("Durasi Tidur (jam/hari)", 0.0, 24.0, 6.0)
    kualitas_tidur = st.slider("Kualitas Tidur (1–10)", 1, 10, 5)
    aktivitas_fisik = st.slider("Aktivitas Fisik (1–100)", 1, 100, 50)
    tingkat_stres = st.slider("Tingkat Stres (1–10)", 1, 10, 5)
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

            # ==============================
            # 🔥 MAPPING LABEL (NaN → Normal)
            # ==============================
            probs = dict(zip(label_classes, prob))

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

            st.success("Prediksi berhasil. Lihat menu **Hasil Prediksi**.")

# ==============================
# 7️⃣ Hasil Prediksi
# ==============================
elif menu == "📊 Hasil Prediksi":
    st.title("📊 Hasil Prediksi")
    if "last_pred" not in st.session_state:
        st.warning("Belum ada prediksi.")
    else:
        data = st.session_state.last_pred

        st.subheader(f"Hasil: **{data['Hasil']}**")
        st.write(f"**BMI:** {data['BMI']:.2f} — *{data['Kategori BMI']}*")
        st.info(keterangan_tidur(data["Hasil"]))

        # ==============================
        # 🎨 UI Tambahan
        # ==============================

        st.markdown("### Tingkat Probabilitas")
        for label, value in data["Probabilitas"].items():
            st.write(f"{label}: {value}%")
            st.progress(value / 100)

        import matplotlib.pyplot as plt

        labels = list(data["Probabilitas"].keys())
        values = list(data["Probabilitas"].values())

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilitas (%)")
        ax.set_title("Distribusi Probabilitas Prediksi")

        st.pyplot(fig)

# ==============================
# 8️⃣ Riwayat Prediksi
# ==============================
elif menu == "🕓 Riwayat Prediksi":
    st.title("🕓 Riwayat Prediksi")

    if len(st.session_state.riwayat) == 0:
        st.write("Belum ada data.")
    else:
        df_riwayat = pd.DataFrame(st.session_state.riwayat)
        st.dataframe(df_riwayat)

        st.markdown("### 📤 Ekspor Data")

        # ==============================
        # 📄 Export CSV
        # ==============================
        csv = df_riwayat.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Unduh CSV",
            data=csv,
            file_name="riwayat_prediksi_tidur.csv",
            mime="text/csv"
        )

        # ==============================
        # 📄 Export PDF
        # ==============================
        from fpdf import FPDF

        def generate_pdf(dataframe):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)

            pdf.cell(0, 10, "Riwayat Prediksi Gangguan Tidur", ln=True, align="C")
            pdf.ln(5)

            # Header
            for col in dataframe.columns:
                pdf.cell(40, 8, col, border=1)
            pdf.ln()

            # Data
            for _, row in dataframe.iterrows():
                for item in row:
                    pdf.cell(40, 8, str(item), border=1)
                pdf.ln()

            return pdf.output(dest="S").encode("latin-1")

        pdf_data = generate_pdf(df_riwayat)

        st.download_button(
            label="⬇️ Unduh PDF",
            data=pdf_data,
            file_name="riwayat_prediksi_tidur.pdf",
            mime="application/pdf"
        )

# ==============================
# 9️⃣ Tips Tidur
# ==============================
elif menu == "💤 Tips Tidur Sehat":
    st.title("💤 Tips Tidur Sehat")
    st.markdown("""
    - Tidur dan bangun di jam yang sama
    - Kurangi kafein dan gadget sebelum tidur
    - Kelola stres dengan relaksasi
    - Rutin berolahraga ringan
    """)

# ==============================
# 🔟 Tentang
# ==============================
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang")
    st.markdown("""
    - **Aplikasi:** Sistem Deteksi Gangguan Tidur
    - **Metode:** Support Vector Machine (SVM)
    - **Pengembang:** Satria Dava Riansa (G.211.22.0006) – Universitas Semarang
    """)
