import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# 1️⃣ Load Model Bundle
# ==============================

import requests

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
            response = requests.post(
                "http://sleep-disorder-app-production.up.railway.app/predict",
                json={
                    "Age": umur,
                    "Sleep_Duration": durasi_tidur,
                    "Quality_of_Sleep": kualitas_tidur,
                    "Physical_Activity_Level": aktivitas_fisik,
                    "Stress_Level": tingkat_stres,
                    "Heart_Rate": heart_rate,
                    "Daily_Steps": daily_steps,
                    "Systolic_BP": systolic,
                    "Diastolic_BP": diastolic
                }
            )
            
            hasil_api = response.json()

            st.write("Status Code:", response.status_code)
            st.write("Response API:", hasil_api)
            
            if response.status_code != 200:
                st.error("API Error")
                st.stop()
            
            hasil = hasil_api["prediction"]
            prob_dict = hasil_api["probability"]
            
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

# ==============================
# 8️⃣ Riwayat Prediksi
# ==============================
elif menu == "🕓 Riwayat Prediksi":
    st.title("🕓 Riwayat Prediksi")
    if len(st.session_state.riwayat) == 0:
        st.write("Belum ada data.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.riwayat))

# ==============================
# 9️⃣ Tips Tidur
# ==============================
elif menu == "💤 Tips Tidur Sehat":
    st.title("💤 Tips Tidur Sehat")
    st.markdown("""
    ### 🕒 1. Konsisten Waktu Tidur
    Tidur dan bangun pada jam yang sama setiap hari membantu menjaga ritme sirkadian tubuh sehingga kualitas tidur lebih stabil.

    ### ☕ 2. Batasi Kafein & Gadget
    Hindari konsumsi kafein (kopi, teh, minuman energi) serta penggunaan gadget minimal 1–2 jam sebelum tidur karena dapat mengganggu produksi hormon melatonin.

    ### 😌 3. Kelola Stres dengan Baik
    Lakukan relaksasi seperti meditasi, pernapasan dalam, atau membaca buku untuk membantu menenangkan pikiran sebelum tidur.

    ### 🏃 4. Rutin Beraktivitas Fisik
    Olahraga ringan hingga sedang secara rutin dapat membantu meningkatkan kualitas tidur, namun hindari olahraga berat menjelang waktu tidur.

    ### 🛏️ 5. Ciptakan Lingkungan Tidur Nyaman
    Pastikan kamar tidur dalam kondisi gelap, sejuk, dan tenang agar tubuh lebih mudah untuk beristirahat.

    ### 🍽️ 6. Perhatikan Pola Makan
    Hindari makan berat menjelang tidur dan batasi konsumsi gula berlebih di malam hari.

    ### 📊 7. Perhatikan Tanda Gangguan Tidur
    Jika mengalami kesulitan tidur, sering terbangun, atau merasa lelah saat bangun, pertimbangkan untuk melakukan pemeriksaan lebih lanjut.
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
