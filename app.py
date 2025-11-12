import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import crop_center, extract_glcm_features
import joblib
from model import load_model, predict  

# --------------------------------------------
# Fungsi bantu
# --------------------------------------------
def preprocess_image(image):
    """Konversi gambar (PIL->np array RGB) ke grayscale, crop tengah, dan resize."""
    # image expected as numpy array in RGB from PIL
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    cropped = crop_center(gray)
    resized = cv2.resize(cropped, (128, 128))
    return gray, cropped, resized

def calculate_glcm_dataframe(img):
    """Hitung fitur GLCM (tidak melakukan scaling di sini)."""
    props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    features = extract_glcm_features(img, props)
    columns = [f"{prop}_{ang}" for prop in props for ang in ['0', '45', '90', '135']]
    df = pd.DataFrame([features], columns=columns)
    return df

def load_confusion_matrix():
    try:
        cm = pd.read_csv("confusion_matrix.csv", index_col=0)
        return cm
    except FileNotFoundError:
        return None

# --------------------------------------------
# Aplikasi Streamlit
# --------------------------------------------
def main():
    st.title("Aplikasi Klasifikasi Motif Batik")
    st.write("Upload gambar batik untuk diprediksi motifnya (Jelamprang, Jlamprang, Buketan).")

    uploaded_file = st.file_uploader("Unggah gambar batik", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Membaca dan menampilkan gambar (PIL -> numpy RGB)
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)  # in RGB
        st.image(image, caption="Gambar Asli", width=300)

        # load scaler (dihasilkan saat training)
        try:
            scaler = joblib.load("scaler.joblib")
        except FileNotFoundError:
            st.error("Scaler tidak ditemukan. Jalankan training terlebih dahulu (python train_model.py).")
            return

        # preprocessing gambar
        gray, cropped, resized = preprocess_image(image)
        glcm_df = calculate_glcm_dataframe(resized)

        # normalisasi / standardisasi menggunakan scaler training
        try:
            glcm_scaled = scaler.transform(glcm_df.values)
        except Exception as e:
            st.error(f"Gagal melakukan transformasi scaler: {e}")
            return

        # load model weights & biases
        try:
            weights, biases = load_model("model_weights.npz")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return

        # Prediksi
        try:
            pred = predict(glcm_scaled, weights, biases)[0]
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            return

        label_map = {0: "Jelamprang", 1: "Jlamprang", 2: "Buketan"}
        result = label_map.get(int(pred), "Tidak terdeteksi")

        st.subheader("Hasil Prediksi")
        st.success(f"Motif Batik: **{result}**")

        # Tampilkan tahapan preprocessing
        st.subheader("Tahapan Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray, caption="Grayscale", use_container_width=True)
        with col2:
            st.image(cropped, caption="Crop Tengah", use_container_width=True)
        with col3:
            st.image(resized, caption="Resize 128x128", use_container_width=True)

        # Tampilkan GLCM fitur (sebelum scaling)
        st.subheader("Fitur GLCM (sebelum scaling)")
        st.write(glcm_df)

        # Confusion matrix dari hasil training
        cm = load_confusion_matrix()
        if cm is not None:
            st.subheader("Confusion Matrix (Dari Training)")
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
            st.pyplot(plt)
        else:
            st.warning("Confusion matrix belum tersedia. Jalankan training terlebih dahulu.")

if __name__ == "__main__":
    main()
