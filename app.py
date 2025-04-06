import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import tempfile
import os

st.set_page_config(page_title="OCR Uygulaması", layout="wide")
st.title("🧾 Görüntüden Yazı Tanıma Uygulaması (OCR)")

uploaded_file = st.file_uploader("📤 Bir görsel dosyası yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Görüntüyü OpenCV formatına çevir
    image_np = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("🛠️ Görüntü İşleme Seçenekleri")
    mode = st.radio("OCR Modu Seçin", ["Basit", "Gelişmiş"], horizontal=True)

    processed = image_cv.copy()

    if mode == "Gelişmiş":
        # Görüntü işlemleri
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        if st.checkbox("📎 Grayscale Uygula", value=True):
            processed = gray

        if st.checkbox("✨ Gürültü Gider (Median Blur)"):
            processed = cv2.medianBlur(processed, 5)

        if st.checkbox("⚫ Threshold (Otsu)"):
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if st.checkbox("📐 Eğim Düzelt (Skew Correction)"):
            coords = np.column_stack(np.where(processed > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = processed.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            processed = cv2.warpAffine(processed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        st.image(processed, caption="İşlenmiş Görsel", use_container_width=True, channels="GRAY" if len(processed.shape)==2 else "BGR")

    # OCR işlemi
    st.subheader("🔍 OCR Sonucu")
    with st.spinner("Yazılar algılanıyor..."):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed)
        extracted_text = "\n".join([res[1] for res in result])

    st.text_area("📋 Tanınan Metin", value=extracted_text, height=200)

    # Görsel üzerinde yazıları kutucuklarla göster
    st.subheader("🖼️ Tespit Edilen Yazılar")
    img_boxed = image_cv.copy()
    for (bbox, text, conf) in result:
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        br = tuple(map(int, br))
        cv2.rectangle(img_boxed, tl, br, (0, 255, 0), 2)
        cv2.putText(img_boxed, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    st.image(cv2.cvtColor(img_boxed, cv2.COLOR_BGR2RGB), caption="Yazılarla İşaretlenmiş Görsel", use_container_width=True)
