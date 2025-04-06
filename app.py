import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import re

st.title("📄 OCR Görüntüden Yazı Tanıma Uygulaması")

# Görsel yükleme
uploaded_file = st.file_uploader("📁 Bir görsel yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # OpenCV formatına dönüştür
    image_cv = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    st.subheader("🔍 OCR Metni")
    custom_config = r'-l eng --oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(image_cv, config=custom_config)
    st.text_area("Tespit Edilen Metin", extracted_text, height=150)

    st.subheader("🧹 Temizlenmiş Metin")
    cleaned_text = re.sub(r"[!()@—*“>+\-/,'|£#%$&^_~]", "", extracted_text)
    st.text_area("Sembolleri Temizlenmiş Metin", cleaned_text, height=150)

    # Görüntü işleme fonksiyonları
    def get_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def remove_noise(img):
        return cv2.medianBlur(img, 5)

    def thresholding(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def erode(img):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    def opening(img):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def canny(img):
        return cv2.Canny(img, 100, 200)

    def deskew(img):
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    st.subheader("⚙️ Görüntü İşleme Adımları")
    gray = get_grayscale(image_cv)
    noise = remove_noise(gray)
    thresh = thresholding(noise)
    eroded = erode(thresh)
    opened = opening(eroded)
    edges = canny(opened)
    deskewed = deskew(gray)

    steps = {
        "Gri Tonlama": gray,
        "Gürültü Temizleme": noise,
        "Eşikleme": thresh,
        "Aşındırma (Erosion)": eroded,
        "Açılma (Morphology)": opened,
        "Kenar Algılama (Canny)": edges,
        "Yamuk Düzeltme (Deskew)": deskewed
    }

    for title, step_img in steps.items():
        st.markdown(f"**{title}**")
        st.image(step_img, channels="GRAY")

    st.subheader("📦 Metin Kutularını Göster")
    box_img = image_cv.copy()
    h, w, _ = box_img.shape
    boxes = pytesseract.image_to_boxes(box_img)
    for b in boxes.splitlines():
        b = b.split()
        box_img = cv2.rectangle(box_img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    st.image(box_img, caption="Metin Kutuları", use_container_width=True)

    st.subheader("🔎 Belirli Kelimeyi Bul")
    search_term = st.text_input("Aranacak kelime (örneğin: artificially)", value="artificially")

    if search_term:
        img_with_word = image_cv.copy()
        data = pytesseract.image_to_data(img_with_word, output_type=Output.DICT)
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            if int(data['conf'][i]) > 60 and re.match(search_term, data['text'][i], re.IGNORECASE):
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                img_with_word = cv2.rectangle(img_with_word, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(img_with_word, caption=f"'{search_term}' kelimesi işaretlendi", use_container_width=True)
