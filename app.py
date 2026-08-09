import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

st.set_page_config(page_title="Forest Fire Detection", layout="centered")
st.title("🔥 Forest Fire Detection AI")

model = joblib.load("fire_detector_model.pkl")

uploaded = st.file_uploader("Choose Image", type=["png","jpg","jpeg"])

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64,64))
    img = img.astype("float32") / 255.0
    img = img.flatten().reshape(1, -1)

    with st.spinner("Analyzing image..."):
        decision = model.decision_function(img)[0]
        confidence = 1 / (1 + np.exp(-decision))  

    if decision > 0:
        st.error(f"🔥 FIRE DETECTED — Confidence: {confidence*100:.1f}%")
    else:
        st.success(f"🌲 NO FIRE — Confidence: {(1-confidence)*100:.1f}%")
 
