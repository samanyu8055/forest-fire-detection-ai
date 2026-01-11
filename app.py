import streamlit as st
import numpy as np
import cv2
import joblib

st.set_page_config(page_title="Forest Fire Detection", layout="centered")

st.title("🔥 Forest Fire Detection AI")

# Load model
model = joblib.load("fire_detector_model.pkl")

uploaded = st.file_uploader("Choose Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        st.warning("Invalid image file")
        st.stop()

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img2 = cv2.resize(img, (64, 64))
    img2 = img2.astype("float32") / 255.0
    img2 = img2.flatten().reshape(1, -1)

    # Predict
    with st.spinner("Analyzing image..."):
        pred = model.predict(img2)
        decision = model.decision_function(img2)[0]
        confidence = (1 / (1 + np.exp(-decision))) * 100

    # Result
           if pred[0] == 1:
               st.error(f"🔥 FIRE DETECTED — {confidence:.2f}% confidence")
           else:
               st.success(f"✅ NO FIRE — {confidence:.2f}% confidence")
 
