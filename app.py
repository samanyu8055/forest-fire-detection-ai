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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64,64))
    img = img.astype("uint8")
    img = img.flatten().reshape(1,-1)

    with st.spinner("Analyzing image..."):
        decision = model.decision_function(img)[0]

    if decision > 0:
        st.error("🔥 FIRE DETECTED")
    else:
        st.success("🌲 NO FIRE")
 
