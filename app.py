import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Forest Fire Detection", layout="centered")
st.title("🔥 Forest Fire Detection AI")

model = tf.keras.models.load_model("fire_detector_model.keras")

uploaded = st.file_uploader("Choose Image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = image.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img)[0][0]

    # class_names order was ['fire_images', 'non_fire_images'] alphabetically
    # so 0 = fire, 1 = non_fire -> lower prediction score = more "fire"
    fire_confidence = (1 - prediction) * 100

    if prediction < 0.5:
        st.error(f"🔥 FIRE DETECTED — Confidence: {fire_confidence:.1f}%")
    else:
        st.success(f"🌲 NO FIRE — Confidence: {(100 - fire_confidence):.1f}%")
