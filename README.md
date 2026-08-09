  # 🔥 Forest Fire Detection AI Model

A web app that detects forest fires from images using a Convolutional Neural Network (CNN), built with Keras/TensorFlow and deployed via Streamlit.

## 🚀 Live Demo
[Try it here](https://forest-fire-detection-ai-qhie2jgntmyk7dsafhu6kk.streamlit.app)

## 📋 Overview
This project uses deep learning to classify images as containing fire or not, aimed at early wildfire detection. The model is trained on the [Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) from Kaggle.

## 🧠 Model
- **Architecture:** Convolutional Neural Network (CNN) built with Keras/TensorFlow
- **Dataset:** Kaggle `phylake1337/fire-dataset`
- **Input:** Uploaded image
- **Output:** Binary classification (Fire / No Fire)

## 🔄 Model Status
This model is actively being retrained for improved accuracy using data augmentation techniques (rotation, flipping, zoom) to reduce false positives and better generalize to real-world images.

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit (web interface + deployment)

## 📦 Installation

```bash
git clone https://github.com/samanyu8055/forest-fire-detection-ai-model.git
cd forest-fire-detection-ai-model
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Upload an image and the model will predict whether it shows a fire.

## 📁 Project Structure

```
forest-fire-detection-ai-model/
├── app.py
├── fire_detector_model.keras
├── requirements.txt
├── runtime.txt
└── README.md
```

## ⚠️ Limitations
- Trained on a relatively small dataset, so accuracy on unseen or unusual images may vary
- May confuse fire with visually similar things like sunsets, red/orange lighting, or campfires
- Performance can drop on blurry, low-resolution, or poorly lit images
- Currently supports single-image classification only — no real-time video or multi-frame analysis
- Binary output only (Fire / No Fire) — does not estimate fire size, severity, or location within the image
