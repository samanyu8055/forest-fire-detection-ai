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

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit (web interface + deployment)

## 📦 Installation

```bash
git clone https://github.com/samanyu8055/forest-fire-detection-ai.git
cd forest-fire-detection-ai
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Upload an image and the model will predict whether it shows a fire.

## 📁 Project Structure
