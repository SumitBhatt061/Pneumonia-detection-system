import os
import gdown
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- CONFIG ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "model.keras"
MODEL_ID = "1hXaUhny8dar1wpRKzClVTXcjR3h1AbGL"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# ---------------- DOWNLOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# ---------------- PREDICTION ----------------
def get_class_name(class_index):
    return "Normal" if class_index == 0 else "Pneumonia"

def predict_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]

    normal_prob = float(preds[0])
    pneumonia_prob = float(preds[1])

    if pneumonia_prob > 0.90:
        return 1, pneumonia_prob
    else:
        return 0, normal_prob

# ---------------- UI ----------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🫁 Pneumonia Detection System")
st.write("Upload a chest X-ray image to predict pneumonia.")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    image_np = np.array(image)

    if st.button("Predict"):
        with st.spinner("Analyzing X-ray..."):
            class_index, confidence = predict_image(image_np)
            result = get_class_name(class_index)

        st.success(f"**{result}** ({confidence*100:.2f}%)")
