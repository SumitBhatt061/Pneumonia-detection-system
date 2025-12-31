import os
import gdown
import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"
MODEL_ID = "1hXaUhny8dar1wpRKzClVTXcjR3h1AbGL"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_class_name(class_index):
    if class_index == 0:
        return "Normal"
    return "Pneumonia"

def predict_image(img_path):
    image = cv2.imread(img_path)
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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    class_index, confidence = predict_image(file_path)
    result = get_class_name(class_index)

    return f"{result} ({confidence*100:.2f}%)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
