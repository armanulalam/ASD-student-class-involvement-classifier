import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf


MODEL_PATH = os.path.join("model", "ensemble_fixed.keras")
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
UPLOAD_FOLDER = "uploads"

CLASS_NAMES = ["Involved", "Not Involved"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB ম্যাক্স ফাইল সাইজ


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Please run python convert_model.py first to generate the correct .keras file from the original .h5 file (see README.md)."
    )

print("Loading model... this might take a moment.")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")   # শুধু raw 0-255 float32
    arr = np.expand_dims(arr, axis=0)       # batch dimension -> (1, 224, 224, 3)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file found."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only png, jpg, jpeg, and webp files are supported."}), 400

    try:
        filename = secure_filename(file.filename)
        img = Image.open(file.stream)

        input_arr = preprocess_image(img)
        preds = model.predict(input_arr, verbose=0)[0]   # shape (2,)

        result = {
            "predicted_class": CLASS_NAMES[int(np.argmax(preds))],
            "confidence": float(np.max(preds)),
            "probabilities": {
                CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
            },
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
