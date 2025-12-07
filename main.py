from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_id = "1nXKrTVGIhNt91V6ZEawWm7QY8Mfcoy_s"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "model.h5"

if not os.path.exists(model_path):
    print("Downloading model...")
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

model = tf.keras.models.load_model(model_path)

# حجم التدريب
IMG_SIZE = 224

# جميع الكلاسات حسب تدريبك
CLASS_NAMES = [
    "hit", "kick", "punch", "push", "shoot_gun",
    "ride_horse", "stand", "wave"
]

# الكلاسات العنيفة
VIOLENCE_CLASSES = {"hit", "kick", "punch", "push", "shoot_gun"}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """نفس تجهيز الصورة في كولاب تمامًا"""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image_array(image: Image.Image):
    """منطق التصنيف نفس الي تستخدمينه على الفيديو"""
    x = preprocess_image(image)
    preds = model.predict(x, verbose=0)[0]

    class_id = int(np.argmax(preds))
    class_name = CLASS_NAMES[class_id]
    confidence = float(preds[class_id])

    binary_label = "VIOLENCE" if class_name in VIOLENCE_CLASSES else "NON-VIOLENCE"

    return binary_label, class_name, confidence


@app.get("/")
def home():
    return {"status": "ok", "message": "Violence Detection API is running"}


@app.post("/predict-image")
async def predict_endpoint(file: UploadFile = File(...)):
    """يرجع نفس النتائج اللي تظهر عندك في كولاب"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    binary_label, action_label, confidence = predict_image_array(image)

    return {
        "binary_label": binary_label,
        "action_label": action_label,
        "confidence": confidence
    }
