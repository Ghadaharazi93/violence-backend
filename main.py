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
model_path = "model.h5"


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


# حمل الملف إذا لم يكن موجودًا
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    download_file_from_google_drive(file_id, model_path)
    print("Model downloaded!")

# تحميل المودل بعد ما نضمن أنه نزل كامل
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
