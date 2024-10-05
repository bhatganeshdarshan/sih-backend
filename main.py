from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

app = FastAPI()

# Load model
model = load_model("model_resnet50_1.h5")

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class_labels = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]  # Ensure these are correct labels

# Health check route
@app.get("/")
async def index():
    return {"message": "Blood Group Prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict
        preds = model.predict(x)
        predicted_class = np.argmax(preds)
        return {
            "predicted_class": class_labels[predicted_class],
            "confidence_scores": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

