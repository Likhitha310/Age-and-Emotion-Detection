from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os
from PIL import Image
import base64

# Initialize the Flask app
app = Flask(__name__)

# Create necessary directories
os.makedirs(os.path.join('age', 'output'), exist_ok=True)
os.makedirs(os.path.join('emotion', 'output'), exist_ok=True)
os.makedirs(os.path.join('gendre', 'output'), exist_ok=True)

# Load models
face_detector = MTCNN()
emotion_path = os.path.join('emotion', 'output', 'emotion_model.keras')
age_path = os.path.join('age', 'output', 'age_model_pretrained.h5')
gender_path = os.path.join('gendre', 'output', 'gender_model.keras')

if not all(os.path.exists(path) for path in [emotion_path, age_path, gender_path]):
    raise FileNotFoundError("Some models are missing. Ensure all models are trained and present.")

emotion_model = load_model(emotion_path)
age_model = load_model(age_path)
gender_model = load_model(gender_path)

# Labels for predictions
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_labels = ['male', 'female']
emotion_labels = ['positive', 'negative', 'neutral']

def predict_age_gender_emotion(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(image)

    results = []

    for face in faces:
        if len(face['box']) == 4:
            x, y, w, h = face['box']
            roi_gray = gray[y:y + h, x:x + w]

            # Resize and normalize ROI for emotion model
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = cv2.equalizeHist(roi_gray)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Emotion prediction
            emotion = emotion_labels[np.argmax(emotion_model.predict(roi))]

            # Gender prediction
            gender_img = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
            gender_input = np.expand_dims(gender_img, axis=0)
            gender = gender_labels[np.argmax(gender_model.predict(gender_input))]

            # Age prediction
            age_img = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            age_input = age_img.reshape(-1, 200, 200, 1)
            age = age_ranges[np.argmax(age_model.predict(age_input))]

            # Store results
            results.append({
                
                "gender": gender,
                "age": age,
                "emotion": emotion,
            })
            print(results)

    return results

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    image = np.array(image)

    try:
        predictions = predict_age_gender_emotion(image)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return jsonify({"message": "Age, Gender, and Emotion Recognition API"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=47986,use_reloader=False)

