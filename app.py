import streamlit as st
import cv2
import numpy as np
from deepface.DeepFace import analyze
from PIL import Image

st.title("Age and Emotion Predictor")
st.write("Upload an image or use your webcam to predict age and emotion.")

option = st.radio("Select Input Type", ("Upload Image", "Use Webcam"))

def predict_age_emotion(image):
    try:
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  
        results = analyze(img_rgb, actions=['age', 'emotion'], enforce_detection=False)

        if isinstance(results, dict):
            results = [results] 

        return results
    except Exception as e:
        print("Error:", str(e))
        return []

def draw_results(frame, results):
    for face in results:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 50), region.get('y', 50), region.get('w', 100), region.get('h', 100)
        
        age = face.get('age', "Unknown")
        emotion = face.get('dominant_emotion', "Unknown").capitalize()
        text = f"Age: {age}, Emotion: {emotion}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing... Please wait.")

        results = predict_age_emotion(image)

        if results:
            image_array = np.array(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            processed_image = draw_results(image_array, results)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            st.write("Could not detect any faces. Please try another image.")

elif option == "Use Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = analyze(frame_rgb, actions=['age', 'emotion'], enforce_detection=False)

        if results:
            processed_frame = draw_results(frame, results)
        else:
            processed_frame = frame  

        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
