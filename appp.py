import streamlit as st
import cv2
import tempfile
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
import os
from pyngrok import ngrok
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from PIL import Image, ImageColor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
os.makedirs(os.path.join('age', 'output'), exist_ok=True)
os.makedirs(os.path.join('emotion', 'output'), exist_ok=True)
os.makedirs(os.path.join('gendre', 'output'), exist_ok=True)


face_detector = MTCNN()
try:
    emotion_path = os.path.join('emotion_model.keras')
    age_path = os.path.join('age_model_pretrained.h5')
    gender_path = os.path.join('gender_model.keras')
    if not os.path.exists(emotion_path):
        st.error(f"Emotion model not found at {emotion_path}")
        st.error("Please run train_emotion.py first")
        st.stop()
    if not os.path.exists(gender_path):
        st.error(f"Gender model not found at {gender_path}")
        st.error("Please run train_gender.py first")
        st.stop()
    emotion_model = load_model(emotion_path)
    age_model = load_model(age_path)
    gender_model = load_model(gender_path)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.error("Please make sure you have trained all models first by running:")
    st.code("python age/train_age.py")
    st.code("python emotion/train_emotion.py")
    st.code("python gendre/train_gender.py")
    st.stop()
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['positive', 'negative', 'neutral']
class_labels = emotion_ranges
gender_labels = gender_ranges
face_detector = MTCNN()
def predict_age_gender_emotion(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(image)
    i = 0
    for face in faces:
        if len(face['box']) == 4:
            i = i + 1
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = cv2.equalizeHist(roi_gray)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            output_emotion = class_labels[np.argmax(emotion_model.predict(roi))]
            gender_img = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
            gender_image_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_image_array, axis=0)
            output_gender = gender_labels[np.argmax(gender_model.predict(gender_input))]
            age_image = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            age_input = age_image.reshape(-1, 200, 200, 1)
            output_age = age_ranges[np.argmax(age_model.predict(age_input))]
            output_str = str(i) + ": " + output_gender + ', ' + output_age + ', ' + output_emotion
            col = (0, 255, 0)
            cv2.putText(image, output_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)
            print(output_str)
    return image


def app():
    st.title("Age, Gender, and Emotion Recognition")
    detection_mode = None
    with st.sidebar:
        title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
        st.markdown(title, unsafe_allow_html=True)
        mode = st.radio("Choose Face Detection Mode", ('Image Upload',
                                                       'Webcam Image Capture',
                                                       'Webcam Realtime frame by frame'), index=0)
        if mode == 'Image Upload':
            detection_mode = mode
        elif mode == 'Video Upload':
            detection_mode = mode
        elif mode == "Webcam Image Capture":
            detection_mode = mode
        elif mode == 'Webcam Realtime frame by frame':
            detection_mode = mode
        elif mode == 'real time face detection':
            detection_mode = mode

    if detection_mode == "Image Upload":
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key=1)
        if image_file is not None:
          
            img = Image.open(image_file)
            img = np.array(img)
            result_image = predict_age_gender_emotion(img)
            st.image(result_image, channels="BGR", use_column_width=True)
    if detection_mode == "Webcam Image Capture":
        image = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                help="Make sure you have given webcam permission to the site")
        if image is not None:
            img = Image.open(image)
            image = np.array(img) 

            predicted_image = predict_age_gender_emotion(image)

            st.image(predicted_image, channels="BGR")


    if detection_mode == "Webcam Realtime frame by frame":
   
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            labels = []
            frame = predict_age_gender_emotion(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    app()