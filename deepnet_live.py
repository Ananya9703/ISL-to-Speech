import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque
import joblib
import pyttsx3
import json
import requests
import threading


# Load models and data
model = load_model('gesture_model.h5')
gesture_labels = np.load('label_encoder_classes.npy', allow_pickle=True)
scaler = joblib.load('gesture_scaler_1.joblib')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine in a thread-safe way
engine = None
engine_lock = threading.Lock()

def initialize_engine():
    global engine
    with engine_lock:
        if engine is None:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)

def text_to_speech(text):
    def speak():
        global engine
        initialize_engine()
        with engine_lock:
            engine.say(text)
            engine.runAndWait()
    
    # Run speech in a separate thread
    thread = threading.Thread(target=speak)
    thread.start()

def get_chat_completion(prompt, api_url="http://localhost:11434/api/generate"):
    payload = {
        "model": "llama3.1:latest",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.01
    }
    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            raw_content = response.content.decode("utf-8")
            ndjson_lines = raw_content.splitlines()
            return ''.join(json.loads(line)["response"] for line in ndjson_lines if json.loads(line).get("response"))
    except Exception as e:
        print(f"API Error: {e}")
    return None

# Define keypoint parameters
hand_keypoints = 21 * 3
face_keypoints = 468 * 3
actual_length = hand_keypoints + face_keypoints
expected_length = 1593

# Initialize capture and buffers
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=10)
phrase = ""
current_gesture = "No hand detected"

# Initialize text-to-speech engine at startup
initialize_engine()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        keypoints = []
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])

        if keypoints:
            keypoints = np.pad(keypoints, (0, expected_length - len(keypoints)), mode='constant') if len(keypoints) < expected_length else keypoints[:expected_length]
            keypoints = np.array(keypoints).reshape(1, -1)
            keypoints_scaled = scaler.transform(keypoints)
            predictions = model.predict(keypoints_scaled)
            gesture_idx = np.argmax(predictions)

            if 0 <= gesture_idx < len(gesture_labels):
                current_gesture = gesture_labels[gesture_idx]
                if len(buffer) == 0 or current_gesture != buffer[-1]:
                    buffer.append(current_gesture)
                    phrase += f" {current_gesture}"
    else:
        current_gesture = "No hand detected"

    # Draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Phrase: {phrase}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Gesture Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        phrase = ""
    elif key == ord('t') and phrase.strip():
        prompt = f"Convert the following words into a grammatically correct sentence, without assuming any extra context. Only return the sentence. The words are: {', '.join(phrase.strip().split())}"
        sentence = get_chat_completion(prompt)
        if sentence:
            print("Generated Sentence:", sentence)
            text_to_speech(sentence)
            phrase = ""

# Cleanup
cap.release()
cv2.destroyAllWindows()
with engine_lock:
    if engine:
        engine.stop()
