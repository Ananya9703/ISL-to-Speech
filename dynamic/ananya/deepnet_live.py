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
import time


# Load models and data
model = load_model('gesture_model.h5')
gesture_labels = np.load('label_encoder_classes.npy', allow_pickle=True)
scaler = joblib.load('gesture_scaler_1.joblib')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
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

# Add these new variables for gesture timing
GESTURE_COLLECTION_TIME = 10  # seconds
gesture_start_time = None
is_collecting = False
temp_predictions = []  # Store predictions during collection period

# Initialize capture and buffers
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=10)
phrase = ""
current_gesture = "No hand detected"

# Initialize text-to-speech engine at startup
initialize_engine()

def reset_collection_state():
    global is_collecting, temp_predictions, gesture_start_time, current_gesture
    is_collecting = False
    temp_predictions = []
    gesture_start_time = None
    current_gesture = "No hand detected"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # Start collection when hand is first detected
    if hand_results.multi_hand_landmarks and not is_collecting:
        gesture_start_time = time.time()
        is_collecting = True
        temp_predictions = []
        print("Started collecting gesture...")

    if is_collecting:
        elapsed_time = time.time() - gesture_start_time
        remaining_time = max(0, GESTURE_COLLECTION_TIME - elapsed_time)

        # Process landmarks during collection period
        if hand_results.multi_hand_landmarks and elapsed_time < GESTURE_COLLECTION_TIME:
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
                    temp_predictions.append(gesture_labels[gesture_idx])
                    current_gesture = gesture_labels[gesture_idx]

        # After collection period ends
        elif elapsed_time >= GESTURE_COLLECTION_TIME:
            if temp_predictions:
                # Get most common prediction during the collection period
                from collections import Counter
                most_common_gesture = Counter(temp_predictions).most_common(1)[0][0]
                if len(buffer) == 0 or most_common_gesture != buffer[-1]:
                    buffer.append(most_common_gesture)
                    phrase += f" {most_common_gesture}"
                    print(f"Added gesture to phrase: {most_common_gesture}")
            
            # Reset collection state
            reset_collection_state()
    
    # Draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

    # Update display
    if is_collecting:
        cv2.putText(frame, f"Collecting gesture: {current_gesture} ({int(remaining_time)}s)", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'r' to reset current collection", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Show hand to start collecting", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.putText(frame, f"Phrase: {phrase}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Gesture Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        phrase = ""
    elif key == ord('r') and is_collecting:  # Reset current collection
        print("Cancelled current gesture collection")
        reset_collection_state()
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
