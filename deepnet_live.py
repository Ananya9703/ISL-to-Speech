import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque
import joblib

# Load the trained gesture recognition model
model = load_model('gesture_deepnet_model.h5')

# Load label encoder for decoding predictions
gesture_labels = np.load('label_encoder.npy', allow_pickle=True)

# Load pre-trained StandardScaler
try:
    scaler = joblib.load('gesture_scaler.joblib')
except FileNotFoundError:
    print("Error: Scaler file not found. Please run training script first.")
    exit()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.96, min_tracking_confidence=0.96)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.96, min_tracking_confidence=0.96)
mp_drawing = mp.solutions.drawing_utils

# Define expected input length (adjusted for model requirements)
hand_keypoints = 21 * 3  # x, y, z for 21 hand landmarks
face_keypoints = 468 * 3  # x, y, z for 468 face landmarks
actual_length = hand_keypoints + face_keypoints  # 1467 features
expected_length = 1593  # Model expects this size

# Initialize webcam
cap = cv2.VideoCapture(0)

# Phrase buffer for constructing text
buffer = deque(maxlen=10)
phrase = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # Collect keypoints
    keypoints = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

    if len(keypoints) > 0:
        # Ensure consistent keypoint length
        if len(keypoints) < expected_length:
            keypoints = np.pad(keypoints, (0, expected_length - len(keypoints)), mode='constant')
        elif len(keypoints) > expected_length:
            keypoints = keypoints[:expected_length]

        # Reshape for prediction
        keypoints = np.array(keypoints).reshape(1, -1)  # Shape (1, 1593)

        # Standardize the input data
        keypoints_scaled = scaler.transform(keypoints)

        # Predict gesture
        predictions = model.predict(keypoints_scaled)
        gesture_idx = np.argmax(predictions)

        # Decode gesture label
        if 0 <= gesture_idx < len(gesture_labels):
            gesture_label = gesture_labels[gesture_idx]
        else:
            gesture_label = "Unknown"

        # Add gesture to buffer
        buffer.append(gesture_label)

        # Construct phrase
        if len(buffer) > 1 and buffer[-1] != buffer[-2]:
            phrase += f" {gesture_label}"

        # Display predicted gesture
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

    # Display constructed phrase
    cv2.putText(frame, f"Phrase: {phrase}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Gesture Recognition', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear phrase
        phrase = ""

cap.release()
cv2.destroyAllWindows()