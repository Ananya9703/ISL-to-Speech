import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Data collection parameters
DATA_PATH = 'data_real_time'  # Path to save data
GESTURES = ["Hello", "Bye", "How are you", "I am fine", "Thank you", "Indian", "Sign", "Language", "Woman", "Yesterday", "Today", "Tomorrow", "Swim", "I"]  # List of gestures
START_COOLDOWN = 5  # Cooldown in seconds at the start for the first gesture
FRAMES_PER_GESTURE = 60  # Number of frames to collect per gesture
TRANSITION_COOLDOWN = 5  # Cooldown before transitioning to the next gesture

def balance_dataset():
    """
    Ensure each gesture has the same number of data items.
    """
    print("Balancing dataset...")
    
    # Count current number of samples for each gesture
    gesture_counts = {}
    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_PATH, gesture)
        gesture_counts[gesture] = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
    
    # Find the minimum number of samples
    min_samples = min(gesture_counts.values())
    print("Minimum samples per gesture:", min_samples)
    
    # Trim excess samples
    for gesture, count in gesture_counts.items():
        if count > min_samples:
            gesture_path = os.path.join(DATA_PATH, gesture)
            # Get all .npy and .png files
            npy_files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
            png_files = [f for f in os.listdir(gesture_path) if f.endswith('.png')]
            
            # Sort files to ensure consistent removal
            npy_files.sort()
            png_files.sort()
            
            # Remove excess files
            for file in npy_files[min_samples:]:
                os.remove(os.path.join(gesture_path, file))
            for file in png_files[min_samples:]:
                os.remove(os.path.join(gesture_path, file))
            
            print(f"Trimmed {gesture} from {count} to {min_samples} samples")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 990)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 990)
current_gesture_index = 0  # Start with the first gesture
start_time = time.time()

# Cooldown logic at the start
initial_cooldown_complete = False
frame_count = 0  # Counter for frames collected for the current gesture
cooldown_start_time = None  # Start time for transition cooldown

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # Draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

    # Display instructions
    gesture_name = GESTURES[current_gesture_index]
    if not initial_cooldown_complete:
        # Display countdown
        elapsed_time = int(time.time() - start_time)
        remaining_time = START_COOLDOWN - elapsed_time
        if remaining_time > 0:
            cv2.putText(frame, f"Get ready! Starting in {remaining_time}s...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            initial_cooldown_complete = True  # End cooldown
    else:
        if cooldown_start_time:  # During the transition cooldown
            elapsed_cooldown = time.time() - cooldown_start_time
            if elapsed_cooldown < TRANSITION_COOLDOWN:
                remaining_cooldown = int(TRANSITION_COOLDOWN - elapsed_cooldown)
                cv2.putText(frame, f"Transitioning in {remaining_cooldown}s...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # End cooldown and transition to the next gesture
                frame_count = 0  # Reset frame count for the next gesture
                cooldown_start_time = None
                current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
        else:
            # Display recording status
            cv2.putText(frame, f"Recording Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Save data continuously
            if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                keypoints = []
                # Hand landmarks
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        keypoints.append([landmark.x, landmark.y, landmark.z])

                # Face landmarks
                for face_landmarks in face_results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        keypoints.append([landmark.x, landmark.y, landmark.z])

                keypoints = np.array(keypoints).flatten()
                timestamp = int(time.time() * 1000)  # Use milliseconds for unique filenames
                gesture_folder = os.path.join(DATA_PATH, gesture_name)
                np.save(os.path.join(gesture_folder, f"{gesture_name}_{timestamp}.npy"), keypoints)
                
                frame_count += 1  # Increment the frame counter
                print(f"Saved: {gesture_name}_{timestamp}, Frame: {frame_count}")

            # Automatically start cooldown after collecting specified frames
            if frame_count >= FRAMES_PER_GESTURE and cooldown_start_time is None:
                cooldown_start_time = time.time()  # Start the cooldown timer

    # Display frame
    cv2.imshow('Gesture Collection', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break

# Close everything
cap.release()
cv2.destroyAllWindows()

# Balance the dataset after collection
balance_dataset()

print("Data collection complete. Dataset balanced.")
