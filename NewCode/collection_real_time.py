import cv2
import mediapipe as mp
import os
import numpy as np
import time
import uuid
import collections

# Initialize Mediapipe with less strict parameters
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lower detection confidence
    min_tracking_confidence=0.5    # Lower tracking confidence
)
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,  # Lower detection confidence
    min_tracking_confidence=0.5    # Lower tracking confidence
)
mp_drawing = mp.solutions.drawing_utils

# Data collection parameters
DATA_PATH = 'data_real_time'  # Path to save data
GESTURES = ["Bye", "Hello", "How are you", "I am fine", "Thank you", "Indian", "Sign", "Language", "Woman", "Yesterday", "Today", "Tomorrow", "Swim", "I"]
START_COOLDOWN = 5  # Cooldown in seconds at the start for the first gesture
SEQUENCE_LENGTH = 30  # Number of frames to collect per gesture sequence
SEQUENCES_PER_GESTURE = 10  # Number of sequences to collect per gesture
TRANSITION_COOLDOWN = 5  # Cooldown before transitioning to the next gesture

# Add configuration options at the top
REQUIRE_BOTH_HAND_AND_FACE = False  # Set to False to be more lenient with detection
MIN_LANDMARK_VISIBILITY = 0.3      # Minimum visibility threshold for landmarks
MIN_NUM_HANDS = 1                 # Minimum number of hands required (if REQUIRE_BOTH_HAND_AND_FACE is False)
MAX_NUM_HANDS = 2                 # Maximum number of hands to detect

# Simple moving average filter for landmarks
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = collections.deque(maxlen=window_size)
        self.y_buffer = collections.deque(maxlen=window_size)
        self.z_buffer = collections.deque(maxlen=window_size)
    
    def filter(self, x, y, z, _):
        # Add new values to buffer
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.z_buffer.append(z)
        
        # Calculate moving average
        x_avg = sum(self.x_buffer) / len(self.x_buffer)
        y_avg = sum(self.y_buffer) / len(self.y_buffer)
        z_avg = sum(self.z_buffer) / len(self.z_buffer)
        
        return (x_avg, y_avg, z_avg)

# Create filters for hand and face landmarks
hand_filter = MovingAverageFilter(window_size=3)
face_filter = MovingAverageFilter(window_size=3)

# Motion detection parameters
MIN_MOTION_AREA = 400  # Reduced threshold for motion area
MOTION_DETECTION_THRESHOLD = 15  # Reduced threshold for motion detection
motion_threshold_count = 3  # Reduced number of frames with motion to trigger recording (more sensitive)

def calculate_motion(prev_frame, current_frame):
    """Calculate motion between frames"""
    if prev_frame is None:
        return False, None
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Apply threshold to difference
    _, thresh = cv2.threshold(frame_diff, MOTION_DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour has significant area
    significant_motion = False
    for contour in contours:
        if cv2.contourArea(contour) > MIN_MOTION_AREA:
            significant_motion = True
            break
    
    return significant_motion, dilated

def save_sequence_data(sequence_data, gesture_name):
    """Save a complete sequence of frames"""
    sequence_id = str(uuid.uuid4())[:8]  # Generate a unique ID for the sequence
    gesture_folder = os.path.join(DATA_PATH, gesture_name)
    sequence_folder = os.path.join(gesture_folder, sequence_id)
    os.makedirs(sequence_folder, exist_ok=True)
    
    # Save each frame in the sequence
    for i, frame_data in enumerate(sequence_data):
        keypoints, frame_image = frame_data
        np.save(os.path.join(sequence_folder, f"frame_{i:03d}.npy"), keypoints)
        cv2.imwrite(os.path.join(sequence_folder, f"frame_{i:03d}.png"), frame_image)
    
    print(f"Saved sequence {sequence_id} for gesture {gesture_name} ({len(sequence_data)} frames)")
    return sequence_id

def balance_dataset():
    """Ensure each gesture has the same number of sequences."""
    print("Balancing dataset...")
    
    # Count current number of sequences for each gesture
    gesture_counts = {}
    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            continue
        # Count sequences (folders) in the gesture directory
        gesture_counts[gesture] = len([d for d in os.listdir(gesture_path) 
                                       if os.path.isdir(os.path.join(gesture_path, d))])
    
    if not gesture_counts:
        print("No data collected yet.")
        return
    
    # Find the minimum number of sequences
    min_sequences = min(gesture_counts.values())
    print("Minimum sequences per gesture:", min_sequences)
    
    # Trim excess sequences
    for gesture, count in gesture_counts.items():
        if count > min_sequences:
            gesture_path = os.path.join(DATA_PATH, gesture)
            # Get all sequence directories
            sequence_dirs = [d for d in os.listdir(gesture_path) 
                             if os.path.isdir(os.path.join(gesture_path, d))]
            
            # Sort directories to ensure consistent removal
            sequence_dirs.sort()
            
            # Remove excess sequences
            for dir_name in sequence_dirs[min_sequences:]:
                dir_path = os.path.join(gesture_path, dir_name)
                # Remove all files in the directory
                for file in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file))
                # Remove the directory
                os.rmdir(dir_path)
            
            print(f"Trimmed {gesture} from {count} to {min_sequences} sequences")

# Create directories
os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 990)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 990)
cap.set(cv2.CAP_PROP_FPS, 30)  # Try to get 30 fps

current_gesture_index = 0
start_time = time.time()
initial_cooldown_complete = False
sequence_count = 0  # Number of sequences recorded for current gesture
cooldown_start_time = None
is_recording_sequence = False
sequence_data = []
prev_frame = None
motion_history = []  # To track recent motion
motion_threshold_count = 5  # Number of frames with motion to trigger recording

print(f"Starting data collection for {len(GESTURES)} gestures...")
print(f"Each gesture will have {SEQUENCES_PER_GESTURE} sequences of {SEQUENCE_LENGTH} frames each.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()  # Keep the original frame for display and saving
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Noise reduction on the mask
    fgmask = cv2.medianBlur(fgmask, 5)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    # Check for motion between frames
    has_motion, motion_mask = calculate_motion(prev_frame, frame)
    prev_frame = frame.copy()
    
    # Update motion history
    motion_history.append(has_motion)
    if len(motion_history) > motion_threshold_count:
        motion_history.pop(0)
    
    # Decide if there's consistent motion
    consistent_motion = sum(motion_history) >= motion_threshold_count // 2
    
    # Convert to RGB for processing with MediaPipe
    rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    # Check if we have any landmarks (less strict requirement)
    has_valid_landmarks = (hand_results.multi_hand_landmarks is not None or 
                          face_results.multi_face_landmarks is not None)
    
    # Draw landmarks on the original frame for display
    display_frame = original_frame.copy()
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(display_frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
    
    current_time = time.time()
    gesture_name = GESTURES[current_gesture_index]
    
    # Display instructions based on the current state
    if not initial_cooldown_complete:
        # Display initial countdown
        elapsed_time = int(current_time - start_time)
        remaining_time = START_COOLDOWN - elapsed_time
        if remaining_time > 0:
            cv2.putText(display_frame, f"Get ready! Starting in {remaining_time}s...", (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            initial_cooldown_complete = True
    else:
        # Check if we're in transition cooldown between gestures
        if cooldown_start_time:
            elapsed_cooldown = current_time - cooldown_start_time
            if elapsed_cooldown < TRANSITION_COOLDOWN:
                remaining_cooldown = int(TRANSITION_COOLDOWN - elapsed_cooldown)
                cv2.putText(display_frame, f"Next gesture: {gesture_name} in {remaining_cooldown}s...", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # End cooldown and reset for the next gesture
                cooldown_start_time = None
                is_recording_sequence = False
                sequence_data = []
                sequence_count = 0
        else:
            # Display recording status
            cv2.putText(display_frame, f"Gesture: {gesture_name} ({sequence_count+1}/{SEQUENCES_PER_GESTURE})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if is_recording_sequence:
                cv2.putText(display_frame, f"RECORDING SEQUENCE: {len(sequence_data)}/{SEQUENCE_LENGTH}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # If we have valid landmarks, collect the data
                if has_valid_landmarks:
                    keypoints = []
                    # Process hand landmarks
                    if hand_results.multi_hand_landmarks:  # Check if not None before iterating
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                x, y, z = hand_filter.filter(landmark.x, landmark.y, landmark.z, current_time)
                                keypoints.extend([x, y, z])
                    
                    # Process face landmarks
                    if face_results.multi_face_landmarks:  # Check if not None before iterating
                        for face_landmarks in face_results.multi_face_landmarks:
                            for landmark in face_landmarks.landmark:
                                x, y, z = face_filter.filter(landmark.x, landmark.y, landmark.z, current_time)
                                keypoints.extend([x, y, z])
                    
                    keypoints = np.array(keypoints).flatten()
                    sequence_data.append((keypoints, original_frame))
                    
                    # Check if we've collected enough frames
                    if len(sequence_data) >= SEQUENCE_LENGTH:
                        # Save the sequence
                        save_sequence_data(sequence_data, gesture_name)
                        sequence_count += 1
                        sequence_data = []
                        is_recording_sequence = False
                        
                        # Check if we've collected enough sequences for this gesture
                        if sequence_count >= SEQUENCES_PER_GESTURE:
                            # Start transition to next gesture
                            cooldown_start_time = current_time
                            current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
                            if current_gesture_index == 0:
                                print("Completed one full cycle of all gestures.")
            else:
                # Not recording yet, wait for consistent motion to start
                if consistent_motion and has_valid_landmarks:
                    cv2.putText(display_frame, "Motion detected! Starting sequence...", 
                              (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    is_recording_sequence = True
                    sequence_data = []  # Clear any old data
                    
                    # Reset filters when starting a new sequence
                    hand_filter = MovingAverageFilter(window_size=3)
                    face_filter = MovingAverageFilter(window_size=3)
                elif consistent_motion:
                    cv2.putText(display_frame, "Motion detected but no landmarks found", 
                              (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "Perform the gesture with movement to start recording", 
                              (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Show motion detection visualization in corner
    if motion_mask is not None:
        motion_mask_small = cv2.resize(motion_mask, (160, 120))
        motion_mask_color = cv2.cvtColor(motion_mask_small, cv2.COLOR_GRAY2BGR)
        display_frame[10:130, display_frame.shape[1]-170:display_frame.shape[1]-10] = motion_mask_color
    
    # Display progress bar for sequence collection
    if is_recording_sequence:
        progress = len(sequence_data) / SEQUENCE_LENGTH
        bar_width = 400
        bar_height = 20
        filled_width = int(bar_width * progress)
        
        # Draw background
        cv2.rectangle(display_frame, (10, display_frame.shape[0] - 40), 
                    (10 + bar_width, display_frame.shape[0] - 40 + bar_height), 
                    (100, 100, 100), -1)
        
        # Draw filled portion
        cv2.rectangle(display_frame, (10, display_frame.shape[0] - 40), 
                    (10 + filled_width, display_frame.shape[0] - 40 + bar_height), 
                    (0, 255, 0), -1)
    
    # Display frame
    cv2.imshow('Dynamic Sign Language Collection', display_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s') and not is_recording_sequence:  # Skip current gesture
        print(f"Skipping gesture: {gesture_name}")
        current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
        cooldown_start_time = None
        sequence_count = 0

# Close everything
cap.release()
cv2.destroyAllWindows()

# Balance the dataset
balance_dataset()

print("Data collection complete and dataset balanced!")
print(f"Data saved to {os.path.abspath(DATA_PATH)}")