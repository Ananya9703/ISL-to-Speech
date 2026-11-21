import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import collections
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Configuration parameters
MODEL_PATH = 'gesture_model_lstm_final2.keras'  # or 'gesture_model_dense.keras'
SCALER_PATH = 'gesture_scaler_lstm2.joblib'
LABEL_ENCODER_PATH = 'label_encoder_classes_lstm2.npy'
SEQUENCE_LENGTH = 30  # Must match the sequence length used during training
PREDICTION_THRESHOLD = 0.7  # Confidence threshold for displaying predictions
MODEL_TYPE = 'lstm'  # 'lstm' or 'dense' - should match your loaded model

# Simple moving average filter for landmarks
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = collections.deque(maxlen=window_size)
        self.y_buffer = collections.deque(maxlen=window_size)
        self.z_buffer = collections.deque(maxlen=window_size)
    
    def filter(self, x, y, z):
        # Add new values to buffer
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.z_buffer.append(z)
        
        # Calculate moving average
        x_avg = sum(self.x_buffer) / len(self.x_buffer)
        y_avg = sum(self.y_buffer) / len(self.y_buffer)
        z_avg = sum(self.z_buffer) / len(self.z_buffer)
        
        return (x_avg, y_avg, z_avg)

# Prediction result class to handle prediction history
class PredictionHistory:
    def __init__(self, max_size=10):
        self.history = deque(maxlen=max_size)
        self.class_counts = {}
        self.max_size = max_size
    
    def add(self, prediction, confidence):
        self.history.append((prediction, confidence))
        
        # Update class counts
        self.class_counts = {}
        for pred, conf in self.history:
            if pred not in self.class_counts:
                self.class_counts[pred] = 0
            self.class_counts[pred] += conf  # Weight by confidence
    
    def get_top_prediction(self):
        if not self.class_counts:
            return None, 0
        
        # Get prediction with highest weighted count
        top_prediction = max(self.class_counts.items(), key=lambda x: x[1])
        top_class = top_prediction[0]
        weighted_confidence = top_prediction[1] / sum([conf for _, conf in self.history])
        
        return top_class, weighted_confidence

def load_model_and_dependencies():
    """Load the trained model, scaler, and label encoder"""
    print("Loading model and dependencies...")
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
        
        # Load the scaler
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
        
        # Load the label encoder classes
        label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
        print(f"Loaded {len(label_classes)} gesture classes")
        
        # Get max feature length from model input shape
        if MODEL_TYPE == 'lstm':
            _, seq_len, feature_len = model.input_shape
        else:  # dense model
            feature_len = model.input_shape[1] // SEQUENCE_LENGTH
        
        print(f"Feature length: {feature_len}")
        return model, scaler, label_classes, feature_len
    
    except Exception as e:
        print(f"Error loading model dependencies: {str(e)}")
        raise

def process_frame(frame, model, scaler, label_classes, feature_len, frame_queue, prediction_history):
    """Process a single frame and update the prediction queue"""
    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Create a copy for display
    display_frame = frame.copy()
    
    # Apply background subtraction (optional, for better isolation)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    fgmask = fgbg.apply(frame)
    
    # Noise reduction on the mask
    fgmask = cv2.medianBlur(fgmask, 5)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    # Convert to RGB for processing with MediaPipe
    rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    # Check if we have any landmarks
    has_landmarks = (hand_results.multi_hand_landmarks is not None or 
                    face_results.multi_face_landmarks is not None)
    
    # Draw landmarks on the display frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(display_frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
    
    # Extract landmarks
    if has_landmarks:
        # Initialize filters for smoothing
        hand_filter = MovingAverageFilter(window_size=3)
        face_filter = MovingAverageFilter(window_size=3)
        
        keypoints = []
        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y, z = hand_filter.filter(landmark.x, landmark.y, landmark.z)
                    keypoints.extend([x, y, z])
        
        # Process face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y, z = face_filter.filter(landmark.x, landmark.y, landmark.z)
                    keypoints.extend([x, y, z])
        
        # Convert to numpy array
        keypoints = np.array(keypoints, dtype=np.float32).flatten()
        
        # Pad or truncate to match expected feature length
        padded_keypoints = np.zeros(feature_len, dtype=np.float32)
        padded_keypoints[:min(len(keypoints), feature_len)] = keypoints[:min(len(keypoints), feature_len)]
        
        # Add to frame queue
        frame_queue.append(padded_keypoints)
        
        # Check if we have enough frames for a prediction
        if len(frame_queue) >= SEQUENCE_LENGTH:
            # Keep only the most recent frames
            while len(frame_queue) > SEQUENCE_LENGTH:
                frame_queue.popleft()
            
            # Create sequence from frame queue
            sequence = np.array(frame_queue)
            
            # Prepare data based on model type
            if MODEL_TYPE == 'lstm':
                # Reshape for LSTM (samples, sequence_length, features)
                input_data = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Apply scaling - reshape to flatten, scale, then reshape back
                orig_shape = input_data.shape
                input_data_flat = input_data.reshape(-1, input_data.shape[-1])
                input_data_scaled = scaler.transform(input_data_flat)
                input_data = input_data_scaled.reshape(orig_shape)
            else:  # dense model
                # Flatten sequence for dense model
                input_data = sequence.reshape(1, -1)
                
                # Apply scaling
                input_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data, verbose=0)
            pred_class_idx = np.argmax(prediction[0])
            pred_class = label_classes[pred_class_idx]
            pred_confidence = prediction[0][pred_class_idx]
            
            # Add to prediction history
            prediction_history.add(pred_class, pred_confidence)
            
            # Get stable prediction
            stable_pred, stable_conf = prediction_history.get_top_prediction()
            
            # Only show predictions above threshold
            if stable_conf > PREDICTION_THRESHOLD:
                # Display the stable prediction
                cv2.putText(display_frame, f"Gesture: {stable_pred}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Confidence: {stable_conf:.2f}", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Waiting for clear gesture...", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        cv2.putText(display_frame, "No hand or face detected", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return display_frame

def main():
    """Main function for real-time gesture recognition"""
    try:
        # Load model and dependencies
        model, scaler, label_classes, feature_len = load_model_and_dependencies()
        
        # Initialize frame queue and prediction history
        frame_queue = deque(maxlen=SEQUENCE_LENGTH)
        prediction_history = PredictionHistory(max_size=15)
        
        # Initialize webcam
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS calculation variables
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        print("Starting real-time prediction. Press 'q' to quit.")
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            display_frame = process_frame(frame, model, scaler, label_classes, 
                                         feature_len, frame_queue, prediction_history)
            
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                      (display_frame.shape[1] - 120, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Press 'q' to quit", 
                      (10, display_frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Real-time Gesture Recognition", display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()