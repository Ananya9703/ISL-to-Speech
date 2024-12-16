import os
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler('hand_landmark_extraction.log'),
                        logging.StreamHandler()
                    ])

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with improved configuration
hands = mp_hands.Hands(
    static_image_mode=False,  # More efficient processing
    min_detection_confidence=0.5,  # Increased detection confidence
    min_tracking_confidence=0.5,   # Added tracking confidence
    max_num_hands=2  # Allow detection of up to 2 hands
)

# Configuration
DATA_DIR = r'C:\Users\rhuch\OneDrive\Documents (1)\College\ISL_Project_Static\custom_dataset'
OUTPUT_CSV = r'C:\Users\rhuch\OneDrive\Documents (1)\College\ISL_Project_Static\csv_dataset\custom_keypoints.csv'

# Constants
NUM_LANDMARKS = 21  # MediaPipe Hands has 21 landmarks per hand
FEATURES_PER_LANDMARK = 2  # x and y coordinates

def normalize_landmarks(landmarks):
    """
    Improved landmark normalization with robust handling
    """
    # Check if landmarks are valid
    if not landmarks or len(landmarks) == 0:
        return [0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK)
    
    try:
        # Extract x and y coordinates
        x_ = [lm.x for lm in landmarks]
        y_ = [lm.y for lm in landmarks]
        
        # Handle edge cases
        if not x_ or not y_:
            return [0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK)
        
        # Compute normalization parameters
        min_x, min_y = min(x_), min(y_)
        max_x, max_y = max(x_), max(y_)
        
        # Compute width and height for scaling
        width = max_x - min_x
        height = max_y - min_y
        
        # Prevent division by zero
        width = width if width > 0 else 1
        height = height if height > 0 else 1
        
        # Normalize landmarks
        normalized_landmarks = []
        for lm in landmarks:
            normalized_x = (lm.x - min_x) / width
            normalized_y = (lm.y - min_y) / height
            normalized_landmarks.extend([normalized_x, normalized_y])
        
        return normalized_landmarks
    
    except Exception as e:
        logging.error(f"Error in landmark normalization: {e}")
        return [0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK)

def process_image(full_img_path):
    """
    Process a single image and extract hand landmarks
    """
    try:
        # Read the image
        img = cv2.imread(full_img_path)
        if img is None:
            logging.warning(f"Could not read image: {full_img_path}")
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = hands.process(img_rgb)
        
        # Detailed logging of hand detection
        if results.multi_hand_landmarks:
            logging.info(f"Hands detected in {os.path.basename(full_img_path)}:")
            for idx, (handedness, landmarks) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
                logging.info(f"Hand {idx+1}:")
                logging.info(f"  Handedness: {handedness.classification[0].label}")
                logging.info(f"  Detection score: {handedness.classification[0].score}")
                logging.info(f"  Landmark count: {len(landmarks.landmark)}")
        else:
            logging.info(f"No hands detected in {os.path.basename(full_img_path)}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error processing {full_img_path}: {e}")
        return None

def extract_landmarks(results, label):
    """
    Extract and normalize landmarks from MediaPipe results
    """
    if not results or not results.multi_hand_landmarks:
        return None

    # Validate handedness information
    if len(results.multi_handedness) != len(results.multi_hand_landmarks):
        logging.warning("Mismatch in handedness and landmarks")
        return None

    # Sort hands by handedness (left first, then right)
    hands_info = sorted(
        zip(results.multi_handedness, results.multi_hand_landmarks), 
        key=lambda x: x[0].classification[0].label
    )

    normalized_landmarks = []
    for handedness, landmarks in hands_info:
        # Only process if detection confidence is good
        if handedness.classification[0].score > 0.5:
            norm_landmarks = normalize_landmarks(landmarks.landmark)
            normalized_landmarks.extend(norm_landmarks)
        else:
            # Add zero landmarks if detection is poor
            normalized_landmarks.extend([0] * (NUM_LANDMARKS * FEATURES_PER_LANDMARK))

    # Ensure consistent number of features
    if len(normalized_landmarks) == NUM_LANDMARKS * FEATURES_PER_LANDMARK * 2:
        return normalized_landmarks
    else:
        logging.warning(f"Inconsistent landmarks for label {label}")
        return None

def main():
    # Lists to store all data and labels
    data_list = []
    labels_list = []

    # Track processing statistics
    total_images = 0
    processed_images = 0

    # Iterate through directories (classes)
    for dir_ in os.listdir(DATA_DIR):
        class_dir_path = os.path.join(DATA_DIR, dir_)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir_path):
            continue

        # Iterate through images in each directory
        for img_path in os.listdir(class_dir_path):
            total_images += 1
            full_img_path = os.path.join(class_dir_path, img_path)
            
            # Process the image
            results = process_image(full_img_path)
            
            # Extract landmarks
            if results and results.multi_hand_landmarks:
                landmarks = extract_landmarks(results, dir_)
                
                if landmarks:
                    data_list.append(landmarks)
                    labels_list.append(dir_)
                    processed_images += 1

    # Convert to numpy array
    if data_list:
        data_array = np.array(data_list)

        # Create DataFrame
        df = pd.DataFrame(data_array)

        # Add label column
        df['label'] = labels_list

        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)

        # Log processing summary
        logging.info(f"Total images processed: {total_images}")
        logging.info(f"Successfully processed images: {processed_images}")
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"Unique labels: {set(labels_list)}")
        logging.info(f"Features per sample: {len(data_list[0]) if data_list else 0}")
    else:
        logging.error("No landmarks were extracted. Please check your dataset.")

if __name__ == "__main__":
    main()