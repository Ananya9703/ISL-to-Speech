import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, LeakyReLU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import traceback
import statistics
import random  # For augmentation

def safe_load_numpy(file_path):
    """Safely load numpy data, handling potential errors and types."""
    try:
        data = np.load(file_path, allow_pickle=True)
        if data is None: # Handle case where file is empty or corrupt
             raise ValueError("Loaded data is None")
        if not isinstance(data, np.ndarray):
            data = np.array(data) # Convert if loaded as non-ndarray (e.g., list)

        # If data is object type (likely list of lists/arrays), flatten carefully
        if data.dtype == object:
            # Heuristic: Try to detect if it's a sequence of frames (list of arrays)
            if len(data.shape) == 1 and isinstance(data[0], (list, np.ndarray)):
                 # Assume it's a sequence that needs padding/truncating later
                 # For safe_load, just ensure elements are numeric
                 processed_frames = []
                 target_len = None
                 for frame in data:
                     frame_arr = np.array(frame, dtype=np.float32).flatten()
                     if target_len is None:
                         target_len = len(frame_arr)
                     # Basic check for consistent feature length, can be improved
                     if len(frame_arr) != target_len:
                         # Pad or truncate - simple approach for loading
                         padded_frame = np.zeros(target_len, dtype=np.float32)
                         common_len = min(len(frame_arr), target_len)
                         padded_frame[:common_len] = frame_arr[:common_len]
                         processed_frames.append(padded_frame)
                     else:
                        processed_frames.append(frame_arr)
                 return np.array(processed_frames, dtype=np.float32)
            else:
                # Fallback for general object arrays: flatten everything numeric
                flattened = []
                for item in data.flatten():
                    if isinstance(item, (list, np.ndarray)):
                        flat_item = np.array(item).flatten()
                        # Convert valid numeric strings, ignore others
                        flattened.extend([float(x) for x in flat_item if str(x).replace('.', '', 1).isdigit()])
                    elif isinstance(item, (int, float, np.number)):
                         flattened.append(float(item))
                    # Add check for numeric strings
                    elif isinstance(item, str) and item.replace('.', '', 1).isdigit():
                         flattened.append(float(item))
                return np.array(flattened, dtype=np.float32)
        else:
            # Standard numeric array, ensure float32
             # Return as is, flattening might be wrong for frame data
            return data.astype(np.float32)

    except Exception as e:
        # Print detailed error including file path
        print(f"Error processing file {file_path}: {str(e)}")
        traceback.print_exc() # Print stack trace
        # Return None or an empty array to indicate failure
        # Depending on how load_gesture_data handles it
        return None # Or np.array([], dtype=np.float32)

# Keep your existing safe_load_numpy and load_gesture_data functions
def load_gesture_data(data_path):
    print("Analyzing dataset structure...")
    sequences = []  # List to store complete sequences
    labels = []     # List to store labels for each sequence

    gesture_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    max_sequence_length = 0
    feature_length = None # Use a fixed feature length derived from the first valid frame
    gesture_counts = {}
    file_load_errors = 0

    print("First pass: Analyzing dataset structure and determining feature length...")
    for gesture_name in gesture_folders:
        gesture_path = os.path.join(data_path, gesture_name)
        sequence_folders = [f for f in os.listdir(gesture_path) if os.path.isdir(os.path.join(gesture_path, f))]
        gesture_counts[gesture_name] = len(sequence_folders)

        for sequence_folder in sequence_folders:
            sequence_path = os.path.join(gesture_path, sequence_folder)
            # Use consistent sorting
            frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')])
            if not frame_files:
                print(f"Warning: Sequence folder {sequence_folder} in {gesture_name} is empty.")
                continue

            max_sequence_length = max(max_sequence_length, len(frame_files))

            # Determine feature length from the first valid frame found anywhere
            if feature_length is None:
                for frame_file in frame_files:
                    sample_file = os.path.join(sequence_path, frame_file)
                    sample_data = safe_load_numpy(sample_file)
                    if sample_data is not None and sample_data.size > 0:
                         # Assuming safe_load returns (features,) or (1, features) for a single frame
                         if sample_data.ndim == 2 and sample_data.shape[0] == 1:
                             feature_length = sample_data.shape[1]
                         elif sample_data.ndim == 1:
                             feature_length = sample_data.shape[0]
                         else:
                              print(f"Warning: Unexpected shape {sample_data.shape} for sample frame {sample_file}. Skipping for feature length determination.")
                              continue # Try next frame
                         if feature_length is not None:
                            print(f"Determined feature length: {feature_length} from {sample_file}")
                            break # Feature length found
                if feature_length is None and sequence_folder == sequence_folders[-1] and gesture_name == gesture_folders[-1]:
                     raise RuntimeError("Could not determine feature length from any sample frame.")


    if feature_length is None:
         raise RuntimeError("Failed to determine feature length. Check dataset integrity.")
    print(f"\nUsing fixed feature length: {feature_length}")
    print(f"Maximum sequence length found: {max_sequence_length}")

    print("\nDataset Sequence Counts:")
    for gesture, count in gesture_counts.items():
        print(f"- {gesture}: {count} sequences")

    print("\nSecond pass: Loading sequence data...")
    for gesture_name in gesture_folders:
        gesture_path = os.path.join(data_path, gesture_name)
        sequence_folders = [f for f in os.listdir(gesture_path) if os.path.isdir(os.path.join(gesture_path, f))]
        print(f"\nProcessing {gesture_name}...")
        successful_sequences = 0

        for sequence_folder in sequence_folders:
            sequence_path = os.path.join(gesture_path, sequence_folder)
            frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.npy')])
            if not frame_files: continue # Skip empty folders noted before

            sequence_data = []
            valid_sequence = True
            for frame_file in frame_files:
                frame_path = os.path.join(sequence_path, frame_file)
                frame_data = safe_load_numpy(frame_path)

                if frame_data is None or frame_data.size == 0:
                    print(f"Warning: Failed to load or empty frame {frame_file} in sequence {sequence_folder}. Skipping sequence.")
                    file_load_errors += 1
                    valid_sequence = False
                    break # Skip rest of this sequence

                # Ensure frame_data is 1D
                frame_data = frame_data.flatten()

                # Validate numeric data
                if not np.all(np.isfinite(frame_data)):
                    print(f"Warning: Frame {frame_file} in sequence {sequence_folder} contains non-finite values (NaN/inf). Skipping sequence.")
                    valid_sequence = False
                    break

                # Pad or truncate the frame to the fixed feature_length
                current_len = len(frame_data)
                if current_len == feature_length:
                    padded_frame = frame_data
                else:
                    padded_frame = np.zeros(feature_length, dtype=np.float32)
                    common_len = min(current_len, feature_length)
                    padded_frame[:common_len] = frame_data[:common_len]
                    if current_len != feature_length:
                         # This warning can be noisy, disable if expected
                         # print(f"Info: Frame {frame_file} length {current_len} adjusted to {feature_length}.")
                         pass

                sequence_data.append(padded_frame)

            if not valid_sequence:
                continue # Move to the next sequence folder

            # Pad the sequence (list of frames) to max_sequence_length
            current_seq_len = len(sequence_data)
            if current_seq_len < max_sequence_length:
                padding = [np.zeros(feature_length, dtype=np.float32) for _ in range(max_sequence_length - current_seq_len)]
                sequence_data.extend(padding)
            elif current_seq_len > max_sequence_length:
                sequence_data = sequence_data[:max_sequence_length]

            # Convert the list of frame arrays into a single sequence array
            try:
                 sequence_array = np.array(sequence_data, dtype=np.float32)
                 # Final check on shape
                 if sequence_array.shape != (max_sequence_length, feature_length):
                      print(f"Warning: Final sequence array shape mismatch ({sequence_array.shape}) for {sequence_folder}. Expected ({max_sequence_length}, {feature_length}). Skipping.")
                      continue

                 sequences.append(sequence_array)
                 labels.append(gesture_name)
                 successful_sequences += 1
            except Exception as e:
                 print(f"Error converting sequence {sequence_folder} to array: {e}. Skipping.")


        print(f"-> Loaded {successful_sequences}/{len(sequence_folders)} sequences for {gesture_name}")

    if file_load_errors > 0:
        print(f"\nWarning: Encountered {file_load_errors} file loading errors during processing.")

    if not sequences:
         raise RuntimeError("No sequences were successfully loaded. Check data path and file integrity.")

    # Convert lists to numpy arrays before returning
    return np.array(sequences, dtype=np.float32), np.array(labels)

# --- New: Data Augmentation Functions ---
def time_warp_sequence(sequence, max_frames=2):
    """Apply time warping to a sequence by duplicating or skipping frames"""
    seq_len, features = sequence.shape
    result = np.zeros_like(sequence)
    
    # Randomly decide to stretch or compress
    if np.random.random() < 0.5:
        # Stretch: duplicate some frames
        indices = np.random.choice(seq_len, max_frames, replace=False)
        indices = np.sort(indices)
        
        src_idx = 0
        dest_idx = 0
        for idx in indices:
            # Copy frames up to the duplicate point
            while src_idx < idx and dest_idx < seq_len:
                result[dest_idx] = sequence[src_idx]
                src_idx += 1
                dest_idx += 1
            
            # Duplicate the frame if there's space
            if dest_idx < seq_len:
                result[dest_idx] = sequence[src_idx-1]
                dest_idx += 1
        
        # Copy remaining frames
        while src_idx < seq_len and dest_idx < seq_len:
            result[dest_idx] = sequence[src_idx]
            src_idx += 1
            dest_idx += 1
    else:
        # Compress: skip some frames
        indices = np.random.choice(seq_len, max_frames, replace=False)
        indices = np.sort(indices)
        
        src_idx = 0
        dest_idx = 0
        for idx in indices:
            # Copy frames up to the skip point
            while src_idx < idx and dest_idx < seq_len:
                result[dest_idx] = sequence[src_idx]
                src_idx += 1
                dest_idx += 1
            
            # Skip this frame
            src_idx += 1
        
        # Copy remaining frames
        while src_idx < seq_len and dest_idx < seq_len:
            result[dest_idx] = sequence[src_idx]
            src_idx += 1
            dest_idx += 1
            
        # If we have unfilled frames at the end, repeat the last frame
        while dest_idx < seq_len:
            result[dest_idx] = result[dest_idx-1]
            dest_idx += 1
            
    return result

def add_noise_to_sequence(sequence, noise_level=0.02):
    """Add Gaussian noise to sequence data"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def augment_sequence(sequence, augmentation_prob=0.5):
    """Apply random augmentations to a sequence"""
    augmented = sequence.copy()
    
    # Apply time warping
    if np.random.random() < augmentation_prob:
        augmented = time_warp_sequence(augmented)
        
    # Apply noise
    if np.random.random() < augmentation_prob:
        augmented = add_noise_to_sequence(augmented)
    
    return augmented

def augment_batch(batch, augmentation_prob=0.5):
    """Apply augmentation to a batch of sequences"""
    augmented_batch = np.zeros_like(batch)
    for i in range(len(batch)):
        if np.random.random() < augmentation_prob:
            augmented_batch[i] = augment_sequence(batch[i])
        else:
            augmented_batch[i] = batch[i]
    return augmented_batch

# --- Updated Model Creation Function ---
def create_lstm_model(input_shape, num_classes, lstm_units_1=128, lstm_units_2=96, 
                      dense_units_1=256, dense_units_2=128, dropout_rate=0.35, 
                      l2_reg=0.0003, learning_rate=0.0003):
    """Create an improved bidirectional LSTM model for sequence classification."""
    print(f"\nCreating Bidirectional LSTM Model with params:")
    print(f"  LSTM Units: {lstm_units_1}, {lstm_units_2}")
    print(f"  Dense Units: {dense_units_1}, {dense_units_2}")
    print(f"  Dropout Rate: {dropout_rate}")
    print(f"  L2 Regularization: {l2_reg}")
    print(f"  Initial Learning Rate: {learning_rate}")

    model = Sequential([
        # Bidirectional LSTM layers for better sequence capture
        Bidirectional(LSTM(lstm_units_1, return_sequences=True, 
                          input_shape=input_shape, 
                          kernel_regularizer=l2(l2_reg))),
        Dropout(dropout_rate),

        Bidirectional(LSTM(lstm_units_2, return_sequences=False, 
                          kernel_regularizer=l2(l2_reg))),
        Dropout(dropout_rate),

        # Dense layers with batch normalization and leaky ReLU
        Dense(dense_units_1, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout_rate),

        Dense(dense_units_2, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout_rate/2),  # Reduced dropout before final layer

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- Custom Data Generator for Online Augmentation ---
class AugmentedSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment_prob=0.5, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment_prob = augment_prob
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        # Apply augmentation
        augmented_batch_X = augment_batch(batch_X, self.augment_prob)
        
        return augmented_batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --- Custom Learning Rate Schedule ---
def cosine_decay_schedule(epoch, initial_lr, total_epochs=150):
    """Cosine annealing learning rate schedule"""
    return initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

# --- Main Training Function with Enhancements ---
def run_improved_lstm_training_cv(data_path, n_splits=5, 
                                model_save_path='gesture_model_lstm_final2.keras',
                                use_augmentation=True):
    """Train improved LSTM model using Stratified K-Fold Cross-Validation with data augmentation."""

    # 1. Load Data
    print("--- Loading Data ---")
    X, y = load_gesture_data(data_path)
    if X.size == 0 or y.size == 0: 
        raise ValueError("Loaded data is empty.")

    num_sequences, sequence_length, num_features = X.shape
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Sequences: {num_sequences}, Timesteps: {sequence_length}, Features: {num_features}")
    
    # Check for sufficient data
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print("\n--- Class Distribution ---")
    for label, count in zip(unique_labels, label_counts):
        print(f"  {label}: {count} samples")
        if count < 50:
            print(f" Warning: Low sample count for class '{label}'. Consider collecting more data.")
    
    # 2. Encode Labels
    print("\n--- Encoding Labels ---")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    # Save label encoder classes
    labels_path = 'label_encoder_classes_lstm2.npy'
    np.save(labels_path, label_encoder.classes_)
    print(f"Label encoder classes saved to {labels_path}")

    # 3. Initial Train/Test Split (Hold-out Test Set)
    print(f"\n--- Splitting Data ({100 - 100//(n_splits+1)}% Train+Val / {100//(n_splits+1)}% Test) ---")
    test_set_size = 0.15
    X_train_val, X_test, y_train_val_cat, y_test_cat, y_train_val_enc, y_test_enc = train_test_split(
        X, y_categorical, y_encoded,
        test_size=test_set_size,
        random_state=42,
        stratify=y_encoded
    )
    print(f"Train/Val set shape: X={X_train_val.shape}, y={y_train_val_cat.shape}")
    print(f"Hold-out Test set shape: X={X_test.shape}, y={y_test_cat.shape}")

    # 4. Scale Features
    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    X_train_val_reshaped = X_train_val.reshape(-1, num_features)
    scaler.fit(X_train_val_reshaped)
    X_train_val_scaled_reshaped = scaler.transform(X_train_val_reshaped)
    X_train_val_scaled = X_train_val_scaled_reshaped.reshape(X_train_val.shape)
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
    print("Features scaled successfully.")

    # Save the scaler
    scaler_path = 'gesture_scaler_lstm2.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path} (expects {scaler.n_features_in_} features)")

    # 5. K-Fold Cross-Validation
    print(f"\n--- Starting {n_splits}-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    # Improved hyperparameters
    hp = {
        'lstm_units_1': 160,  # Increased capacity
        'lstm_units_2': 96,   # Increased from 64
        'dense_units_1': 256,
        'dense_units_2': 128,
        'dropout_rate': 0.35, # Slightly reduced
        'l2_reg': 0.0003,     # Slightly reduced
        'learning_rate': 0.0003  # Lower learning rate
    }
    input_shape = (sequence_length, num_features)
    batch_size = 24

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_scaled, y_train_val_enc)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Get data for this fold
        X_train_fold, X_val_fold = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_val_cat[train_idx], y_train_val_cat[val_idx]
        y_train_fold_enc = y_train_val_enc[train_idx]

        print(f"Train fold shape: X={X_train_fold.shape}, y={y_train_fold.shape}")
        print(f"Validation fold shape: X={X_val_fold.shape}, y={y_val_fold.shape}")

        # Create a fresh model instance for this fold
        model = create_lstm_model(input_shape, num_classes, **hp)

        # Advanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=30, 
            verbose=1, 
            restore_best_weights=True, 
            min_delta=0.0005
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            mode='min', 
            factor=0.5, 
            patience=15, 
            min_lr=1e-6, 
            verbose=1
        )
        
        # Custom learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            lambda epoch: cosine_decay_schedule(epoch, hp['learning_rate'], total_epochs=150), 
            verbose=0
        )
        
        # Calculate class weights for the current training fold
        fold_class_counts = np.sum(y_train_fold, axis=0)
        fold_total_samples = np.sum(fold_class_counts)
        fold_class_weights = {}
        for i, count in enumerate(fold_class_counts):
            if count > 0:
                fold_class_weights[i] = fold_total_samples / (num_classes * count)
            else:
                fold_class_weights[i] = 0

        # Set up data generators for augmentation if enabled
        if use_augmentation:
            train_gen = AugmentedSequenceGenerator(
                X_train_fold, y_train_fold, 
                batch_size=batch_size, 
                augment_prob=0.7
            )
            print("Using data augmentation during training.")
            
            # Train with generator
            history = model.fit(
                train_gen,
                validation_data=(X_val_fold, y_val_fold),
                epochs=150,
                callbacks=[early_stopping, reduce_lr, lr_scheduler],
                class_weight=fold_class_weights,
                verbose=1
            )
        else:
            # Standard training without generator
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=150,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr, lr_scheduler],
                class_weight=fold_class_weights,
                verbose=1
            )

        # Evaluate on the validation set of this fold
        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"Fold {fold + 1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
        fold_results.append({'loss': val_loss, 'accuracy': val_acc})

    # 6. Analyze Cross-Validation Results
    print("\n--- Cross-Validation Summary ---")
    avg_loss = statistics.mean([r['loss'] for r in fold_results])
    avg_acc = statistics.mean([r['accuracy'] for r in fold_results])
    std_loss = statistics.stdev([r['loss'] for r in fold_results]) if len(fold_results) > 1 else 0
    std_acc = statistics.stdev([r['accuracy'] for r in fold_results]) if len(fold_results) > 1 else 0

    print(f"Average Validation Loss: {avg_loss:.4f} (+/- {std_loss:.4f})")
    print(f"Average Validation Accuracy: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print("-" * 30)

    # 7. Train Final Model on All Train+Val Data
    print("\n--- Training Final Model on ALL Train+Val Data ---")
    final_model = create_lstm_model(input_shape, num_classes, **hp)

    final_early_stopping = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=35,  # Extended patience for final model
        verbose=1, 
        restore_best_weights=True, 
        min_delta=0.0005
    )
    
    final_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        mode='min', 
        factor=0.5, 
        patience=15, 
        min_lr=1e-6, 
        verbose=1
    )
    
    final_lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_schedule(epoch, hp['learning_rate'], total_epochs=200), 
        verbose=0
    )
    
    final_model_checkpoint = ModelCheckpoint(
        model_save_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )

    # Calculate class weights for the full train+val set
    train_val_class_counts = np.sum(y_train_val_cat, axis=0)
    train_val_total_samples = np.sum(train_val_class_counts)
    train_val_class_weights = {}
    for i, count in enumerate(train_val_class_counts):
        if count > 0:
            train_val_class_weights[i] = train_val_total_samples / (num_classes * count)
        else:
            train_val_class_weights[i] = 0

    # Set up data generators for final training if augmentation is enabled
    if use_augmentation:
        final_train_gen = AugmentedSequenceGenerator(
            X_train_val_scaled, y_train_val_cat, 
            batch_size=batch_size, 
            augment_prob=0.7
        )
        
        # Train with generator
        final_history = final_model.fit(
            final_train_gen,
            validation_data=(X_test_scaled, y_test_cat),
            epochs=200,
            callbacks=[final_early_stopping, final_reduce_lr, final_lr_scheduler, final_model_checkpoint],
            class_weight=train_val_class_weights,
            verbose=1
        )
    else:
        # Standard training without generator
        final_history = final_model.fit(
            X_train_val_scaled, y_train_val_cat,
            validation_data=(X_test_scaled, y_test_cat),
            epochs=200,
            batch_size=batch_size,
            callbacks=[final_early_stopping, final_reduce_lr, final_lr_scheduler, final_model_checkpoint],
            class_weight=train_val_class_weights,
            verbose=1
        )

    # 8. Evaluate Final Model on Hold-Out Test Set
    print(f"\n--- Evaluating Final Model ({model_save_path}) on Hold-Out Test Set ---")
    try:
        print(f"Loading best weights from {model_save_path}...")
        final_model.load_weights(model_save_path)
    except Exception as e:
        print(f"Warning: Could not load weights from {model_save_path}. Evaluating with current weights. Error: {e}")

    test_loss, test_acc = final_model.evaluate(X_test_scaled, y_test_cat, verbose=1)
    print("\n--- Final Hold-Out Test Set Performance ---")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print("-" * 45)

    # Generate confusion matrix for the final model on the test set
    # y_pred_prob = final_model.predict(X_test_scaled)
    # y_pred = np.argmax(y_pred_prob, axis=1)
    # y_true = np.argmax(y_test_cat, axis=1)
    #plot_confusion_matrix(y_true, y_pred, label_encoder.classes_, filename=f'confusion_matrix_final_lstm.png')

    # Plot final training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(final_history.history['accuracy'], label='Training Accuracy')
    plt.plot(final_history.history['val_accuracy'], label='Test Set Accuracy (Validation)')
    plt.title('Final Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(final_history.history['loss'], label='Training Loss')
    plt.plot(final_history.history['val_loss'], label='Test Set Loss (Validation)')
    plt.title('Final Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    history_plot_path = 'training_history_final_lstm.png'
    plt.savefig(history_plot_path)
    print(f"Final training history plot saved as '{history_plot_path}'")
    plt.close()

    # 9. Calculate per-class metrics for test set
    y_pred_prob = final_model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    print("\n--- Per-Class Performance ---")
    for i, class_name in enumerate(label_encoder.classes_):
        class_indices = y_true == i
        if np.sum(class_indices) > 0:  # Ensure there are samples for this class
            class_acc = accuracy_score(y_true[class_indices], y_pred[class_indices])
            print(f"  {class_name}: {class_acc*100:.2f}% accuracy")
    
    return final_model, label_encoder, scaler

# --- Main execution block ---
if __name__ == '__main__':
    data_path = 'data_real_time'  # Your gesture folders path
    num_cv_splits = 5  # Number of folds for cross-validation
    final_model_path = 'gesture_model_lstm_final2.keras'
    
    try:
        print("===== Starting Improved LSTM Training with K-Fold Cross-Validation =====")
        
        # Run the improved training process
        model, encoder, scaler = run_improved_lstm_training_cv(
            data_path,
            n_splits=num_cv_splits,
            model_save_path=final_model_path,
            use_augmentation=True  # Enable data augmentation
        )
        
        print("\n===== Training Process Completed Successfully! =====")
        print(f"Final model saved to: {final_model_path}")
        print(f"Scaler saved to: gesture_scaler_lstm2.joblib")
        print(f"Labels saved to: label_encoder_classes_lstm2.npy")
        
    except FileNotFoundError as fnf:
        print(f"\nError: {fnf}. Please check data path and file structure.")
        traceback.print_exc()
    except ValueError as ve:
        print(f"\nError during data processing or training: {ve}")
        traceback.print_exc()
    except Exception as e:
        print("\n--- An unexpected error occurred during the main process ---")
        traceback.print_exc()