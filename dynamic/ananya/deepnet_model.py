import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib

def safe_load_numpy(file_path):
    """Safely load and process numpy files with robust error handling."""
    try:
        data = np.load(file_path, allow_pickle=True)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.dtype == object:
            flattened = []
            for item in data.flatten():
                if isinstance(item, (list, np.ndarray)):
                    flat_item = np.array(item).flatten()
                    flattened.extend([float(x) for x in flat_item if str(x).strip()])
                else:
                    if str(item).strip():
                        flattened.append(float(item))
            return np.array(flattened, dtype=np.float32)
        else:
            if len(data.shape) > 1:
                return data.flatten().astype(np.float32)
            return data.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {str(e)}")

def load_gesture_data(data_path):
    """Load and preprocess gesture data with advanced error handling."""
    print("Analyzing dataset structure...")
    X, y = [], []
    max_length = 0
    folder_counts = {}
    
    # First pass: analyze dataset structure
    for gesture_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, gesture_folder)
        if not os.path.isdir(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        folder_counts[gesture_folder] = len(files)
        
        # Sample files to get maximum feature length
        for filename in files:
            try:
                file_path = os.path.join(folder_path, filename)
                sample_data = safe_load_numpy(file_path)
                max_length = max(max_length, len(sample_data))
            except Exception as e:
                print(f"Warning while analyzing {filename}: {str(e)}")
                continue
    
    print("\nDataset Statistics:")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count} files")
    print(f"Maximum feature length: {max_length}")
    
    # Second pass: load and process files
    for gesture_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, gesture_folder)
        if not os.path.isdir(folder_path):
            continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        print(f"\nProcessing {gesture_folder}...")
        
        successful_loads = 0
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            try:
                data = safe_load_numpy(file_path)
                
                # Validate data
                if not np.all(np.isfinite(data)):
                    raise ValueError("Data contains invalid values")
                
                # Pad or truncate to max_length
                padded_data = np.zeros(max_length, dtype=np.float32)
                padded_data[:min(len(data), max_length)] = data[:min(len(data), max_length)]
                
                X.append(padded_data)
                y.append(gesture_folder)
                successful_loads += 1
                
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
                continue
        
        print(f"Loaded {successful_loads}/{len(files)} files")
    
    return np.array(X, dtype=np.float32), np.array(y)

def create_optimized_model(input_shape, num_classes):
    """Create an optimized model architecture for the gesture dataset."""
    model = Sequential([
        # Input block with larger capacity
        Dense(2048, kernel_regularizer=l2(0.00001), input_shape=(input_shape,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        # Deep feature extraction blocks with gradual dimension reduction
        Dense(1536, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        Dense(1024, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.25),
        
        Dense(512, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.25),
        
        Dense(256, kernel_regularizer=l2(0.00001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_gesture_model(data_path, model_path='gesture_model.h5'):
    """Train the optimized gesture recognition model."""
    # Load and preprocess data
    X, y = load_gesture_data(data_path)
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'gesture_scaler_1.joblib')
    
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Three-way split for better validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_categorical,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    # Advanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        min_delta=0.0001
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Create model
    print("\nCreating model...")
    model = create_optimized_model(X_train.shape[1], y_categorical.shape[1])
    
    # Calculate class weights for balanced training
    class_counts = np.sum(y_train, axis=0)
    class_weights = dict(enumerate(
        np.sum(class_counts) / (len(class_counts) * class_counts)
    ))
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating final model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nFinal Test Metrics:")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Save label encoder classes
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    return model, label_encoder, history

if __name__ == '__main__':
    data_path = 'data_real_time'
    try:
        print("Starting gesture recognition model training...")
        model, encoder, history = train_gesture_model(data_path)
        
        # Plot training history
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("\nTraining history plot saved as 'training_history.png'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

