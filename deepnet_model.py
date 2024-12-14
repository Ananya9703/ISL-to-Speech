# First, update the model.py (training script)
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
import joblib  # For saving scaler

def load_gesture_data(data_path):
    """Load and preprocess gesture data from .npy files."""
    X, y = [], []
    max_length = 0
    
    for gesture_folder in os.listdir(data_path):
        gesture_path = os.path.join(data_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue
        
        for filename in os.listdir(gesture_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(gesture_path, filename)
                data = np.load(file_path)
                
                max_length = max(max_length, len(data))
                X.append(data)
                y.append(gesture_folder)
    
    # Pad arrays to consistent length
    X_padded = np.zeros((len(X), max_length))
    for i, arr in enumerate(X):
        X_padded[i, :len(arr)] = arr
    
    return X_padded, np.array(y)

def create_deepnet_model(input_shape, num_classes):
    """Create a deep neural network model with advanced techniques."""
    model = Sequential([
        # Input layer with L2 regularization and advanced activation
        Dense(512, activation='linear', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        # First deep block
        Dense(384, activation='linear', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        
        # Second deep block
        Dense(256, activation='linear', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        
        # Third deep block
        Dense(192, activation='linear', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        # Fourth deep block
        Dense(128, activation='linear', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        # Final layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Advanced optimization with adaptive learning
    optimizer = Adam(
        learning_rate=0.0003, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def train_gesture_model(data_path, model_path='gesture_deepnet_model.h5'):
    """Train deep neural network gesture recognition model."""
    # Load data
    X, y = load_gesture_data(data_path)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for later use in prediction
    joblib.dump(scaler, 'gesture_scaler.joblib')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )
    
    # Improved callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=50,  # Reasonable patience for deep learning
        restore_best_weights=True,
        min_delta=0.001  # Minimum change to qualify as an improvement
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,  # More aggressive learning rate reduction
        patience=10, 
        min_lr=0.000005,
        verbose=1
    )
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_accuracy', 
        save_best_only=True,
        mode='max'
    )
    
    # Create and train model
    model = create_deepnet_model(X_train.shape[1], y_categorical.shape[1])
    
    history = model.fit(
        X_train, y_train, 
        epochs=150,  # Increased epochs for deep learning
        batch_size=32, 
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print("Total training samples:", X_train.shape[0])
    print("Batch size:", 32)
    print("Number of batches:", X_train.shape[0] // 32)
    
    # Save label encoder
    np.save('label_encoder.npy', label_encoder.classes_)
    
    return model, label_encoder

# Main execution
if __name__ == '__main__':
    data_path = 'gesture_data'
    model, encoder = train_gesture_model(data_path)
    print("Deep neural network model and label encoder saved successfully.")