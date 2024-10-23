import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for saving and loading the SVM model
from common import *

def load_data(data_path):
    X = []
    y = []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(folder, file)
                # Load audio file
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                # Ensure that all signals are the same length
                if len(signal) < SAMPLE_RATE * DURATION:
                    pad_width = SAMPLE_RATE * DURATION - len(signal)
                    signal = np.pad(signal, (0, pad_width), 'constant')
                else:
                    signal = signal[:int(SAMPLE_RATE * DURATION)]
                # Compute Mel spectrogram
                mel_spect = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                # Normalize
                mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min() + 1e-6)
                X.append(mel_spect)
                y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_svm(X_train, X_val, y_train, y_val, svm_model_path):
    # Reshape data for SVM (flatten the spectrograms)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    # Standardize the data
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    # Train the SVM classifier
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train_flat, y_train)

    # Validate the model
    y_pred = svm.predict(X_val_flat)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"SVM Validation Accuracy: {accuracy}")

    # Save the trained SVM model
    joblib.dump(svm, svm_model_path)
    print(f"SVM model saved to {svm_model_path}")

    return svm

def train_cnn(X_train, X_val, y_train, y_val):
    # Build the CNN model
    model = models.Sequential([
        layers.Input(shape=(N_MELS, 9, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    callback = EarlyStopping(monitor='loss', patience=2)
    history = model.fit(X_train, y_train, epochs=EPOCHS, 
                        validation_data=(X_val, y_val), callbacks=[callback])
    
    # Save the model
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    print(f"Model saved to {MODEL_DIR}/{MODEL_NAME}")
    
    return model

# Load and preprocess the data
print("Loading data...")
X, y = load_data(DATA_DIR)
print(f"Data loaded. Number of samples: {len(X)}")

# Reshape X for CNN input
X = X[..., np.newaxis]  # Add channel dimension

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')

if CLASSIFIER_TYPE == 'cnn':
    print("Training CNN model...")
    train_cnn(X_train, X_val, y_train, y_val)
elif CLASSIFIER_TYPE == 'svm':
    print("Training SVM model...")
    train_svm(X_train, X_val, y_train, y_val, svm_model_path)
