import numpy as np
import librosa
import joblib
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import sounddevice as sd
import queue
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import os
import threading
from pynput import keyboard
from common import *

cnn_model_path = os.path.join(MODEL_DIR, MODEL_NAME)  # Path to the CNN model
svm_model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')  # Path to the SVM model

# Load the appropriate model based on the classifier type
if CLASSIFIER_TYPE == 'cnn':
    # Load the saved CNN model
    model = models.load_model(cnn_model_path)
    print("CNN model loaded.")
elif CLASSIFIER_TYPE == 'svm':
    # Load the saved SVM model
    model = joblib.load(svm_model_path)
    print("SVM model loaded.")


# Initialize variables
recording = False
classifications = []
q = queue.Queue()

# Define audio callback
def audio_callback(indata, frames, time_info, status):
    if recording:
        q.put(indata.copy())

# Function to process audio data
def process_audio():
    global recording, classifications
    while True:
        if recording:
            if not q.empty():
                data = q.get()
                signal = data.flatten()
                # Ensure signal length
                required_length = int(SAMPLE_RATE * DURATION)
                if len(signal) < required_length:
                    pad_width = required_length - len(signal)
                    signal = np.pad(signal, (0, pad_width), 'constant')
                else:
                    signal = signal[:required_length]
                # Compute Mel spectrogram
                mel_spect = librosa.feature.melspectrogram(
                    y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
                )
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
                # Normalize
                mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min() + 1e-6)
                # Reshape for model input
                input_data = mel_spect[np.newaxis, ..., np.newaxis]
                # Predict depending on classifier type
                if CLASSIFIER_TYPE == 'cnn':
                    # Reshape for CNN input (4D: batch, height, width, channels)
                    input_data = mel_spect[np.newaxis, ..., np.newaxis]
                    # Predict using the CNN
                    prediction = model.predict(input_data, verbose=None)
                    predicted_class = CLASSES[np.argmax(prediction)]
                elif CLASSIFIER_TYPE == 'svm':
                    # Reshape for SVM input (flatten the spectrograms)
                    input_data = mel_spect.flatten().reshape(1, -1)
                    # Predict using the SVM
                    prediction = model.predict(input_data)
                    predicted_class = CLASSES[prediction[0]]
                
                # Add to list
                classifications.append(predicted_class)
                # Sleep to maintain 200ms intervals
                time.sleep(DURATION)
            else:
                time.sleep(0.01)  # Slight delay to prevent busy waiting
        else:
            time.sleep(0.01)  # Wait until recording starts

# Function to handle key presses
def on_press(key):
    global recording
    try:
        if key == keyboard.Key.space:
            recording = not recording
            if recording:
                print("Recording started. Press spacebar again to stop.")
            else:
                print("Classifications during this session:")
                interactions = set()
                for idx, classification in enumerate(classifications, 1):
                    print(f"{idx}. {classification}")
                    interactions.add(classification)
                # Print the interaction set 
                print("Total Interactions:", interactions)
                # Clear classifications for the next recording session
                classifications.clear()
                print("\nPress spacebar to start new recording.")
    except AttributeError:
        pass

# Get user's input
select_audio_input_device()

# Set up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start audio stream
stream = sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION))
stream.start()
print("Audio stream started.")
print("Press spacebar to start/stop recording.")

# Start processing thread
processing_thread = threading.Thread(target=process_audio)
processing_thread.daemon = True
processing_thread.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

# Clean up
stream.stop()
stream.close()
listener.stop()
print("Audio stream stopped.")