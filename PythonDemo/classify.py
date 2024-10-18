import numpy as np
import librosa
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
from common import *

# Current model to use for classification
MODEL_NAME = 'audio_classification_model.keras'

# Load the saved model
model = models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))
print("Model loaded.")

# Create a queue to communicate between the audio callback and main thread
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    q.put(indata.copy())


# Select audio input device
select_audio_input_device()

# Set up GUI
root = tk.Tk()
root.title("Live Audio Classification")

# Initialize plot
fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root) 
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Initialize text for classification result
label_var = tk.StringVar()
label = tk.Label(root, textvariable=label_var, font=('Arial', 24))
label.pack()

# Start audio stream
stream = sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION))
stream.start()
print("Audio stream started.")

last_class = 'neutral'

def update_gui():
    global last_class

    # Check if there is audio data in the queue
    if last_class != 'neutral':
        # A class was detected; pause updates
        last_class = 'neutral'
        # Clear the queue to prevent re-processing the same data
        with q.mutex:
            q.queue.clear()
        # Schedule the next update after 1 second
        root.after(1000, update_gui)
    else:
        if not q.empty():
            data = q.get()
            # Process audio data
            signal = data.flatten()
            # Ensure signal length
            if len(signal) < int(SAMPLE_RATE * DURATION):
                pad_width = int(SAMPLE_RATE * DURATION) - len(signal)
                signal = np.pad(signal, (0, pad_width), 'constant')
            else:
                signal = signal[:int(SAMPLE_RATE * DURATION)]
            # Compute Mel spectrogram
            mel_spect = librosa.feature.melspectrogram(
                y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
            )
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            # Normalize
            mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min() + 1e-6)
            # Reshape for model input
            input_data = mel_spect[np.newaxis, ..., np.newaxis]
            # Predict
            prediction = model.predict(input_data, verbose=None)
            predicted_class = CLASSES[np.argmax(prediction)]
            # Update text
            label_var.set(f"Classification: {predicted_class}")
            # Update spectrogram plot
            ax.clear()
            librosa.display.specshow(
                mel_spect, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis='time', y_axis='mel', ax=ax
            )
            ax.set(title='Live Spectrogram')
            fig.canvas.draw()
            # Update last_class
            last_class = predicted_class

        # Schedule the next call to this function regardless of queue status
        root.after(100, update_gui)

def on_closing():
    stream.stop()
    stream.close()
    root.quit()
    print("Audio stream stopped.")

root.protocol("WM_DELETE_WINDOW", on_closing)

update_gui()
root.mainloop()
