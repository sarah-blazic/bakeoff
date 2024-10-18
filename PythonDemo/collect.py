import os
import tkinter as tk
import sounddevice as sd
import numpy as np
import wave
from common import *

NUM_SECONDS_TO_RECORD = 3

# Ensure class directories exist
for cls in CLASSES:
    class_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")
        self.current_class_index = 0

        # Display current class
        self.class_var = tk.StringVar()
        self.class_var.set(f"Current Class: {CLASSES[self.current_class_index]}")
        self.label = tk.Label(master, textvariable=self.class_var, font=(FONT, 24))
        self.label.pack(pady=20)

        # Instructions
        self.instructions = tk.Label(master, text="Use Left/Right Arrow keys to change class\n\nPress Spacebar to record", font=(FONT, 14), justify="left")
        self.instructions.pack(pady=10, padx=10)

        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set("")
        self.status_label = tk.Label(master, textvariable=self.status_var, font=(FONT, 14))
        self.status_label.pack(pady=10)

        # Countdown message
        self.countdown_var = tk.StringVar()
        self.countdown_var.set("")
        self.countdown_label = tk.Label(master, textvariable=self.countdown_var, font=(FONT, 14))
        self.countdown_label.pack(pady=10)

        # Automation button
        self.automation_on = False
        self.automation_button = tk.Button(master, text=f"Start {NUM_SECONDS_TO_RECORD} Second Countdown", command=self.toggle_automation)
        self.automation_button.pack(pady=10)

        # Variables for managing countdown and automation
        self.remaining_seconds = 0
        self.countdown_after_id = None

        # Bind keys
        master.bind('<Left>', self.prev_class)
        master.bind('<Right>', self.next_class)
        master.bind('<space>', self.record_audio)

        # Focus on the window to capture key events
        master.focus_set()

    def prev_class(self, event):
        self.current_class_index = (self.current_class_index - 1) % len(CLASSES)
        self.class_var.set(f"Current Class: {CLASSES[self.current_class_index]}")

    def next_class(self, event):
        self.current_class_index = (self.current_class_index + 1) % len(CLASSES)
        self.class_var.set(f"Current Class: {CLASSES[self.current_class_index]}")

    def toggle_automation(self):
        if not self.automation_on:
            self.automation_on = True
            self.automation_button.config(text="Stop Countdown")
            self.start_countdown(NUM_SECONDS_TO_RECORD)  # Start with a 5-second countdown
        else:
            self.automation_on = False
            self.automation_button.config(text="Start Countdown")
            self.countdown_var.set("")
            # Cancel any scheduled countdown or recording
            if self.countdown_after_id:
                self.master.after_cancel(self.countdown_after_id)
                self.countdown_after_id = None

    def start_countdown(self, seconds):
        self.remaining_seconds = seconds
        self.update_countdown()

    def update_countdown(self):
        if not self.automation_on:
            self.countdown_var.set("")
            return
        if self.remaining_seconds > 0:
            self.countdown_var.set(f"Recording in {self.remaining_seconds} seconds...")
            self.remaining_seconds -= 1
            self.countdown_after_id = self.master.after(1000, self.update_countdown)
        else:
            self.countdown_var.set("Recording now...")
            self.record_audio()
            self.toggle_automation()

    def record_audio(self, event=None):
        if not self.automation_on and event is None:
            return  # Prevent manual recording without pressing the spacebar
        self.status_var.set("Recording...")
        self.master.update_idletasks()
        print("Recording started...")
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished.")
        self.status_var.set("Recording finished.")
        self.save_audio(audio_data)

    def save_audio(self, audio_data):
        # Convert float32 data to int16
        audio_int16 = np.int16(audio_data * 32767)

        # Save to the appropriate class directory
        class_name = CLASSES[self.current_class_index]
        class_dir = os.path.join(DATA_DIR, class_name)

        # Ensure the class directory exists
        os.makedirs(class_dir, exist_ok=True)

        # List existing files in the directory
        files = os.listdir(class_dir)

        # Extract numbers from filenames
        numbers = []
        for f in files:
            if f.endswith('.wav'):
                name_part = f[:-4]  # Remove '.wav'
                try:
                    number = int(name_part)
                    numbers.append(number)
                except ValueError:
                    pass  # Skip files that don't have numeric names

        # Determine the next highest number
        if numbers:
            next_number = max(numbers) + 1
        else:
            next_number = 1  # Start from 1 if no files are present

        # Create the filename using the next highest number
        filename = f"{next_number}.wav"
        file_path = os.path.join(class_dir, filename)

        # Write WAV file
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        self.status_var.set(f"Saved: {file_path}")
        print(f"Saved audio to {file_path}")

if __name__ == "__main__":
    select_audio_input_device()

    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
