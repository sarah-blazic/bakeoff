import sounddevice as sd

FONT = "Cambria"

# Audio settings
SAMPLE_RATE = 22050  # Sample rate of audio files
DURATION = 0.2  # Duration to record in seconds
N_MELS = 64  # Number of Mel bands to generate
HOP_LENGTH = 512  # Number of samples between successive frames

# Training settings
EPOCHS = 100

# Define the classes (should match the classes used during training)
CLASSES = ['neutral', 'net', 'rim']

DATA_DIR = 'PythonDemo/data'

MODEL_DIR = 'PythonDemo/models'
MODEL_NAME = 'audio_classification_model.keras'

CLASSIFIER_TYPE = "svm"

def select_audio_input_device():
    print("Available audio input devices:")
    input_devices = []
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']}")
            input_devices.append((idx, device['name']))
    if len(input_devices) == 0:
        print("No audio input devices found.")
        exit()
    else:
        device_ids = [device[0] for device in input_devices]
        while True:
            try:
                device_id = int(input("Select the input device ID: "))
                if device_id in device_ids:
                    sd.default.device = device_id
                    print(f"Selected input device: {devices[device_id]['name']}")
                    break
                else:
                    print("Invalid device ID. Please select from the listed IDs.")
            except ValueError:
                print("Invalid input. Please enter a valid device ID.")