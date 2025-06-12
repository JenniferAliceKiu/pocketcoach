# Whisper speech-to-Text
import os
import keras
from transformers import pipeline

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
audio_path = os.path.join(script_dir, "..", "raw_data", "test_audio.wav")
model_path = os.path.join(script_dir, "..", "models", "whisper-tiny-local")

# Verify files exist
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found at: {audio_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found at: {model_path}")

# Initialize the pipeline with the online model
print("Initializing online model pipeline...")
transcription_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
)

# Transcribe using online model
print("\nTranscribing with online model...")
result_online = transcription_pipe(audio_path, return_timestamps=True)
print("Online model transcription result:")
print(result_online)

# Load the local model
print("\nInitializing local model pipeline...")
loaded_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_path
)

# Configure the model
loaded_pipeline.model.generation_config.forced_decoder_ids = None
loaded_pipeline.model.generation_config.suppress_tokens = None

# Transcribe using local model
print("\nTranscribing with local model...")
result_local = loaded_pipeline(audio_path, return_timestamps=True)
print("Local model transcription result:")
print(result_local)
