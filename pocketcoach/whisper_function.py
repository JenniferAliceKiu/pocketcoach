# Whisper speech-to-Text
import os
from transformers import pipeline
from datetime import datetime

def transcribe_audio(audio_file_path, model_type="online"):
    """
    Transcribe audio file and save the transcription to raw_data directory.

    Args:
        audio_file_path (str): Path to the audio file to transcribe
        model_type (str): Either "online" or "local" to specify which model to use

    Returns:
        tuple: (transcription_result, saved_file_path)
    """
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths
    model_path = os.path.join(script_dir, "..", "models", "whisper-tiny-local")
    raw_data_dir = os.path.join(script_dir, "..", "raw_data")

    # Verify directories exist
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found at: {raw_data_dir}")
    if model_type == "local" and not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model directory not found at: {model_path}")

    def save_transcription(transcription, model_type):
        """Save transcription to a text file in raw_data directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{model_type}_{timestamp}.txt"
        filepath = os.path.join(raw_data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(transcription))
        print(f"Transcription saved to: {filepath}")
        return filepath

    # Initialize the appropriate pipeline
    if model_type == "online":
        print("Initializing online model pipeline...")
        transcription_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
        )
    else:  # local model
        print("Initializing local model pipeline...")
        transcription_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path
        )
        transcription_pipe.model.generation_config.forced_decoder_ids = None
        transcription_pipe.model.generation_config.suppress_tokens = None

    # Perform transcription
    print(f"\nTranscribing with {model_type} model...")
    result = transcription_pipe(audio_file_path, return_timestamps=True)

    # Save transcription
    saved_file_path = save_transcription(result, model_type)

    return result, saved_file_path

# Example usage when running this script directly
if __name__ == "__main__":
    # Get the absolute path to the audio file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, "..", "raw_data", "test_audio.wav")

    # Verify audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # Transcribe with both models
    result_online, file_online = transcribe_audio(audio_path, "online")
    result_local, file_local = transcribe_audio(audio_path, "local")
