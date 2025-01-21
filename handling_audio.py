import torch
from transformers import pipeline
import librosa
import io

def convert_bytes_to_array(audio_bytes):
    try:
        audio_bytes = io.BytesIO(audio_bytes)
        audio, sample_rate = librosa.load(audio_bytes, sr=None)
        return audio, sample_rate
    except Exception as e:
        raise ValueError(f"Error converting audio bytes: {e}")

def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize the Whisper pipeline
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        device=device,
    )

    try:
        # Convert audio bytes to array and sample rate
        audio_array, sample_rate = convert_bytes_to_array(audio_bytes)
        
        # Ensure the pipeline gets the correct input
        prediction = pipe(
            {
                "array": audio_array,
                "sampling_rate": sample_rate
            }
        )["text"]
        return prediction
    except Exception as e:
        raise ValueError(f"Error during transcription: {e}")
