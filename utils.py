from gtts import gTTS
from io import BytesIO
import io
import numpy as np
import numpy as np
import wave
from scipy.io import wavfile
import speech_recognition as sr
import whisper


def audio_to_text(whisper_model, audio_file_path):
    # r = sr.Recognizer()
    # with sr.AudioFile(audio_file_path) as source:
    #     audio_data = r.record(source)  # read the entire audio file

    # # Recognize speech using Google Web Speech API (default)
    # text = r.recognize_google(audio_data, language="en-US")
    # print(text)
    # return text
    result = whisper_model.transcribe(audio_file_path)
    return result["text"]


def save_audio_wav(
    audio_input,
    output_path="./data/input_audio/recorded_audio.wav",
    desired_sample_rate=16000,
    desired_channels=1,
):
    """Saves audio from a BytesIO object to a WAV file, resampling if needed.

    Args:
        audio_input: A BytesIO object containing the audio data.
        output_path: The path to save the WAV file.
        desired_sample_rate: The desired sample rate for the output audio.
        desired_channels: The desired number of channels for the output audio.
    """
    try:
        # 1. Load WAV file from BytesIO
        with wave.open(io.BytesIO(audio_input.getvalue()), "rb") as wav:
            params = wav.getparams()
            original_sample_rate = wav.getframerate()
            original_channels = wav.getnchannels()
            audio_data = np.frombuffer(wav.readframes(params.nframes), dtype=np.int16)

        # 2. Resample and Convert to Mono if Necessary
        if (
            original_sample_rate != desired_sample_rate
            or original_channels != desired_channels
        ):
            from scipy.signal import (
                resample,
            )  # Import here to avoid unnecessary dependency

            # Resample
            if original_sample_rate != desired_sample_rate:
                num_samples = int(
                    len(audio_data) * (desired_sample_rate / original_sample_rate)
                )
                audio_data = resample(audio_data, num_samples)

            # Convert to Mono (if needed)
            if (
                original_channels != desired_channels and original_channels > 1
            ):  # Only if it is in stereo
                audio_data = audio_data.reshape(-1, original_channels).mean(
                    axis=1
                )  # Convert to mono

            audio_data = audio_data.astype(
                np.int16
            )  # Important to convert back to int16 after processing

        # 3. Save modified audio
        wavfile.write(output_path, desired_sample_rate, audio_data)
        print(f"Audio saved to {output_path}")

    except Exception as e:
        print(f"Error saving audio: {e}")
        return None  # Or handle the error as needed

    return output_path  # Return the path to the file


def text_to_audio(text: str, language: str, path_to_save: str) -> None:
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save(path_to_save)


def generate_audio(text, engine):
    audio_stream = BytesIO()
    engine.save_to_file(text, audio_stream)
    audio_stream.seek(0)
    return audio_stream


if __name__ == "main":
    pass
