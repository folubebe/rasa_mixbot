import os
import wave
import numpy as np
import pyaudio
import pyttsx3
import requests
import google.generativeai as genai
import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import threading

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY in your environment.")
genai.configure(api_key=GOOGLE_API_KEY)

# Rasa Server URL
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"  # Adjust if using a different port

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speech speed

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
sample_rate = 16000  # For consistency with your sounddevice code
SILENCE_DURATION = 1.8  # End recording after silence


def speak(text):
    """Convert text to speech using pyttsx3."""
    engine.say(text)
    engine.runAndWait()


def get_peak_value(duration=5):
    """
    Record audio and determine the appropriate threshold for voice detection.
    
    Args:
        duration (int): Recording duration in seconds
        
    Returns:
        float: The calculated gate threshold value
    """
    print("Calibrating microphone... (please stay quiet)")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    print("Calibration finished!")
    
    # Save recording to a WAV file
    output_file = "ambient_noise.wav"
    write(output_file, sample_rate, recording)
    
    # Load audio with librosa
    y, sr = librosa.load(output_file, sr=sample_rate)
    
    # Calculate the peak value
    peak_value = np.max(np.abs(y))
    gate_threshold = peak_value * 700
    print(f"Threshold set to: {gate_threshold:.5f}")
    
    # Clean up temporary file
    try:
        os.remove(output_file)
    except:
        pass
        
    return gate_threshold


def wait_for_voice_activation(threshold):
    """Wait for voice to exceed threshold before starting recording."""
    threshold_crossed = threading.Event()
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio stream to detect threshold crossing"""
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold and not threshold_crossed.is_set():
            print(f"Voice detected! Level: {volume_norm:.5f}")
            threshold_crossed.set()
    
    # Start monitoring audio
    print(f"Listening for voice... (speak to activate)")
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate)
    stream.start()
    
    # Wait for threshold crossing
    while not threshold_crossed.is_set():
        time.sleep(0.1)
    
    # Stop monitoring once threshold is crossed
    stream.stop()
    stream.close()
    
    # Add a delay after detection in the main thread
    print("Voice detected! Waiting for 2 seconds before starting...")
    time.sleep(0.5)
    
    return True


def record_voice():
    """Record user voice input until silence is detected."""
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Recording settings
    SILENCE_THRESHOLD = 300  # Adjust based on environment

    print("🎤 Recording...")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    silence_limit = int(SILENCE_DURATION * RATE / CHUNK)

    # Record until silence is detected
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Check if this chunk is silent
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            silent_chunks += 1
            if silent_chunks > silence_limit:
                break  # Stop recording when silence is detected
        else:
            silent_chunks = 0

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a temporary file
    temp_audio_file = "temp_command.wav"
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Recording complete.")
    return temp_audio_file


def transcribe_audio_with_gemini(audio_file):
    """Send recorded speech to Google Gemini for transcription."""
    
    with open(audio_file, 'rb') as f:
        file_content = f.read()

    # Initialize Gemini model
    google_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    # Improved prompt for transcription - be more specific
    formatted_prompt = """Transcribe the human speech in this audio accurately. 
    If there's no clear speech, just respond with 'NO_SPEECH_DETECTED'.
    If you hear music or background noise but no clear speech, respond with 'NO_SPEECH_DETECTED'."""

    try:
        response = google_model.generate_content(
            contents=[
                {"text": formatted_prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": file_content}}
            ]
        )

        # Extract text from Gemini's response
        if response and hasattr(response, "text"):
            transcript = response.text.strip()
            
            # Handle the case where Gemini indicates no speech
            if transcript == "NO_SPEECH_DETECTED" or "cannot fulfill" in transcript.lower():
                return None
                
            # Filter out Gemini's explanations about audio content
            if transcript.startswith("The audio") or "appears to contain" in transcript:
                return None
                
            return transcript
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        
    return None


def send_text_to_rasa(text):
    """Send transcribed text to Rasa for intent processing and handle errors properly."""
    
    payload = {"sender": "user", "message": text}
    print(f"Sending to Rasa: {payload}")
    
    try:
        # Increase timeout to 10 seconds to handle slow responses
        response = requests.post(RASA_URL, json=payload, timeout=10)
        print(f"Rasa status code: {response.status_code}")
        print(f"Rasa raw response: {response.text}")
        
        response.raise_for_status()
        
        # Ensure response is valid JSON before parsing
        try:
            rasa_response = response.json()
            print(f"Parsed Rasa response: {rasa_response}")
        except ValueError:
            return "Received an invalid response from Rasa."

        # Check if Rasa returned a valid response
        if rasa_response and isinstance(rasa_response, list) and len(rasa_response) > 0:
            return rasa_response[0].get("text", "I didn't understand that.")
        else:
            # Return a more helpful default message for empty responses
            return "I'm not sure how to respond to that. What would you like help with?"

    except requests.exceptions.Timeout:
        return "I'm having trouble connecting to my brain. Let's try a simpler question or try again in a moment."

    except requests.exceptions.ConnectionError:
        return "I'm having trouble connecting right now. Please make sure the Rasa server is running."

    except requests.exceptions.RequestException as e:
        return f"I encountered an error: {e}"


def main():
    """Main loop for the voice assistant."""
    
    # Initial calibration to get threshold
    threshold = get_peak_value()
    speak("Voice assistant ready. Speak to activate. I'm here to assist with **audio-related topics**")
    
    while True:
        # Step 1: Wait for voice activation
        print("Waiting for voice activation...")
        wait_for_voice_activation(threshold)
        
        # Step 2: Record voice once activated
        audio_file = record_voice()
        transcribed_text = transcribe_audio_with_gemini(audio_file)

        # Remove temporary audio file
        try:
            os.remove(audio_file)
        except:
            pass

        if not transcribed_text:
            speak("I didn't catch that. Could you please speak clearly?")
            continue

        print(f"🗣 You: {transcribed_text}")

        if transcribed_text.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye! Have a peaceful day.")
            break

        # Step 3: Send text to Rasa for processing
        rasa_response = send_text_to_rasa(transcribed_text)
        print(f"🤖 Rasa: {rasa_response}")

        # Step 4: Speak back the response directly
        speak(rasa_response)


if __name__ == "__main__":
    main()