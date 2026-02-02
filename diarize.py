from google import genai
from google.genai import types
import json
import time
import wave
import numpy as np
import os

# --- CONFIGURATION ---
LOCATION = "us-central1"
PROJECT_ID = 'jackngare-amboseli-demos'

class GeminiSpeechProcessor:
    def __init__(self, project_id, location, model_name="gemini-2.5-flash"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        
        # Audio analysis thresholds
        self.silence_threshold = 0.005  # RMS threshold for silence detection (more sensitive)
        self.min_speech_duration = 0.2  # Minimum seconds of speech required (allows single words)
        
        # Generation configuration for deterministic JSON output
        self.config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
            ]
        )

    def analyze_audio_content(self, audio_path):
        """Analyze audio file to detect if it contains meaningful speech content."""
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.float32
                
                audio_data = np.frombuffer(frames, dtype=dtype)
                
                # Handle stereo audio
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    audio_data = np.mean(audio_data, axis=1)
                
                # Normalize audio data
                if dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    if dtype == np.int16:
                        audio_data /= 32768.0
                    elif dtype == np.int32:
                        audio_data /= 2147483648.0
                    elif dtype == np.uint8:
                        audio_data = (audio_data - 128) / 128.0
                
                # Calculate RMS (Root Mean Square) energy
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Calculate duration
                duration = len(audio_data) / sample_rate
                
                # Detect speech segments
                window_size = int(0.1 * sample_rate)  # 100ms windows
                speech_segments = 0
                
                for i in range(0, len(audio_data) - window_size, window_size):
                    window = audio_data[i:i + window_size]
                    window_rms = np.sqrt(np.mean(window**2))
                    if window_rms > self.silence_threshold:
                        speech_segments += 1
                
                speech_duration = speech_segments * 0.1
                
                analysis = {
                    'duration': duration,
                    'rms_energy': rms,
                    'speech_duration': speech_duration,
                    'has_speech': speech_duration >= self.min_speech_duration and rms > self.silence_threshold,
                    'silence_ratio': 1 - (speech_duration / duration) if duration > 0 else 1
                }
                
                print(f"Audio Analysis:")
                print(f"  Duration: {duration:.2f}s")
                print(f"  RMS Energy: {rms:.4f}")
                print(f"  Speech Duration: {speech_duration:.2f}s")
                print(f"  Silence Ratio: {analysis['silence_ratio']:.2f}")
                print(f"  Has Speech: {analysis['has_speech']}")
                
                return analysis
                
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {'has_speech': True, 'duration': 0, 'rms_energy': 0, 'speech_duration': 0, 'silence_ratio': 1}

    def process(self, audio_path):
        print(f"Loading audio: {audio_path}...")
        
        # 1. Pre-analyze audio content to detect silence/speech
        audio_analysis = self.analyze_audio_content(audio_path)
        
        # If no meaningful speech detected, return early
        if not audio_analysis['has_speech']:
            print("\n--- Audio Analysis Result ---")
            print("No meaningful speech content detected in audio file.")
            print("Skipping transcription to prevent hallucination.")
            return
        
        # 2. Load Audio
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # 3. Construct Enhanced Anti-Hallucination Prompt
        prompt_text = f"""
        You are an expert transcriptionist with strict accuracy requirements.
        
        CRITICAL INSTRUCTIONS:
        - ONLY transcribe speech that you can clearly hear in the audio
        - DO NOT generate fictional content, conversations, or made-up speech
        - If you cannot clearly hear speech, return an empty array []
        - DO NOT create timestamps for silence or background noise
        - BE EXTREMELY CONSERVATIVE - when in doubt, return empty array
        
        Audio Analysis Context:
        - Audio duration: {audio_analysis['duration']:.2f} seconds
        - Speech content detected: {audio_analysis['speech_duration']:.2f} seconds
        - Silence ratio: {audio_analysis['silence_ratio']:.2f}
        
        Task: Transcribe ONLY the clearly audible speech in the provided audio file.
        
        Rules:
        1. Identify speakers as "Speaker 1", "Speaker 2", etc. ONLY if you can clearly distinguish different voices
        2. Provide timestamps in [MM:SS] format ONLY for actual speech you can hear
        3. COMPLETELY IGNORE silence, background noise, or unclear audio
        4. If a word is repeated due to echo/stutter, write it once
        5. If there is no clear speech or only silence/noise, return []
        6. DO NOT INVENT or HALLUCINATE any speech content
        7. Confidence check: If you're not 90% certain you heard specific words, don't include them
        
        Output Format:
        Return a JSON array of objects. Each object must have:
        - "time": string (e.g. "[00:12]")
        - "speaker": string
        - "text": string
        - "confidence": number (0.0-1.0, where 1.0 is completely certain)
        
        REMEMBER: It's better to return an empty array than to generate fictional content!
        """

        print(f"Sending to {self.model_name}...")
        
        start_time = time.time()
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                    prompt_text
                ],
                config=self.config
            )
            
            elapsed_time = time.time() - start_time
            print(f"\nProcessing time: {elapsed_time:.2f} seconds")
            
            # Validate and filter the response
            self._validate_and_print_transcript(response.text, audio_analysis)
            
        except Exception as e:
            print(f"Error processing audio: {e}")

    def _validate_and_print_transcript(self, json_text, audio_analysis):
        """Validate transcript against audio analysis and filter low-confidence results."""
        try:
            transcript = json.loads(json_text)
            
            print("\n--- Gemini 2.0 Transcript ---\n")
            
            if not transcript:
                print("No speech detected.")
                return

            # Filter and validate transcript entries
            validated_entries = []
            for entry in transcript:
                time_str = entry.get("time", "")
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                confidence = entry.get("confidence", 0.5)  # Default to medium confidence if not provided
                
                # Skip entries with very low confidence
                if confidence < 0.7:
                    print(f"[FILTERED] Low confidence ({confidence:.2f}): {time_str} {speaker}: {text}")
                    continue
                
                # Skip entries that are suspiciously generic or common hallucinations
                suspicious_phrases = [
                ]
                
                if any(phrase in text.lower() for phrase in suspicious_phrases):
                    print(f"[FILTERED] Suspicious content: {time_str} {speaker}: {text}")
                    continue
                
                validated_entries.append(entry)
            
            # Additional validation: Check if transcript length makes sense given audio analysis
            if len(validated_entries) > 0:
                expected_max_entries = max(1, int(audio_analysis['speech_duration'] / 2))  # Rough estimate
                if len(validated_entries) > expected_max_entries * 3:  # Allow some flexibility
                    print(f"[WARNING] Transcript seems too long for detected speech duration")
                    print(f"Expected max ~{expected_max_entries} entries, got {len(validated_entries)}")
            
            # Print validated results
            if not validated_entries:
                print("No high-confidence speech detected after validation.")
                return
                
            for entry in validated_entries:
                time_str = entry.get("time", "")
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                confidence = entry.get("confidence", "N/A")
                print(f"{time_str} {speaker}: {text} [confidence: {confidence}]")
                
        except json.JSONDecodeError:
            print("\n--- Raw Transcript (JSON Parse Failed) ---\n")
            print(json_text)

    def _print_json_transcript(self, json_text):
        """Legacy method for backward compatibility."""
        try:
            transcript = json.loads(json_text)
            
            print("\n--- Gemini 2.0 Transcript ---\n")
            
            if not transcript:
                print("No speech detected.")
                return

            for entry in transcript:
                time = entry.get("time", "")
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                print(f"{time} {speaker}: {text}")
                
        except json.JSONDecodeError:
            print("\n--- Raw Transcript (JSON Parse Failed) ---\n")
            print(json_text)

# --- USAGE ---
# Make sure to replace this path with your actual local or drive path
audio_path = "./audio/TIC-a02ee4b6.wav"  # Test with the original silent file again

# Initializing with Gemini 2.5 Flash
processor = GeminiSpeechProcessor(project_id=PROJECT_ID, location=LOCATION)
processor.process(audio_path)
