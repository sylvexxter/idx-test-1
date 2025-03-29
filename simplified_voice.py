"""
Simplified voice processing module for the Risk Assessment application.
This provides basic voice input/output capabilities without complex integration.
"""

import os
import asyncio
import time  # Added import for time.time()
from typing import Optional, Callable
import simpleaudio as sa
import sounddevice as sd
import numpy as np
import wave
import io
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables for OpenAI API key
load_dotenv()

class SimplifiedVoice:
    """
    A simplified voice handler with basic recording and playback capabilities.
    Uses GPT-4o models for transcription and speech synthesis.
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sample_rate = 44100  # Higher sample rate for better audio quality
        self.channels = 1
        self.recording = False
        self.audio_data = []
        self._play_obj = None
        self.temp_dir = tempfile.mkdtemp()
        # Voice activity detection state
        self.voice_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        # Debug mode for silence detection
        self.debug_mode = True  # Set to True to see detailed silence detection info
        
    def check_audio_devices(self) -> bool:
        """
        Check if audio input devices are available and working.
        Returns True if devices are available, False otherwise.
        """
        try:
            # Get list of available devices
            devices = sd.query_devices()
            
            # Check if any input devices exist
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                print("No audio input devices found. Please check your microphone connection.")
                return False
                
            # Try to get the default input device
            try:
                default_input = sd.query_devices(kind='input')
                print(f"Default audio input device: {default_input['name']}")
            except Exception as e:
                print(f"Warning: Unable to get default input device: {str(e)}")
                # There are input devices, but no default set
                if input_devices:
                    print(f"Available input devices: {len(input_devices)}")
                    # Just because there's no default doesn't mean we can't use the devices
                    return True
                return False
            
            # Try a very brief test recording to verify device works
            try:
                print("Testing audio input device...")
                test_duration = 0.1  # Very short test, just to check if device works
                test_data = sd.rec(
                    int(test_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    blocking=True
                )
                print("Audio input device test successful")
                return True
            except Exception as e:
                error_str = str(e)
                print(f"Audio test failed: {error_str}")
                
                # Special handling for macOS PortAudio error -9986
                if "Error" in error_str and "-9986" in error_str:
                    print("\nDetected macOS PortAudio error -9986.")
                    print("This is often caused by permission issues or audio configuration problems.")
                    print("Try these steps:")
                    print("1. Check System Preferences -> Security & Privacy -> Microphone permissions")
                    print("2. Restart your computer")
                    print("3. Try a different microphone or audio input device")
                
                return False
                
            return True
        except Exception as e:
            print(f"Error checking audio devices: {str(e)}")
            return False
    
    async def record_audio(self, duration: int = 30, auto_stop: bool = True) -> bytes:
        """
        Record audio with natural voice activity detection following OpenAI Agents SDK patterns.
        
        This implementation follows the OpenAI Agents SDK approach with smooth, natural
        silence detection that doesn't interrupt the user.
        
        Args:
            duration: Maximum recording duration in seconds
            auto_stop: Whether to automatically stop recording when silence is detected
            
        Returns:
            Audio data as bytes or empty bytes if text fallback was used
        """
        # Check audio devices first
        if not self.check_audio_devices():
            print("Audio input unavailable. Please check your microphone connection.")
            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("FALLBACK TO TEXT INPUT DUE TO AUDIO DEVICE ISSUES")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            # Return empty bytes - caller should handle this by checking if audio_data is empty
            return b""
        
        # Voice detection parameters for natural conversation
        vad_params = {
            # Higher thresholds to avoid detecting background noise
            'speech_threshold': 0.03,      # Threshold for speech detection
            
            # Elongated silence detection parameters
            'silence_threshold': 0.01,     # Energy below this is considered silence
            'min_silence_duration': 1.5,   # Reduced from 2.0 to 1.5 seconds for faster stopping
            
            # Debouncing to prevent premature cutoffs
            'speech_debounce_frames': 5,   # Frames to keep recording after brief silence
            'vad_filter_size': 3,          # Size of the filtering window for voice activity detection
            
            # Dynamic parameters for natural conversation
            'patience_after_speech': 2.0,  # Time to wait after speech detected (seconds)
            'min_speech_duration': 0.5     # Minimum amount of speech required (seconds)
        }
        
        print("\nListening for your response... (natural silence detection enabled)")
        
        print("\n------ VOICE RECORDING ------")
        print("[LISTENING] Waiting for speech...")
        print("[SPEAKING] Recording your response...")
        print("[SILENCE] Waiting to see if you'll continue...")
        print("[DONE] Recording complete when you're finished speaking")
        print("----------------------------")
        
        # Reset state
        self.recording = True
        self.audio_data = []
        
        # Voice activity detection state
        self.voice_detected = False
        self.speech_duration = 0.0
        self.silence_duration = 0.0
        self.last_speech_time = None
        
        # Use time for accurate tracking
        start_time = time.time()
        
        # Create a stream to record audio
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Error: {status}")
            if not self.recording:
                return
            
            # Add data to buffer
            self.audio_data.append(indata.copy())
            
            # Calculate energy level using RMS
            rms = np.sqrt(np.mean(indata**2))
            
            # Current time for timing calculations
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Speech detection with debouncing
            is_speech = rms >= vad_params['speech_threshold']
            
            if is_speech:
                # Speech detected
                if not self.voice_detected:
                    # First detection of speech
                    self.voice_detected = True
                    print('\r[SPEAKING] Recording your response...', end='', flush=True)
                
                # Reset silence duration since we detected speech
                self.silence_duration = 0.0
                # Update last speech time
                self.last_speech_time = current_time
                # Track speech duration
                self.speech_duration += frames / self.sample_rate
                
                # Visual indicator for speech
                bar_length = min(20, int(rms / vad_params['speech_threshold'] * 20))
                volume_bar = '█' * bar_length + ' ' * (20 - bar_length)
                print(f'\r[SPEAKING] {volume_bar}', end='', flush=True)
            else:
                # No speech detected
                if self.voice_detected:
                    # We've detected speech before, so this might be a pause
                    
                    # Calculate time since last speech
                    time_since_speech = current_time - self.last_speech_time if self.last_speech_time else 0
                    
                    # Update silence duration
                    self.silence_duration = time_since_speech
                    
                    # Only consider stopping if:
                    # 1. We've detected enough speech already
                    # 2. We've had a significant silence period
                    # 3. We've waited patiently after detecting speech
                    has_enough_speech = self.speech_duration >= vad_params['min_speech_duration']
                    has_patience_elapsed = elapsed_time >= vad_params['patience_after_speech']
                    has_silence = self.silence_duration >= vad_params['min_silence_duration']
                    
                    # Visual indicator for silence with countdown
                    silence_percentage = min(100, int(self.silence_duration / vad_params['min_silence_duration'] * 100))
                    blocks = int(silence_percentage / 5)
                    silence_bar = '█' * blocks + '░' * (20 - blocks)
                    print(f'\r[SILENCE {silence_percentage}%] {silence_bar}', end='', flush=True)
                    
                    # Keep track of how long we've been at 100%
                    if silence_percentage >= 100:
                        if not hasattr(self, 'at_100_percent_since'):
                            self.at_100_percent_since = current_time
                        elif current_time - self.at_100_percent_since > 0.5:  # Force stop after 0.5 seconds at 100%
                            print("\n[DONE] Recording complete - silence threshold reached.")
                            self.recording = False
                            return
                    else:
                        if hasattr(self, 'at_100_percent_since'):
                            delattr(self, 'at_100_percent_since')
                    
                    # Debug info to help diagnose issues
                    if self.debug_mode and silence_percentage >= 90:
                        conditions = f"Speech: {has_enough_speech}, Silence: {has_silence}, Patience: {has_patience_elapsed}"
                        print(f"\r[DBG] {conditions} | Silence: {self.silence_duration:.1f}/{vad_params['min_silence_duration']:.1f}s", end='', flush=True)
                    
                    # CRITICAL FIX: Immediately stop when all conditions are met
                    # Check specifically for silence_duration reaching or exceeding min_silence_duration
                    if self.silence_duration >= vad_params['min_silence_duration'] and has_enough_speech and has_patience_elapsed:
                        print("\n[DONE] Speech ended, recording complete.")
                        self.recording = False
                        return  # Immediately exit the callback to ensure recording stops
                else:
                    # Still waiting for speech to begin
                    # Visual indicator for listening
                    noise_bar = '·' * min(20, int(rms / vad_params['speech_threshold'] * 30)) + ' ' * 20
                    print(f'\r[LISTENING] {noise_bar[:20]}', end='', flush=True)
        
        try:
            # Try to create the audio stream with error handling
            try:
                stream = sd.InputStream(
                    callback=callback,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=1024,
                    dtype=np.float32
                )
            except Exception as e:
                # Handle PortAudio initialization errors
                error_msg = str(e)
                print(f"Error initializing audio stream: {error_msg}")
                
                if "Error opening InputStream" in error_msg:
                    print("\nAudio device initialization failed. This could be due to:")
                    print("1. No microphone available or permission denied")
                    print("2. Audio device is in use by another application")
                    print("3. Driver or hardware issues with your audio device")
                    # Return empty bytes to indicate failure
                    return b""
                else:
                    # For other errors, just propagate
                    raise
            
            # Process recording with the stream
            with stream:
                # Record for the specified duration or until auto-stopped
                while (time.time() - start_time) < duration and self.recording:
                    try:
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        # Allow asyncio cancellation to propagate
                        print("\nRecording cancelled")
                        self.recording = False
                        raise
                
                # If we reach the maximum duration
                if self.recording:
                    print("\n[DONE] Maximum recording time reached")
                    self.recording = False
                    
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nRecording interrupted by user")
            self.recording = False
            # Let the KeyboardInterrupt propagate so the caller can handle it
            raise
        except Exception as e:
            print(f"\nAudio recording error: {type(e).__name__}: {str(e)}")
            self.recording = False
            raise
        finally:
            # Ensure we stop recording even if there's an exception
            self.recording = False
            print("\nProcessing your recording...")
        
        # Check if we recorded anything
        if not self.audio_data:
            return b""
            
        # Convert to WAV format
        audio_data = np.concatenate(self.audio_data, axis=0)
        
        # Apply audio filtering for better quality
        try:
            from scipy import signal
            
            # 1. High-pass filter to remove low frequency noise (below 70 Hz)
            sos = signal.butter(8, 70, 'hp', fs=self.sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
            
            # 2. Gentle normalization
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        except ImportError:
            # Fallback if scipy is not installed
            print("Using basic audio processing without filters")
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        except Exception as e:
            print(f"Error during audio processing: {str(e)}")
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
            
        # Convert to WAV
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        return byte_io.getvalue()
    
    def stop_recording(self):
        """Stop the current recording."""
        self.recording = False
    
    async def play_audio(self, audio_data: bytes) -> None:
        """Play audio from bytes data."""
        try:
            # Try to read as WAV
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
        except wave.Error:
            # If not WAV, assume it's raw PCM data
            sample_width = 2  # 16-bit audio
            channels = self.channels
            sample_rate = self.sample_rate
        
        # Create wave object
        wave_obj = sa.WaveObject(audio_data, channels, sample_width, sample_rate)
        self._play_obj = wave_obj.play()
        
        # Wait for playback to finish
        while self._play_obj.is_playing():
            await asyncio.sleep(0.1)
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio using GPT-4o model instead of Whisper.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Transcribed text
        """
        if not audio_data:
            return ""
            
        print("Transcribing audio with gpt-4o-transcribe...")
        
        # Save audio to a temporary file
        temp_file = os.path.join(self.temp_dir, "recording.wav")
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        
        # Transcribe using OpenAI GPT-4o
        try:
            with open(temp_file, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",  # Use GPT-4o for transcription
                    file=audio_file,
                    language="en",
                    response_format="text",
                    temperature=0.3  # Lower temperature for more accuracy
                )
            
            # Return the transcribed text
            transcribed_text = transcript
            print(f"Transcribed: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            # Try fallback to audio API if the model is not available
            try:
                print("Attempting fallback to available model...")
                with open(temp_file, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",  # Fallback model
                        file=audio_file,
                        language="en",
                        response_format="text"
                    )
                transcribed_text = transcript
                print(f"Transcribed (fallback): {transcribed_text}")
                return transcribed_text
            except Exception as fallback_error:
                print(f"Fallback transcription failed: {str(fallback_error)}")
                return ""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def synthesize_speech(self, text: str) -> None:
        """
        Convert text to speech using GPT-4o-mini TTS and play it.
        
        Args:
            text: Text to convert to speech
        """
        print("Generating speech with gpt-4o-mini-tts...")
        
        try:
            # Use OpenAI's TTS API with GPT-4o-mini
            response = self.openai_client.audio.speech.create(
                model="gpt-4o-mini-tts",  # Use GPT-4o-mini for TTS
                voice="onyx",  # Use a clear, natural voice
                input=text,
                response_format="wav",  # Request WAV format
                speed=0.9  # Slightly slower for better clarity
            )
            
            # Get the audio content as bytes
            audio_data = response.content
            
            # Play the audio
            print("Playing audio...")
            await self.play_audio(audio_data)
        except Exception as e:
            print(f"Error with gpt-4o-mini-tts: {str(e)}")
            # Fallback to tts-1 if gpt-4o-mini-tts is not available
            try:
                print("Falling back to tts-1 model...")
                response = self.openai_client.audio.speech.create(
                    model="tts-1",  # Fallback model
                    voice="onyx",   # Clear voice
                    input=text,
                    response_format="wav",
                    speed=0.9  # Slightly slower for clarity
                )
                audio_data = response.content
                print("Playing audio (fallback)...")
                await self.play_audio(audio_data)
            except Exception as fallback_error:
                print(f"Fallback TTS failed: {str(fallback_error)}")
    
    def cleanup(self):
        """Clean up resources."""
        # Stop any ongoing playback
        if self._play_obj and self._play_obj.is_playing():
            self._play_obj.stop()
            
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.temp_dir)

# Simple voice interaction functions for demo
async def voice_demo():
    """Demo the voice module."""
    voice = SimplifiedVoice()
    
    try:
        # Synthesize speech
        await voice.synthesize_speech("Hello, I am your risk assessment assistant. Please speak after the beep.")
        
        # Record audio
        audio_data = await voice.record_audio(duration=5)
        
        # Transcribe audio
        text = await voice.transcribe_audio(audio_data)
        
        # Respond to the user
        if text:
            await voice.synthesize_speech(f"You said: {text}")
        else:
            await voice.synthesize_speech("I didn't hear anything. Please try again.")
    finally:
        voice.cleanup()

if __name__ == "__main__":
    # Run the demo
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(voice_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error in demo: {str(e)}")
    finally:
        loop.close() 