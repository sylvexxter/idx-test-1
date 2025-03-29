from agents.voice import SingleAgentVoiceWorkflow, VoiceStreamEventAudio
from audio_handler import AudioHandler, AudioConfig, AudioDeviceError
import asyncio
from typing import Optional, List, Dict, Any, Callable
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

class VoiceRiskAssessment:
    """Voice workflow for risk assessment."""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure audio with custom settings
        audio_config = AudioConfig(
            sample_rate=16000,  # Optimized for Whisper model
            channels=1,
            volume_threshold=0.01,  # Adjust this based on your microphone
            block_size=1024,  # Smaller block size for more responsive volume indication
            silence_duration=2.0,  # 2 seconds of silence to detect end of speech
            min_speech_duration=1.0  # At least 1 second of speech needed
        )
        self.audio_handler = AudioHandler(config=audio_config)
        self._recording_task: Optional[asyncio.Task] = None
        self._speaking = False
        self._auto_interaction_mode = False
        self._listener_task: Optional[asyncio.Task] = None
        self._callback: Optional[Callable[[str], None]] = None
        # Thread safety lock
        self._operation_lock = threading.Lock()
        # Store main event loop reference
        self._main_loop = asyncio.get_event_loop()
    
    async def start_recording(self, auto_stop: bool = False) -> None:
        """
        Start recording audio.
        
        Args:
            auto_stop: If True, recording will automatically stop when silence is detected.
        """
        with self._operation_lock:
            if self._recording_task is not None:
                return
            
            if auto_stop:
                print("\nListening... (automatic detection of speech end)")
            else:
                print("\nRecording... (Press Enter when done)")
                
            print("Volume indicator below shows your voice level:")
            self._recording_task = asyncio.create_task(
                self.audio_handler.record_audio(auto_stop=auto_stop)
            )
    
    async def stop_recording(self) -> bytes:
        """Stop recording and return the audio data."""
        with self._operation_lock:
            if self._recording_task is None:
                raise AudioDeviceError("No active recording")
            
            self.audio_handler.stop_recording()
            try:
                audio_data = await self._recording_task
                self._recording_task = None
                return audio_data
            except Exception as e:
                self._recording_task = None
                raise e
    
    async def process_audio(self, audio_input: bytes) -> str:
        """Process incoming audio and return transcribed text."""
        try:
            print("\nProcessing audio...")
            
            # Save the audio data to a temporary file
            temp_file_path = "temp_recording.wav"
            with open(temp_file_path, "wb") as f:
                f.write(audio_input)
            
            # Use OpenAI's Audio API directly for transcription with improved parameters
            with open(temp_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # Specify English for better recognition
                    response_format="text",
                    temperature=0.2,  # Lower temperature for more focused results
                    prompt="This is a response to a cybersecurity risk assessment question."  # Context helps with domain-specific terms
                )
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            # Return the transcribed text
            transcribed_text = transcript
            print(f"Transcribed: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            raise AudioDeviceError(f"Error processing audio: {str(e)}")
    
    async def synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech and play it."""
        try:
            with self._operation_lock:
                if self._speaking:
                    return b""  # Skip if already speaking
                
                self._speaking = True
            
            print("\nGenerating speech...")
            
            try:
                # Use OpenAI's TTS API directly
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="wav"  # Request WAV format
                )
                
                # Get the audio content as bytes
                audio_data = response.content
                
                if audio_data:
                    # Play the audio
                    print("Playing audio...")
                    await self.audio_handler.play_audio(audio_data)
                    return audio_data
                
                return b""
            except Exception as e:
                print(f"Error during speech synthesis: {str(e)}")
                raise e
            finally:
                with self._operation_lock:
                    self._speaking = False
            
        except Exception as e:
            with self._operation_lock:
                self._speaking = False
            raise AudioDeviceError(f"Error synthesizing speech: {str(e)}")
            
    async def start_continuous_listening(self, callback: Callable[[str], None]) -> None:
        """
        Start continuous listening mode - automatically detect speech, transcribe it,
        and call the callback with the transcribed text.
        
        Args:
            callback: Function to call with the transcribed text when speech is detected
        """
        with self._operation_lock:
            if self._listener_task is not None:
                return
            
            self._auto_interaction_mode = True
            self._callback = callback
            self._listener_task = asyncio.create_task(self._continuous_listening_loop())
        
    async def _continuous_listening_loop(self) -> None:
        """Internal loop for continuous listening."""
        try:
            print("\n--- Starting continuous voice interaction mode ---")
            print("Speak naturally and pause when you're done. The system will automatically detect")
            print("when you've finished speaking and process your response.")
            print("(Speak 'exit' or 'quit' to stop the continuous mode)\n")
            
            while True:
                with self._operation_lock:
                    if not self._auto_interaction_mode:
                        break
                
                try:
                    # Start recording with auto-stop
                    await self.start_recording(auto_stop=True)
                    
                    # Wait for recording to complete
                    audio_data = await self.stop_recording()
                    
                    # Process the audio
                    text = await self.process_audio(audio_data)
                    
                    # Check for exit command
                    if text.strip().lower() in ["exit", "quit", "stop"]:
                        print("\nExiting continuous voice interaction mode")
                        with self._operation_lock:
                            self._auto_interaction_mode = False
                        break
                    
                    # Call the callback with the transcribed text
                    if self._callback and text.strip():
                        try:
                            await self._callback(text)
                        except Exception as callback_error:
                            print(f"Error in voice response callback: {str(callback_error)}")
                            # Continue listening even if callback fails
                    
                except asyncio.CancelledError:
                    # Task was cancelled - don't treat this as an error
                    print("Voice recording interrupted")
                    await asyncio.sleep(1)
                except AudioDeviceError as e:
                    print(f"Audio device error: {str(e)}")
                    await asyncio.sleep(1)  # Short pause before retrying
                except Exception as e:
                    print(f"Error in continuous listening: {str(e)}")
                    await asyncio.sleep(1)  # Short pause before retrying
                
                # Small pause between listening sessions
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            # Task was cancelled - clean up
            with self._operation_lock:
                self._auto_interaction_mode = False
            print("Continuous listening mode interrupted")
        except Exception as e:
            print(f"Fatal error in continuous listening: {str(e)}")
            with self._operation_lock:
                self._auto_interaction_mode = False
        finally:
            with self._operation_lock:
                self._listener_task = None
    
    def stop_continuous_listening(self) -> None:
        """Stop the continuous listening mode."""
        with self._operation_lock:
            self._auto_interaction_mode = False
            if self._listener_task:
                self._listener_task.cancel()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # No need for lock here as this is called during shutdown
        self.stop_continuous_listening()
        if self._recording_task is not None:
            self.audio_handler.stop_recording()
        self.audio_handler.cleanup() 