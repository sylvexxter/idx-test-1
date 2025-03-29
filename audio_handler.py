import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import os
from typing import Optional, Tuple, Callable, List
import asyncio
from dataclasses import dataclass
import sys
import simpleaudio as sa
import wave
import io
import time
import threading

@dataclass
class AudioConfig:
    """Configuration for audio recording and playback."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: np.dtype = np.float32
    block_size: int = 1024
    device: Optional[int] = None
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    volume_threshold: float = 0.01  # Threshold for considering audio as speech
    silence_duration: float = 2.0  # Duration of silence (in seconds) to detect end of speech
    min_speech_duration: float = 1.0  # Minimum duration of speech before considering it complete

class AudioDeviceError(Exception):
    """Exception raised for audio device related errors."""
    pass

class AudioHandler:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self._recording = False
        self._audio_data = []
        self._play_obj: Optional[sa.PlayObject] = None
        self._temp_dir = tempfile.mkdtemp()
        self._volume_callback: Optional[Callable[[float], None]] = None
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_frames = 0
        self._speech_detected = False
        self._is_auto_stop = False
        self._auto_stop_event = asyncio.Event()
        self._thread_lock = threading.Lock()
        self._stop_requested = False

    def _validate_devices(self) -> None:
        """Validate audio devices and their capabilities."""
        try:
            devices = sd.query_devices()
            if self.config.device is None:
                # Use default devices
                self.config.input_device = sd.default.device[0]
                self.config.output_device = sd.default.device[1]
            else:
                self.config.input_device = self.config.device
                self.config.output_device = self.config.device

            # Validate input device
            input_info = sd.query_devices(self.config.input_device, 'input')
            if input_info is None:
                raise AudioDeviceError(f"Input device {self.config.input_device} not found")

        except sd.PortAudioError as e:
            raise AudioDeviceError(f"Error accessing audio devices: {str(e)}")

    def set_volume_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for volume level updates."""
        self._volume_callback = callback

    def _get_volume_indicator(self, volume: float, width: int = 40) -> str:
        """Generate a visual volume indicator."""
        threshold = self.config.volume_threshold
        if volume < threshold:
            return "[ " + "-" * width + " ]"
        
        # Scale volume to width
        filled = min(width, int(volume * width * 2))
        empty = width - filled
        return f"[ {'|' * filled}{'-' * empty} ]"

    async def _wait_for_auto_stop(self) -> None:
        """Wait for automatic stop based on silence detection."""
        await self._auto_stop_event.wait()
        with self._thread_lock:
            if self._is_auto_stop and self._recording:
                self.stop_recording()

    async def record_audio(self, auto_stop: bool = False) -> bytes:
        """
        Record audio from the microphone.
        
        Args:
            auto_stop: If True, recording will automatically stop when silence is detected.
        """
        with self._thread_lock:
            self._recording = True
            self._stop_requested = False
            self._audio_data = []
            self._silence_frames = 0
            self._speech_frames = 0
            self._total_frames = 0
            self._speech_detected = False
            self._is_auto_stop = auto_stop
            self._auto_stop_event.clear()
        
        # Start the auto-stop monitor if needed
        auto_stop_task = None
        if auto_stop:
            auto_stop_task = asyncio.create_task(self._wait_for_auto_stop())
        
        stream = None
        try:
            def callback(indata, frames, time, status):
                if status:
                    print(f'Error: {status}')
                
                with self._thread_lock:
                    if not self._recording:
                        return
                        
                    # Calculate volume level
                    volume_norm = np.linalg.norm(indata) * 10
                    # Create a simple volume indicator
                    bars = int(volume_norm / self.config.volume_threshold)
                    print('\r' + '█' * min(bars, 50) + ' ' * (50 - min(bars, 50)), end='', flush=True)
                    
                    self._total_frames += 1
                    
                    # Voice activity detection
                    if volume_norm >= self.config.volume_threshold:
                        self._speech_frames += 1
                        self._silence_frames = 0
                        self._speech_detected = True
                        
                        # Apply gain to improve speech clarity
                        gain_audio = indata * 1.5  # Amplify by 1.5x
                        # Clip to avoid distortion
                        gain_audio = np.clip(gain_audio, -1.0, 1.0)
                        self._audio_data.append(gain_audio.copy())
                    else:
                        # Count silence frames if we've already detected speech
                        if self._speech_detected:
                            self._silence_frames += 1
                        
                        # Still append some minimal audio to maintain continuity
                        self._audio_data.append(indata * 0.1)  # Reduced volume for noise
                    
                    # Auto-stop logic - if we have enough speech followed by enough silence
                    if (self._is_auto_stop and 
                        self._speech_detected and
                        self._speech_frames > self.config.min_speech_duration * self.config.sample_rate / self.config.block_size and
                        self._silence_frames > self.config.silence_duration * self.config.sample_rate / self.config.block_size):
                        print("\nDetected end of speech")
                        self._auto_stop_event.set()
            
            try:
                stream = sd.InputStream(callback=callback,
                                    channels=self.config.channels,
                                    samplerate=self.config.sample_rate,
                                    blocksize=self.config.block_size)
                stream.start()
                
                while True:
                    with self._thread_lock:
                        if not self._recording or self._stop_requested:
                            break
                            
                    try:
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        # Handle cancellation gracefully
                        print("\nRecording interrupted")
                        with self._thread_lock:
                            self._recording = False
                            self._stop_requested = True
                        raise
            finally:
                if stream:
                    stream.stop()
                    stream.close()
                
                # Cancel auto-stop task if it's still running
                if auto_stop_task and not auto_stop_task.done():
                    auto_stop_task.cancel()
                    try:
                        await auto_stop_task
                    except asyncio.CancelledError:
                        pass
            
            if not self._audio_data:
                raise AudioDeviceError("No audio data recorded")
            
            # Convert to WAV format with proper normalization
            audio_data = np.concatenate(self._audio_data, axis=0)
            
            # Normalize audio to use full dynamic range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
                
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.config.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            return byte_io.getvalue()
            
        except asyncio.CancelledError:
            # Recording was cancelled - handle gracefully
            print("\nRecording cancelled")
            self._recording = False
            
            # Still try to return what we have so far
            if self._audio_data:
                try:
                    audio_data = np.concatenate(self._audio_data, axis=0)
                    
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
                        
                    byte_io = io.BytesIO()
                    with wave.open(byte_io, 'wb') as wf:
                        wf.setnchannels(self.config.channels)
                        wf.setsampwidth(2)
                        wf.setframerate(self.config.sample_rate)
                        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                    
                    return byte_io.getvalue()
                except Exception:
                    # If processing fails, just propagate the cancellation
                    raise
            
            # Propagate the cancellation if we couldn't recover
            raise
        except Exception as e:
            self._recording = False
            raise AudioDeviceError(f"Error during audio recording: {str(e)}")
    
    def stop_recording(self) -> None:
        """Stop the audio recording."""
        with self._thread_lock:
            self._recording = False
            self._stop_requested = True
    
    async def play_audio(self, audio_data: bytes) -> None:
        """Play audio from bytes data."""
        playback_active = False
        local_play_obj = None
        
        try:
            # Stop any currently playing audio
            with self._thread_lock:
                if self._play_obj and self._play_obj.is_playing():
                    try:
                        self._play_obj.stop()
                    except Exception:
                        # Ignore errors when stopping previous playback
                        pass
            
            # Convert audio data to WAV format if it's not already
            try:
                # Try to read as WAV first
                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    sample_rate = wf.getframerate()
            except wave.Error:
                # If not WAV, assume it's raw PCM data
                sample_width = 2  # 16-bit audio
                channels = self.config.channels
                sample_rate = self.config.sample_rate
            
            # Create wave object
            wave_obj = sa.WaveObject(audio_data, channels, sample_width, sample_rate)
            local_play_obj = wave_obj.play()
            
            with self._thread_lock:
                self._play_obj = local_play_obj
                playback_active = True
            
            # Wait for playback to finish
            while True:
                with self._thread_lock:
                    current_obj = self._play_obj
                    is_active = playback_active and current_obj == local_play_obj
                    
                    if not is_active or not current_obj or not current_obj.is_playing():
                        break
                
                try:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    # Handle task cancellation gracefully
                    with self._thread_lock:
                        if current_obj and current_obj.is_playing():
                            try:
                                current_obj.stop()
                            except Exception:
                                pass
                    raise
            
        except asyncio.CancelledError:
            # Playback was cancelled - handle gracefully
            print("Audio playback interrupted")
            with self._thread_lock:
                if playback_active and local_play_obj and local_play_obj.is_playing():
                    try:
                        local_play_obj.stop()
                    except Exception:
                        pass
            raise
        except Exception as e:
            raise AudioDeviceError(f"Error during audio playback: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        with self._thread_lock:
            self._recording = False
            self._stop_requested = True
            current_play_obj = self._play_obj
            
        if current_play_obj and current_play_obj.is_playing():
            try:
                current_play_obj.stop()
            except Exception:
                pass
                
        try:
            # Clean up temporary files
            if os.path.exists(self._temp_dir):
                for file in os.listdir(self._temp_dir):
                    os.remove(os.path.join(self._temp_dir, file))
                os.rmdir(self._temp_dir)
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}") 