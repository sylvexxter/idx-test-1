"""
Test utility for audio recording and transcription.
Run with: python test_audio.py
"""

import asyncio
from voice_pipeline import VoiceRiskAssessment
from audio_handler import AudioDeviceError
from dotenv import load_dotenv

async def test_audio_transcription():
    """Test audio recording and transcription."""
    print("\n=== Audio Recording and Transcription Test ===\n")
    
    # Load environment variables (.env file)
    load_dotenv()
    
    # Initialize voice workflow
    voice = VoiceRiskAssessment()
    
    try:
        # Start recording
        print("Recording audio... (speak clearly and press Enter when done)")
        await voice.start_recording()
        
        # Wait for Enter key
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Stop recording and get audio data
        print("Stopping recording...")
        audio_data = await voice.stop_recording()
        
        # Process the audio
        print("Transcribing audio...")
        text = await voice.process_audio(audio_data)
        
        # Print results
        print("\n=== Results ===")
        print(f"Transcribed text: '{text}'")
        print("\nTest completed successfully!")
        
    except AudioDeviceError as e:
        print(f"Audio device error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        # Clean up resources
        voice.cleanup()
        print("\nResources cleaned up.")

if __name__ == "__main__":
    asyncio.run(test_audio_transcription()) 