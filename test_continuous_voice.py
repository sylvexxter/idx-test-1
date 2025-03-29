"""
Test utility for continuous voice interaction.
Run with: python test_continuous_voice.py
"""

import asyncio
from voice_pipeline import VoiceRiskAssessment
from audio_handler import AudioDeviceError
from dotenv import load_dotenv

async def handle_voice_response(text: str) -> None:
    """Callback function for speech recognition."""
    print(f"\n=== Speech Detected ===\nUser said: '{text}'")
    
    # Echo the response back
    if text.strip().lower() not in ["exit", "quit", "stop"]:
        print("AI would respond to this input...")
        await voice.synthesize_speech(f"I heard you say: {text}")

async def test_continuous_voice():
    """Test continuous voice interaction."""
    global voice
    
    print("\n=== Continuous Voice Interaction Test ===\n")
    print("This test simulates a hands-free conversation.")
    print("Speak naturally and pause when you're finished speaking.")
    print("The system will automatically detect when you've stopped speaking.")
    print("Say 'exit', 'quit', or 'stop' to end the test.\n")
    
    try:
        # Initialize continuous listening
        await voice.start_continuous_listening(handle_voice_response)
        
        # Keep the program running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        # Clean up resources
        voice.cleanup()
        print("\nResources cleaned up.")

if __name__ == "__main__":
    # Load environment variables (.env file)
    load_dotenv()
    
    # Initialize voice workflow
    voice = VoiceRiskAssessment()
    
    # Run the test
    asyncio.run(test_continuous_voice()) 