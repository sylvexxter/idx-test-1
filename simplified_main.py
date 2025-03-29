"""
Simplified Risk Assessment Application
Using voice capabilities for a better user experience
"""

import asyncio
import signal
import sys
import os
import re
from questionnaire import questionnaire
from simplified_voice import SimplifiedVoice
from openai import OpenAI
from dotenv import load_dotenv
import sounddevice as sd

# Load environment variables
load_dotenv()

# Create the voice handler
voice = SimplifiedVoice()

# Create OpenAI client for the clarification agent
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flag to track if we're using voice
use_voice = False

# Setup the signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nCleaning up and exiting...")
    voice.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def is_clarification_question(text: str) -> bool:
    """
    Determine if the text is a clarification question.
    
    Args:
        text: The text to analyze
        
    Returns:
        True if the text appears to be a clarification question
    """
    # Normalize text
    text = text.strip().lower()
    
    # Check for question marks
    if "?" in text:
        return True
    
    # Check for question words at the beginning
    question_starters = ["what", "why", "how", "when", "where", "which", "who", "can", "could", "would", "should", "is", "are", "do", "does"]
    first_word = text.split()[0] if text.split() else ""
    if first_word in question_starters:
        return True
    
    # Check for phrases indicating confusion
    clarification_phrases = ["i don't understand", "please explain", "what do you mean", "not clear", "clarify", "elaborate", "confused about"]
    if any(phrase in text for phrase in clarification_phrases):
        return True
    
    return False

async def get_clarification(question: str, user_query: str) -> str:
    """
    Generate a clarification for the question based on the user's query.
    
    Args:
        question: The original question
        user_query: The user's clarification request
        
    Returns:
        A clarification response
    """
    print("Generating clarification...")
    
    try:
        # Use GPT-4o to generate a clarification
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for high-quality clarifications
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a cybersecurity risk assessment. Your task is to clarify questions when users are confused. Keep explanations clear, concise, and focused on helping the user understand the question. Don't answer the question for them, just clarify what it's asking."},
                {"role": "user", "content": f"Original question: {question}\n\nUser's clarification request: {user_query}\n\nPlease provide a clear explanation of what this question is asking."}
            ],
            temperature=0.7
        )
        
        clarification = response.choices[0].message.content
        return clarification
    except Exception as e:
        print(f"Error generating clarification: {str(e)}")
        return f"I apologize, but I couldn't generate a clarification. The question is asking about: {question}"

async def ask_question_with_voice(question: str) -> str:
    """
    Ask a question using voice and get response with advanced voice activity detection.
    
    This function uses automatic voice activity detection to determine when the user has 
    finished speaking, making the interaction more natural.
    """
    try:
        # Speak the question
        print("\nReading question aloud...")
        await voice.synthesize_speech(question)
        
        # Pause before recording to avoid capturing playback echo
        await asyncio.sleep(0.5)
        
        print("\n╭───────────────────────────────────────────────────╮")
        print("│            VOICE RECORDING ACTIVATED              │")
        print("├───────────────────────────────────────────────────┤")
        print("│ ✓ Natural voice activity detection                │")
        print("│ ✓ Take your time - the system will wait for you   │")
        print("│ ✓ Visual indicators show recording status         │")
        print("│ ✓ Recording stops after 1.5 seconds of silence    │")
        print("│ ✓ Automatic transcription without confirmation    │")
        print("│ ✓ For clarification, just ask about the question  │")
        print("│ ✓ Press Ctrl+C anytime to type your answer        │")
        print("╰───────────────────────────────────────────────────╯")
        
        while True:  # Loop to handle clarifications
            # Record the answer with automatic voice activity detection
            try:
                print("\nListening now - speak naturally and pause when you're done...")
                audio_data = await voice.record_audio(auto_stop=True)
            except KeyboardInterrupt:
                print("\n─── Recording stopped manually ───")
                text = input("Please type your response or question: ").strip()
            except (RuntimeError, sd.PortAudioError) as e:
                # Handle PortAudio and other runtime errors
                print(f"\n⚠️  Microphone access error: {str(e)}")
                print("Please check your microphone connection or permissions.")
                print("Falling back to text input for this question.")
                text = input("Your answer: ").strip()
            else:
                print("\n─── Processing your response ───")
                # Check if we got any audio
                if not audio_data or len(audio_data) < 100:
                    print("⚠️  No speech detected or recording too short")
                    await voice.synthesize_speech("I didn't hear your response. Please try again or type your answer.")
                    text = input("Your answer (or type 'retry' to speak again): ").strip()
                    if text.lower() == 'retry':
                        continue
                else:
                    # Transcribe the audio if recording completed
                    print("Transcribing speech...")
                    text = await voice.transcribe_audio(audio_data)
                    
                    if not text:
                        print("⚠️  Speech detection failed. No transcription available.")
                        await voice.synthesize_speech("I couldn't understand what you said. Please type your answer instead.")
                        text = input("Your answer: ").strip()
                    else:
                        # Display transcription without asking for confirmation
                        print(f"✓ Transcribed: \"{text}\"")
                        print("Continuing with transcription...")
                        
                        # No user confirmation needed - proceed directly with the transcribed text
            
            # Check if this is a clarification question
            if is_clarification_question(text):
                print("\n─── Clarification Requested ───")
                
                # Get clarification
                clarification = await get_clarification(question, text)
                print(f"Clarification: {clarification}")
                
                # Provide the clarification using voice if enabled
                try:
                    await voice.synthesize_speech(clarification)
                except Exception as voice_error:
                    print(f"Unable to speak clarification: {str(voice_error)}")
                
                # Ask the original question again
                print("\n─── Original Question ───")
                print(question)
                try:
                    await voice.synthesize_speech("Let me repeat the original question: " + question)
                except Exception as voice_error:
                    print(f"Unable to speak question: {str(voice_error)}")
                
                # Continue the loop to get a new response
                continue
            
            # If we get here, it's a valid answer, not a clarification request
            return text
            
    except Exception as e:
        print(f"\n⚠️  Error during voice interaction: {type(e).__name__}: {str(e)}")
        print("Falling back to text input.")
        return input("Your answer: ").strip()

async def ask_question_text_only(question: str) -> str:
    """Ask a question in text-only mode and handle clarifications."""
    print(f"Question: {question}")
    print("TIP: You can ask for clarification by typing 'what does this mean?' or similar questions.")
    
    while True:  # Loop to handle clarifications
        response = input("Your answer: ").strip()
        
        # Check if this is a clarification question
        if is_clarification_question(response):
            print("Detected a clarification request.")
            
            # Get clarification
            clarification = await get_clarification(question, response)
            print(f"Clarification: {clarification}")
            
            # Ask the original question again
            print("\nOriginal question:", question)
            
            # Continue the loop to get a new response
            continue
        
        # If we get here, it's a valid answer, not a clarification
        return response

async def run_assessment():
    """Run the main risk assessment with optional voice interaction."""
    global use_voice
    
    print("=== Risk Assessment Application ===\n")
    print("Welcome to the Risk Assessment Application!")
    print("This tool allows you to complete a cybersecurity risk assessment questionnaire.")
    print("\nIMPORTANT FEATURES:")
    print("1. You can ask for clarification on any question by:")
    print("   - Asking 'What does this mean?' or any question about the original question")
    print("   - Starting your response with question words like 'how', 'what', 'why', etc.")
    print("   - Including phrases like 'I don't understand' or 'please explain'")
    print("2. The system will detect when you're asking for clarification and provide help")
    print("3. After clarification, you'll be prompted to answer the original question again")
    
    # Check audio devices before asking about voice mode
    audio_available = voice.check_audio_devices()
    
    if not audio_available:
        print("\n⚠️  Audio input devices not available or not working properly.")
        print("Continuing in text-only mode.")
        use_voice = False
    else:
        # Ask if the user wants to use voice
        voice_choice = input("\nWould you like to use voice interaction? (yes/no): ").lower()
        use_voice = voice_choice.startswith('y')
    
    if use_voice:
        print("\nVOICE MODE INSTRUCTIONS:")
        print("• The system uses natural voice activity detection")
        print("• Speak freely without worrying about timing or pauses")
        print("• The system will wait patiently for you to complete your thoughts")
        print("• Recording stops after 1.5 seconds of silence")
        print("• Transcription is processed automatically without confirmation")
        print("• Visual indicators show your progress:")
        print("  - [LISTENING] ······    : Waiting for you to speak")
        print("  - [SPEAKING] ████████   : Recording your voice with volume indicator")
        print("  - [SILENCE 50%] ██████  : Progress toward automatic stopping")
        print("• Press Ctrl+C anytime to stop recording and type your response")
        print("• You can ask for clarification by speaking a question about the question")
        
        try:
            # Test voice capabilities
            print("\nTesting voice capabilities...")
            await voice.synthesize_speech("Voice interaction enabled with natural silence detection. Speak freely and take your time. Your responses will be automatically transcribed and processed without asking for confirmation.")
        except Exception as e:
            print(f"Error initializing voice: {str(e)}")
            print("Falling back to text-only mode.")
            use_voice = False
    else:
        print("\nTEXT MODE INSTRUCTIONS:")
        print("- Type your answers after each question")
        print("- To get clarification, just type a question about the question")
        print("- After receiving clarification, you'll be prompted to answer again")
    
    # Process each question in the questionnaire
    for domain, questions in questionnaire.items():
        print(f"\n--- Domain: {domain} ---")
        
        for question in questions:
            print(f"\nQuestion: {question}")
            
            if use_voice:
                answer = await ask_question_with_voice(question)
            else:
                answer = await ask_question_text_only(question)
                
            print(f"Answer recorded: {answer}")
            
            # Give feedback to the user
            if use_voice:
                await voice.synthesize_speech("Thank you for your answer.")
            else:
                print("Thank you for your answer.")
    
    # Completion message
    completion_msg = "Assessment completed. Thank you for your participation."
    print(f"\n{completion_msg}")
    
    if use_voice:
        await voice.synthesize_speech(completion_msg)

if __name__ == "__main__":
    # Set up asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the assessment
        loop.run_until_complete(run_assessment())
    except KeyboardInterrupt:
        print("\nAssessment interrupted by user.")
    except Exception as e:
        print(f"\nError during assessment: {str(e)}")
        # Try text-only mode as a fallback
        try:
            use_voice = False
            print("\nRetrying in text-only mode...")
            loop.run_until_complete(run_assessment())
        except Exception as e2:
            print(f"Fatal error in fallback mode: {str(e2)}")
    finally:
        # Clean up resources
        voice.cleanup()
        loop.close()
        print("\nApplication terminated.") 