from __future__ import annotations
import asyncio
import uuid
import signal
from typing import Optional, Union, List
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    handoff,
    trace,
    TResponseInputItem,
    MessageOutputItem,
    HandoffOutputItem,
    ItemHelpers,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from questionnaire import questionnaire
from voice_pipeline import VoiceRiskAssessment
from audio_handler import AudioDeviceError

# -----------------------
# Shared Context Model
# -----------------------

class RiskAssessmentContext(BaseModel):
    current_domain: Optional[str] = None
    current_question: Optional[str] = None
    user_response: Optional[str] = None
    use_voice: bool = False
    continuous_mode: bool = False
    
# -----------------------
# Global State
# -----------------------

class GlobalState:
    voice_response_received = asyncio.Event()
    current_voice_response = ""
    conversation_active = True

global_state = GlobalState()

# -----------------------
# Agent Definitions
# -----------------------

# Main Controller: Orchestrates the conversation by delegating to specialized agents.
main_controller = Agent[RiskAssessmentContext](
    name="MainController",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Main Controller for a risk assessment chatbot. Your role is to coordinate the conversation.
When a new question is to be asked, transfer control to the Question Agent.
When a user answer is received that appears to be a clarification (for example, a question or a request for more details), 
you will transfer control to the Clarification Agent.
You will not hand over to the Clarification Agent until after the question has been asked and the user replies.
After the Clarification Agent provides an explanation, re-ask the exact original question.
Once a final answer is received (one that does not look like a clarification), record the answer and proceed to the next question.
DO NOT CHANGE the current question in context at any point in the assessment.""",
    handoffs=[],  # Set below.
)

# Question Agent: Outputs the current question verbatim.
question_agent = Agent[RiskAssessmentContext](
    name="QuestionAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Question Agent. Your sole responsibility is to output exactly the current question stored in the context.
Do not add, modify, or ask any additional questions. Simply output the exact text of the current question.
WAIT for the user's answer; do not prompt or ask for clarification.
After outputting the question, immediately transfer control back to the Main Controller.""",
    handoffs=[handoff(agent=main_controller)]
)

# Clarification Agent: Provides a concise explanation of the current question.
clarification_agent = Agent[RiskAssessmentContext](
    name="ClarificationAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Clarification Agent. Using the user_response and the current_question from the context, provide a brief, clear explanation 
of what the question is asking so the user can respond appropriately.
DO NOT RETURN or modify the current question – simply explain it in simple language.
After providing your explanation, immediately transfer control back to the Main Controller.""",
    handoffs=[handoff(agent=main_controller)]
)

# Configure the Main Controller to hand off to the specialized agents.
main_controller.handoffs = [
    handoff(agent=question_agent),
    handoff(agent=clarification_agent),
]

# -----------------------
# Helper: Classify Answer
# -----------------------

def classify_answer(answer: str) -> str:
    """
    Returns "clarification" if the answer appears to be a clarifying question,
    and "valid" otherwise.
    """
    answer = answer.strip().lower()
    question_words = {"what", "why", "how", "when", "where", "which", "who"}
    if "?" in answer:
        return "clarification"
    if answer.split() and answer.split()[0] in question_words:
        return "clarification"
    return "valid"

# -----------------------
# Voice Integration
# -----------------------

voice_workflow = VoiceRiskAssessment()

async def handle_voice_response(response: str) -> None:
    """
    Callback function for the continuous voice mode.
    Saves the transcribed response and signals that it has been received.
    """
    if not global_state.conversation_active:
        return
        
    global_state.current_voice_response = response
    global_state.voice_response_received.set()
    
async def get_voice_input(continuous_mode: bool = False) -> str:
    """
    Get user input through voice.
    
    Args:
        continuous_mode: If True, uses the continuous listening mode
                        with automatic speech detection.
    """
    try:
        if continuous_mode:
            print("\nWaiting for voice input in continuous mode...")
            # Reset the event
            global_state.voice_response_received.clear()
            
            # Wait for the response to be received through the callback
            try:
                await asyncio.wait_for(global_state.voice_response_received.wait(), timeout=60.0)
                
                # Return the captured response
                return global_state.current_voice_response
            except asyncio.TimeoutError:
                print("\nTimeout waiting for voice input. Please try again.")
                return input("Your answer (text fallback): ").strip()
        else:
            # Traditional mode with manual Enter key
            print("Listening... (speak your answer)")
            print("Press Enter when you're done speaking...")
            
            try:
                # Start recording
                await voice_workflow.start_recording(auto_stop=False)
                
                # Wait for Enter key
                await asyncio.get_event_loop().run_in_executor(None, input)
                
                # Stop recording and get audio data
                audio_data = await voice_workflow.stop_recording()
                
                # Process the audio
                text = await voice_workflow.process_audio(audio_data)
                
                # Validate transcription result
                if not text or text.strip() == "":
                    print("No speech detected. Please try again with a clearer voice.")
                    print("Falling back to text input...")
                    return input("Your answer: ").strip()
                    
                print(f"Transcription complete: '{text}'")
                print("If this is not what you said, you can type your answer instead.")
                confirm = input("Use this transcription? (Y/n): ").lower()
                if confirm.startswith('n'):
                    return input("Your answer: ").strip()
                return text
            except AttributeError as e:
                print(f"Voice module implementation error: {str(e)}")
                print("Voice input not supported. Using text-only input.")
                return input("Your answer (text fallback): ").strip()
    
    except AudioDeviceError as e:
        print(f"Audio device error: {str(e)}")
        print("Falling back to text input...")
        return input("Your answer: ").strip()
    except Exception as e:
        print(f"Unexpected error during voice input: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Falling back to text input...")
        return input("Your answer: ").strip()

async def output_voice(text: str) -> None:
    """Output text through voice."""
    try:
        print(f"[Text output: {text}]")  # Always show text output as backup
        await voice_workflow.synthesize_speech(text)
    except asyncio.CancelledError:
        print("\nVoice output interrupted")
    except AudioDeviceError as e:
        print(f"Audio device error during voice output: {str(e)}")
    except AttributeError as e:
        print(f"Voice module implementation error: {str(e)}")
        print("Voice output not supported in this configuration. Using text-only output.")
    except Exception as e:
        print(f"Unexpected error during voice output: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

# -----------------------
# Cleanup Handler
# -----------------------

def cleanup_handler(signum, frame):
    """Handle cleanup when the program is terminated."""
    print("\nCleaning up...")
    global_state.conversation_active = False
    voice_workflow.cleanup()
    exit(0)

# Register cleanup handler
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)

# -----------------------
# Main Conversation Loop
# -----------------------

async def run_question(domain: str, question: str, use_voice: bool = False, continuous_mode: bool = False):
    # Create a fresh context for this question.
    context = RiskAssessmentContext(
        current_domain=domain,
        current_question=question,
        user_response=None,
        use_voice=use_voice,
        continuous_mode=continuous_mode
    )
    conversation_id = uuid.uuid4().hex[:16]
    current_agent = main_controller

    try:
        # Step 1: Ask the question using the Question Agent.
        input_items = [{"content": context.current_question, "role": "system"}]
        with trace("Risk Assessment", group_id=conversation_id):
            result = await Runner.run(main_controller, input_items, context=context)
        
        asked_question = ""
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                asked_question = ItemHelpers.text_message_output(item).strip()
                break
        if asked_question != context.current_question.strip():
            asked_question = context.current_question.strip()
        
        print("Question:", asked_question)
        if context.use_voice:
            try:
                await output_voice(asked_question)
            except asyncio.CancelledError:
                print("\nQuestion speech output interrupted")
                # Continue with the conversation despite the interruption
        
        for item in result.new_items:
            if isinstance(item, HandoffOutputItem):
                print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
        
        current_agent = result.last_agent

        # Step 2: Wait for the user's answer.
        try:
            if context.use_voice:
                user_ans = await get_voice_input(continuous_mode=context.continuous_mode)
                print("Voice input received:", user_ans)
            else:
                user_ans = input("Your answer: ").strip()
        except asyncio.CancelledError:
            print("\nUser input interrupted, using empty response")
            user_ans = ""
        
        context.user_response = user_ans
        classification = classify_answer(user_ans)

        # Loop until a valid answer is received.
        while classification == "clarification":
            # Step 3: Trigger the Clarification Agent.
            clar_prompt = f"Clarify the question: {context.current_question}\nUser asked: {user_ans}"
            input_items = [{"content": clar_prompt, "role": "user"}]
            with trace("Risk Assessment", group_id=conversation_id):
                result = await Runner.run(main_controller, input_items, context=context)
            
            for item in result.new_items:
                if isinstance(item, MessageOutputItem):
                    clarification = ItemHelpers.text_message_output(item)
                    print("Clarification:", clarification)
                    if context.use_voice:
                        try:
                            await output_voice(clarification)
                        except asyncio.CancelledError:
                            print("\nClarification speech output interrupted")
                            # Continue despite the interruption
                elif isinstance(item, HandoffOutputItem):
                    print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
            
            # Step 4: Re-ask the original question exactly.
            input_items = [{"content": context.current_question, "role": "system"}]
            with trace("Risk Assessment", group_id=conversation_id):
                result = await Runner.run(main_controller, input_items, context=context)
            
            for item in result.new_items:
                if isinstance(item, MessageOutputItem):
                    asked_question = ItemHelpers.text_message_output(item).strip()
                    if asked_question != context.current_question.strip():
                        asked_question = context.current_question.strip()
                    print("Question re-asked:", asked_question)
                    if context.use_voice:
                        try:
                            await output_voice(asked_question)
                        except asyncio.CancelledError:
                            print("\nQuestion re-ask speech output interrupted")
                            # Continue despite the interruption
                elif isinstance(item, HandoffOutputItem):
                    print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
            
            # Step 5: Get a new answer.
            try:
                if context.use_voice:
                    user_ans = await get_voice_input(continuous_mode=context.continuous_mode)
                    print("Voice input received:", user_ans)
                else:
                    user_ans = input("Your answer after clarification: ").strip()
            except asyncio.CancelledError:
                print("\nUser input interrupted, using empty response")
                user_ans = ""
            
            context.user_response = user_ans
            classification = classify_answer(user_ans)
        
        # Step 6: Accept the valid answer.
        print("Answer accepted:", user_ans)
        if context.use_voice:
            try:
                await output_voice("Answer accepted.")
            except asyncio.CancelledError:
                print("\nAnswer acceptance speech output interrupted")
                # Continue despite the interruption
        print("Final answer for question:", context.current_question, "->", context.user_response)
    
    except asyncio.CancelledError:
        print("\nQuestion processing interrupted")
        raise
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        if context.use_voice:
            try:
                await output_voice("Sorry, there was an error processing this question. Moving to the next one.")
            except:
                # Ignore any errors in the error handler
                pass

async def main():
    try:
        print("Starting Risk Assessment application...")
        
        # Ask user preference for voice interaction
        use_voice_input = input("Would you like to use voice interaction? (yes/no): ").lower().startswith('y')
        
        continuous_mode = False
        if use_voice_input:
            try:
                print("Initializing voice module...")
                continuous_mode_input = input("Would you like hands-free continuous voice mode? (yes/no): ").lower().startswith('y')
                if continuous_mode_input:
                    continuous_mode = True
                    try:
                        # Start the continuous listening mode
                        print("Starting continuous listening mode...")
                        await voice_workflow.start_continuous_listening(handle_voice_response)
                        
                        # Brief delay to allow the system to start up
                        await asyncio.sleep(1)
                        
                        # Introduce the system
                        intro_text = ("Welcome to the hands-free Risk Assessment system. "
                                    "I will ask you questions about cybersecurity practices. "
                                    "Simply speak your answers naturally and pause when you're done. "
                                    "I'll automatically detect when you've finished speaking.")
                        print("\n" + intro_text)
                        
                        try:
                            await output_voice(intro_text)
                        except Exception as e:
                            print(f"Error during introduction speech: {str(e)}")
                            print("Continuing with text-only output for introduction.")
                    except Exception as e:
                        print(f"Error starting continuous mode: {str(e)}")
                        print("Falling back to manual voice mode.")
                        continuous_mode = False
            except Exception as e:
                print(f"Error initializing voice: {str(e)}")
                print("Falling back to text-only mode.")
                use_voice_input = False
        
        # Flatten the questionnaire into a list of (domain, question) pairs.
        questions_list = []
        for domain, qs in questionnaire.items():
            for q in qs:
                questions_list.append((domain, q))
        
        for domain, q in questions_list:
            try:
                await run_question(domain, q, use_voice_input, continuous_mode)
                print()  # Blank line for readability.
            except Exception as e:
                print(f"Error processing question '{q}': {str(e)}")
                print("Moving to the next question.")
    
    finally:
        # Ensure cleanup happens
        print("Cleaning up resources...")
        global_state.conversation_active = False
        try:
            voice_workflow.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Create a new event loop for the main thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the main function in the newly created loop
        print("=== Risk Assessment Application ===")
        print("Starting main loop...")
        
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"\nApplication error: {str(e)}")
            print("Trying to run in text-only fallback mode...")
            
            # Try again with a simpler approach
            try:
                # Ask each question in text mode
                for domain, questions in questionnaire.items():
                    print(f"\n--- Domain: {domain} ---")
                    for question in questions:
                        print(f"\nQuestion: {question}")
                        answer = input("Your answer: ").strip()
                        print(f"Answer recorded: {answer}")
            except Exception as e2:
                print(f"Fatal error in fallback mode: {str(e2)}")
    finally:
        # Clean up the loop
        try:
            loop.close()
        except:
            pass
        
        print("\nApplication terminated.")
