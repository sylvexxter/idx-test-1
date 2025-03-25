from __future__ import annotations
import asyncio
import uuid
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
from questionnaire import questionnaire  # Expected to be a dict: {domain: [questions, ...]}

# -----------------------
# Shared Context Model
# -----------------------

class RiskAssessmentContext(BaseModel):
    current_domain: str | None = None
    current_question: str | None = None
    user_response: str | None = None

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
# Main Conversation Loop
# -----------------------

async def run_question(domain: str, question: str):
    # Create a fresh context for this question.
    context = RiskAssessmentContext(
        current_domain=domain,
        current_question=question,
        user_response=None,
    )
    conversation_id = uuid.uuid4().hex[:16]
    current_agent = main_controller

    # Step 1: Ask the question using the Question Agent.
    # Use a fresh input with only the system message.
    input_items = [{"content": context.current_question, "role": "system"}]
    with trace("Risk Assessment", group_id=conversation_id):
        result = await Runner.run(main_controller, input_items, context=context)
    # Extract the output; if it doesn't match exactly the current question, override it.
    asked_question = ""
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            asked_question = ItemHelpers.text_message_output(item).strip()
            break
    if asked_question != context.current_question.strip():
        # Override any extra output with the original question.
        asked_question = context.current_question.strip()
    print("Question:", asked_question)
    for item in result.new_items:
        if isinstance(item, HandoffOutputItem):
            print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
    # Ensure control returns to Main Controller.
    current_agent = result.last_agent

    # Step 2: Wait for the user's answer.
    user_ans = input("Your answer: ").strip()
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
                print("Clarification:", ItemHelpers.text_message_output(item))
            elif isinstance(item, HandoffOutputItem):
                print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
        # Step 4: Re-ask the original question exactly.
        input_items = [{"content": context.current_question, "role": "system"}]
        with trace("Risk Assessment", group_id=conversation_id):
            result = await Runner.run(main_controller, input_items, context=context)
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                # Again, override any deviation.
                asked_question = ItemHelpers.text_message_output(item).strip()
                if asked_question != context.current_question.strip():
                    asked_question = context.current_question.strip()
                print("Question re-asked:", asked_question)
            elif isinstance(item, HandoffOutputItem):
                print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
        # Step 5: Get a new answer.
        user_ans = input("Your answer after clarification: ").strip()
        context.user_response = user_ans
        classification = classify_answer(user_ans)
    
    # Step 6: Accept the valid answer.
    print("Answer accepted:", user_ans)
    print("Final answer for question:", context.current_question, "->", context.user_response)

async def main():
    # Flatten the questionnaire into a list of (domain, question) pairs.
    questions_list = []
    for domain, qs in questionnaire.items():
        for q in qs:
            questions_list.append((domain, q))
    for domain, q in questions_list:
        await run_question(domain, q)
        print()  # Blank line for readability.

if __name__ == "__main__":
    asyncio.run(main())
