from agents import Agent, Runner
from questionnaire import questionnaire

# =======================
# Main Controller Agent
# =======================

MAIN_CONTROLLER_INSTRUCTIONS = (
    "You are the Main Controller for a company risk assessment chatbot. "
    "You are provided with a predefined questionnaire divided into multiple domains. "
    "Your role is to orchestrate the conversation as follows:\n"
    "1. Present each question by calling the Question Agent.\n"
    "2. Accept user responses that must begin with YES, NO, or NOT APPLICABLE (with optional additional context).\n"
    "3. If the user asks for clarification, hand off that query to the Clarification Agent and then re-ask the original question.\n"
    "4. Use a dedicated Routing Agent to determine if the input is valid, a clarification request, or off-topic.\n"
    "5. Maintain deterministic, on-script behavior throughout the conversation.\n"
    "Note: This prototype does not store responses or generate a final report; it focuses solely on interaction."
)

main_controller = Agent(
    name="MainController",
    instructions=MAIN_CONTROLLER_INSTRUCTIONS
)

# =======================
# Routing Agent
# =======================

routing_agent = Agent(
    name="RoutingAgent",
    instructions=(
        "You are the Routing Agent for a risk assessment chatbot. "
        "Your task is to analyze user input and classify it as one of the following: "
        "'valid' if the input is a valid answer (i.e., it begins with YES, NO, or NOT APPLICABLE, possibly with additional context), "
        "'clarification' if the user is asking for more details or explanation (for example, if the input contains words like 'clarify', 'explain', or 'what do you mean'), "
        "or 'off-topic' if the input does not conform to these expected responses. "
        "Return only one of these words: valid, clarification, or off-topic."
    )
)

# =======================
# Question Agent
# =======================

question_agent = Agent(
    name="QuestionAgent",
    instructions=(
        "You are the Question Agent. When provided with a risk assessment question as input, "
        "output the question verbatim in a clear and concise manner. "
        "Do not include any additional commentary or deviation from the original question."
    )
)

# =======================
# Clarification Agent
# =======================

clarification_agent = Agent(
    name="ClarificationAgent",
    instructions=(
        "You are the Clarification Agent. When given a clarifying query, "
        "provide a concise explanation about what the question is asking and any additional context that may help the user answer the question. "
        "Make sure to use all provided context to improve clarity."
    )
)

# =======================
# Main Conversation Loop
# =======================

def run_risk_assessment(questionnaire, question_agent, clarification_agent, routing_agent):
    """
    Orchestrates the risk assessment conversation.
    
    Parameters:
      questionnaire: dict where each key is a domain and the value is a list of questions.
      question_agent: Agent that asks the given question.
      clarification_agent: Agent that provides clarifications when needed.
      routing_agent: Agent that routes user inputs to determine their type.
    """
    for domain, questions in questionnaire.items():
        print(f"\n--- Domain: {domain} ---")
        for question in questions:
            # Ask the question via the Question Agent.
            q_result = Runner.run_sync(question_agent, question)
            print("Question:", q_result.final_output, "\n")
            
            # Initialize clarification context for the current question.
            clarification_context = ""
            user_input = input("Your answer: ").strip()+ "\n"
            
            # Loop until a valid answer is provided.
            while True:
                # Use the Routing Agent to classify the input.
                route_result = Runner.run_sync(routing_agent, user_input)
                route_type = route_result.final_output.lower()  # expected: "valid", "clarification", or "off-topic"
                
                if route_type == "valid":
                    break  # A valid answer was given.
                elif route_type == "off-topic":
                    print("REMINDER: Please answer with YES, NO, or NOT APPLICABLE, or ask for clarification if needed.", "\n")
                    user_input = input("Now please provide your answer: ").strip()+ "\n"
                elif route_type == "clarification":
                    # Append the clarification request to accumulated context.
                    if clarification_context:
                        clarification_context += "\n" + user_input
                    else:
                        clarification_context = user_input
                    # Build a prompt that includes the original question and the accumulated clarification context.
                    clarification_prompt = f"Question: {question}\nClarification request: {clarification_context}"
                    c_result = Runner.run_sync(clarification_agent, clarification_prompt)
                    print("Clarification:", c_result.final_output)
                    user_input = input("Now please provide your answer: ").strip()
            
            print("Answer accepted:", user_input, "\n")
        
        next_domain = input("Do you want to proceed to the next domain? (YES/NO): ").strip().upper()
        if next_domain != "YES":
            break

    print("\nRisk assessment conversation complete.")

# =======================
# Example Usage
# =======================

if __name__ == "__main__":    
    # Run the risk assessment process.
    run_risk_assessment(questionnaire, question_agent, clarification_agent, routing_agent)
