# Minimal text-only risk assessment application

# Define the questionnaire directly in this file to minimize dependencies
questionnaire = {
    "Training and awareness": [
        "Security Awareness Training: The organisation has put in place cybersecurity awareness training for all employees to ensure that employees are aware of the security practices and behaviour expected of them. Organisations may meet this requirement in different ways, e.g., provide self-learning materials for employees or engaging external training providers. Employees are equipped with the security knowledge and awareness to identify and mitigate against cyber threats?"
    ],
    "Asset management": [
        "An up-to-date asset inventory: Is there an up-to-date inventory of all the hardware and software assets in the organisation.",
        "Cloud Usage: Does the organization use certified cloud applications and cloud instances?",
        "Usage of unauthorized or End of Life assets: Does the organization replace hardware and software assets that are unauthorised or have reached their respective End of Life Support?"
    ],
    "Data protection": [
        "Inventory of Data: Does the organisation identifies and maintains an inventory of business-critical data in the organisation?",
        "Data Protection: The organisation establishes a process to protect its business-critical data, e.g., password protected documents, encryption of personal data (at rest) and/or emails."
    ]
}

def run_minimal_assessment():
    """Run a minimal text-based risk assessment."""
    print("=== Risk Assessment Application (Minimal Text Mode) ===\n")
    print("This is a simplified version with no external dependencies.\n")
    
    # Process each domain and question
    for domain, questions in questionnaire.items():
        print(f"\n--- Domain: {domain} ---")
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = input("Your answer: ").strip()
            print(f"Answer recorded: {answer}\n")
    
    print("\nAssessment completed. Thank you for your participation.")

if __name__ == "__main__":
    try:
        run_minimal_assessment()
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
    except Exception as e:
        print(f"\nError during assessment: {str(e)}")
    finally:
        print("\nApplication terminated.") 