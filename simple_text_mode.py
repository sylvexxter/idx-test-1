from questionnaire import questionnaire

def run_text_assessment():
    """
    Run a simple text-based risk assessment without voice integration
    or complex agent features.
    """
    print("=== Risk Assessment Application (Text Mode) ===\n")
    print("This is a simplified version that uses text-only interaction.\n")
    
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
        run_text_assessment()
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
    except Exception as e:
        print(f"\nError during assessment: {str(e)}")
    finally:
        print("\nApplication terminated.") 