from inference import generate
import os
import argparse

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    print("-" * 30)
    print("TinyTechLM CLI v1.0")
    print("-" * 30)
    print(f"Model: {MODEL_PATH}")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            print("TinyTechLM: ", end="", flush=True)
            # generate function returns a string logic, simple print
            # ideally we'd stream, but inference.py currently returns full string
            try:
                response = generate(MODEL_PATH, TOKENIZER_PATH, prompt)
                print(response)
            except Exception as e:
                print(f"[Error: {e}]")
                
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
