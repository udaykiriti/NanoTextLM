import torch
from config import ModelConfig
from model import TinyTechLM
from tokenizers import Tokenizer
import argparse
import os

def generate(model_path, tokenizer_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load components
    m_conf = ModelConfig()
    model = TinyTechLM(m_conf).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model checkpoint not found. Using random weights.")
    model.eval()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Encode
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=100)
        
    # Decode
    output = tokenizer.decode(generated[0].cpu().numpy())
    return output

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Interactive loop
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    print("TinyTechLM CLI (Type 'exit' to quit)")
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit"]: break
        
        response = generate(model_path, tokenizer_path, prompt)
        print(f"Model: {response}\n")
