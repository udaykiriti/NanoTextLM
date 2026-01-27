import torch
from config import ModelConfig
from model import NanoTextLM
from tokenizers import Tokenizer
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

# Initialize Rich Console
console = Console()

def stream_generate(model, tokenizer, prompt, device, max_new_tokens=100):
    # Encode
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Placeholder for accumulated text
    generated_text = ""
    
    with Live(Panel("", title="NanoTextLM", border_style="blue"), refresh_per_second=10) as live:
        # Stream tokens
        for token_idx in model.generate_stream(idx, max_new_tokens=max_new_tokens):
            # Decode single token (careful with subwords, but simple decoding usually works for display)
            # A better approach for BPE is to decode the whole sequence and take the diff, 
            # but for simplicity we decode the single token here. 
            # Note: This might look weird for partial bytes, but usually fine for English.
            
            token_val = token_idx.item()
            decoded_part = tokenizer.decode([token_val])
            
            generated_text += decoded_part
            
            # Update the panel with accumulated text
            live.update(Panel(generated_text, title="NanoTextLM", border_style="green"))

    return generated_text

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
    # If final model doesn't exist, try best model
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
        
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"[bold yellow]Loading NanoTextLM on {device}...[/bold yellow]")

    # Load Model
    m_conf = ModelConfig()
    model = NanoTextLM(m_conf).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            console.print(f"[bold green]Loaded model from {model_path}[/bold green]")
        except Exception as e:
             console.print(f"[bold red]Error loading weights: {e}[/bold red]")
             console.print("[yellow]Using random weights for testing...[/yellow]")
    else:
        console.print("[bold red]Checkpoint not found. Using random weights.[/bold red]")
    
    model.eval()
    
    # Load Tokenizer
    if not os.path.exists(tokenizer_path):
        console.print("[bold red]Tokenizer not found. Please run training first.[/bold red]")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    console.print(Panel("Type 'exit' to quit. Enjoy!", title="Welcome", border_style="bold magenta"))

    while True:
        prompt = console.input("[bold cyan]User > [/bold cyan]")
        if prompt.lower() in ["exit", "quit"]:
            break
            
        stream_generate(model, tokenizer, prompt, device)
        console.print() # Newline

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")