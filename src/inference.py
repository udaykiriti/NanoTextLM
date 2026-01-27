import torch
from config import ModelConfig
from model import NanoTextLM
from tokenizers import Tokenizer
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

# Initialize Rich Console
console = Console()

def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
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
            state_dict = torch.load(model_path, map_location=device)
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            console.print(f"[bold green]Loaded model from {model_path}[/bold green]")
        except Exception as e:
             console.print(f"[bold red]Error loading weights: {e}[/bold red]")
             console.print("[yellow]Using random weights for testing...[/yellow]")
    else:
        console.print("[bold red]Checkpoint not found. Using random weights.[/bold red]")
    
    model.eval()
    
    # Optimization: Compile
    if hasattr(torch, "compile"):
        console.print("[yellow]Compiling model... (this may take a moment)[/yellow]")
        model = torch.compile(model)
    
    # Load Tokenizer
    if not os.path.exists(tokenizer_path):
        console.print("[bold red]Tokenizer not found. Please run training first.[/bold red]")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    console.print(Panel("Type 'exit' to quit. Context is maintained.", title="NanoTextLM Chat", border_style="bold magenta"))

    # Chat History
    history_ids = None
    
    while True:
        prompt = console.input("[bold cyan]User > [/bold cyan]")
        if prompt.lower() in ["exit", "quit"]:
            break
            
        # Encode new user input
        new_ids = tokenizer.encode(prompt).ids
        new_idx = torch.tensor([new_ids], dtype=torch.long, device=device)
        
        # Append to history
        if history_ids is None:
            history_ids = new_idx
        else:
            history_ids = torch.cat((history_ids, new_idx), dim=1)
            
        # Context Window Check (Simple Truncation)
        if history_ids.size(1) > m_conf.max_seq_len - 100:
             # Keep last N tokens
             history_ids = history_ids[:, -(m_conf.max_seq_len - 100):]
        
        generated_text = ""
        
        with Live(Panel("", title="NanoTextLM", border_style="blue"), refresh_per_second=10) as live:
            # Generate response (streaming)
            # We must be careful: generate_stream returns ALL tokens in the new sequence or just new ones?
            # My implementation yields `idx_next`. So it's just the new token.
            
            for token_idx in model.generate_stream(history_ids, max_new_tokens=150, temperature=0.8, top_p=0.9):
                token_val = token_idx.item()
                text_chunk = tokenizer.decode([token_val])
                generated_text += text_chunk
                live.update(Panel(generated_text, title="NanoTextLM", border_style="green"))
                
                # Append to history
                history_ids = torch.cat((history_ids, token_idx), dim=1)

        console.print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")
