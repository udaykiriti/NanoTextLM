import torch
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from runtime import load_inference_resources

# Initialize Rich Console
console = Console()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold yellow]Loading NanoTextLM on {device}...[/bold yellow]")
    try:
        model, tokenizer, device, model_path, checkpoint_exists = load_inference_resources()
    except Exception as e:
        console.print(f"[bold red]Error loading resources: {e}[/bold red]")
        return

    if checkpoint_exists:
        console.print(f"[bold green]Loaded model from {model_path}[/bold green]")
    else:
        console.print("[bold red]Checkpoint not found. Using random weights.[/bold red]")
    
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
        if history_ids.size(1) > model.config.max_seq_len - 100:
             # Keep last N tokens
             history_ids = history_ids[:, -(model.config.max_seq_len - 100):]
        
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
