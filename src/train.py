import torch
from torch.utils.data import DataLoader
from config import ModelConfig, TrainingConfig
from model import TinyTechLM
from dataset import TextDataset
import os
import tqdm
import sys

def train():
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.bin")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")

    # Check for binary data
    if not os.path.exists(data_path):
        print("Binary dataset not found. Attempting to run tokenization...")
        tokenize_script = os.path.join(PROJECT_ROOT, "scripts", "tokenize_data.py")
        if os.path.exists(tokenize_script):
            import subprocess
            subprocess.run([sys.executable, tokenize_script], check=True)
        else:
            print(f"Error: Tokenize script not found at {tokenize_script}")
            return
    
    # Config
    m_conf = ModelConfig()
    t_conf = TrainingConfig()
    
    # Explicit Device Selection
    if torch.cuda.is_available():
        t_conf.device = "cuda"
        print("CUDA is available. Using GPU.")
        # Reset threads for GPU (usually handled by driver, but good practice to not over-subscribe CPU)
    else:
        t_conf.device = "cpu"
        print("CUDA not available. Using CPU.")
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"   Using {num_threads} threads for CPU computation.")
    
    # Data
    print(f"Loading dataset from {data_path}...")
    # Dataset now takes just data_path and block_size (no tokenizer needed inside)
    dataset = TextDataset(data_path, m_conf.max_seq_len)
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=t_conf.batch_size, shuffle=True, 
                        num_workers=0 if t_conf.device == "cuda" else min(4, os.cpu_count() or 1), 
                        pin_memory=True)

    # Model
    model = TinyTechLM(m_conf).to(t_conf.device)
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_conf.learning_rate, weight_decay=t_conf.weight_decay)

    # Train Loop
    model.train()
    step = 0
    print("Starting training...")
    
    for epoch in range(t_conf.max_epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(t_conf.device), y.to(t_conf.device)
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % t_conf.log_every == 0:
                pbar.set_postfix(loss=loss.item())
                
            if step % t_conf.save_every == 0:
                os.makedirs(t_conf.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(t_conf.output_dir, f"model_step_{step}.pt"))

    # Save final
    os.makedirs(t_conf.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(t_conf.output_dir, "final_model.pt"))
    print("Training complete.")

if __name__ == "__main__":
    train()
