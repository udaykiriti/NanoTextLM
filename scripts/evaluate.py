import torch
from torch.utils.data import DataLoader
from config import ModelConfig
from model import NanoTextLM
from dataset import TextDataset
import os
import argparse
import math
from tqdm import tqdm

def evaluate(model_path, data_path, batch_size=32, device="cuda"):
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {data_path}")
    
    # Load Model
    # Note: We need to know the config used for the model. 
    # Ideally, config is saved with checkpoint. For now, we assume default ModelConfig or NanoConfig.
    # In a real setup, save config.json alongside model.pt
    
    # Try loading as NanoConfig first if it fails size mismatch, or just standard.
    # We'll use standard ModelConfig for now.
    config = ModelConfig() 
    model = NanoTextLM(config)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Standard config failed: {e}")
        print("Trying NanoConfig...")
        from config import NanoConfig
        config = NanoConfig()
        model = NanoTextLM(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.to(device)
    model.eval()
    
    # Load Data
    if not os.path.exists(data_path):
        print("Data path not found.")
        return

    # Use 'val' split logic or just load the file
    # We'll use TextDataset directly
    dataset = TextDataset(data_path, config.max_seq_len, split='val', split_ratio=0.0) # 0.0 split ratio means all data is 'val' if split='val'? 
    # Wait, TextDataset split logic:
    # if split == 'train': data[:split_idx]
    # if split == 'val': data[split_idx:]
    # If we want to evaluate on a specific file (e.g. val.bin), we should probably treat it as a full file.
    # Let's adjust TextDataset logic or just mock it.
    # For now, let's assume standard val.bin is what we want.
    
    # Actually, let's just use split='val' on train.bin if val.bin doesn't exist, 
    # OR if data_path points to val.bin, use split='train' with ratio 1.0 to get full file.
    
    # Simplest: Just read the file and wrap it.
    # We'll rely on the existing class logic.
    dataset = TextDataset(data_path, config.max_seq_len, split='train', split_ratio=1.0) # Load full file
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    total_loss = 0
    total_steps = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            total_steps += 1
            
    if total_steps == 0:
        print("No data.")
        return

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    
    print("-" * 30)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity:      {perplexity:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="checkpoints/final_model.pt")
    parser.add_argument('--data', type=str, default="data/processed/val.bin")
    args = parser.parse_args()
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve paths
    model_path = args.model if os.path.isabs(args.model) else os.path.join(PROJECT_ROOT, args.model)
    data_path = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)
    
    # Fallback to train.bin if val.bin missing
    if not os.path.exists(data_path) and "val.bin" in data_path:
        print("val.bin not found, falling back to train.bin (evaluating on training data!)")
        data_path = data_path.replace("val.bin", "train.bin")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(model_path, data_path, device=device)
