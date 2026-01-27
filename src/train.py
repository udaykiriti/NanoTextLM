import torch
from torch.utils.data import DataLoader
from config import ModelConfig, TrainingConfig
from model import NanoTextLM
from dataset import TextDataset
import os
import tqdm
import sys
import math
import time

def get_lr(it, t_conf):
    # 1) linear warmup
    if it < t_conf.warmup_iters:
        return t_conf.learning_rate * it / t_conf.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > t_conf.lr_decay_iters:
        return t_conf.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - t_conf.warmup_iters) / (t_conf.lr_decay_iters - t_conf.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return t_conf.min_lr + coeff * (t_conf.learning_rate - t_conf.min_lr)

@torch.no_grad()
def estimate_loss(model, dataloaders, device, eval_iters=50):
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        losses = torch.zeros(eval_iters)
        iter_loader = iter(loader)
        for k in range(eval_iters):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                x, y = next(iter_loader)
                
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=(device.type == 'cuda')):
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (small model, shakespeare config)')
    args = parser.parse_args()

    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.bin")
    
    # Config
    if args.demo:
        print("Running in DEMO mode (NanoConfig)")
        from config import NanoConfig
        m_conf = NanoConfig()
        t_conf = TrainingConfig(
            batch_size=32, # Larger batch size for smaller model
            max_epochs=20, # More epochs for small dataset
            eval_every=50,
            save_every=500,
            learning_rate=1e-3,
            warmup_iters=50,
            lr_decay_iters=2000
        )
    else:
        m_conf = ModelConfig()
        t_conf = TrainingConfig()
    
    # Device setup
    if t_conf.device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    else:
        print(f"Using Device: {t_conf.device}")

    # Data
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run data processing scripts first.")
        return

    print("Preparing datasets...")
    train_dataset = TextDataset(data_path, m_conf.max_seq_len, split='train')
    val_dataset = TextDataset(data_path, m_conf.max_seq_len, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=t_conf.batch_size, shuffle=True, 
                            num_workers=4 if t_conf.device.type == 'cuda' else 0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=t_conf.batch_size, shuffle=False, 
                          num_workers=4 if t_conf.device.type == 'cuda' else 0, pin_memory=True)

    if len(train_dataset) == 0:
        print("Training dataset is empty.")
        return

    # Model
    model = NanoTextLM(m_conf).to(t_conf.device)
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_conf.learning_rate, weight_decay=t_conf.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(t_conf.device.type == 'cuda'))

    # Training Loop
    model.train()
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()
    
    print("Starting training with validation and LR scheduling...")
    
    # We'll train based on total iterations rather than strict epochs for simplicity in this loop
    # or keep epochs but track global steps.
    
    for epoch in range(t_conf.max_epochs):
        for x, y in train_loader:
            # Update Learning Rate
            lr = get_lr(iter_num, t_conf)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            x, y = x.to(t_conf.device), y.to(t_conf.device)
            
            # Forward pass with Mixed Precision
            # 'cuda' works for NVIDIA GPUs. For others, autocast might need adjustment or be disabled.
            # We used t_conf.device.type logic in estimate_loss, reusing here.
            ctx = torch.amp.autocast(device_type=t_conf.device.type, dtype=torch.float16) if t_conf.device.type == 'cuda' else torch.nullcontext()
            
            with ctx:
                logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_conf.grad_clip)
            
            # Step
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            if iter_num % t_conf.log_every == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"Epoch {epoch+1} | Step {iter_num} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms")

            # Evaluation & Checkpointing
            if iter_num > 0 and iter_num % t_conf.eval_every == 0:
                losses = estimate_loss(model, {'train': train_loader, 'val': val_loader}, t_conf.device)
                print(f"Step {iter_num} Evaluation: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
                
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    os.makedirs(t_conf.output_dir, exist_ok=True)
                    print(f"Saving best model (val_loss {best_val_loss:.4f})")
                    torch.save(model.state_dict(), os.path.join(t_conf.output_dir, "best_model.pt"))
            
            iter_num += 1

    # Save final
    os.makedirs(t_conf.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(t_conf.output_dir, "final_model.pt"))
    print("Training complete.")

if __name__ == "__main__":
    train()