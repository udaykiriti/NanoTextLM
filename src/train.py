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
import argparse
import random
import numpy as np
import wandb

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
            # Handle AMP context
            ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if device.type == 'cuda' else torch.nullcontext()
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (small model, shakespeare config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Config
    if args.demo:
        print("Running in DEMO mode (NanoConfig)")
        from config import NanoConfig
        m_conf = NanoConfig()
        t_conf = TrainingConfig(
            batch_size=32,
            gradient_accumulation_steps=1,
            max_epochs=20, 
            eval_every=50,
            save_every=500,
            learning_rate=1e-3,
            warmup_iters=50,
            lr_decay_iters=2000,
            output_dir="checkpoints/demo"
        )
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.bin")
    else:
        m_conf = ModelConfig()
        t_conf = TrainingConfig()
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.bin")
    
    # Device setup
    if t_conf.device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for huge speedups on Ampere+ (A100, 3090, 4090, etc.)
        torch.set_float32_matmul_precision('high')
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
                            num_workers=4 if t_conf.device.type == 'cuda' else 0, 
                            pin_memory=True, 
                            persistent_workers=True if t_conf.device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=t_conf.batch_size, shuffle=False, 
                          num_workers=4 if t_conf.device.type == 'cuda' else 0, 
                          pin_memory=True,
                          persistent_workers=True if t_conf.device.type == 'cuda' else False)

    if len(train_dataset) == 0:
        print("Training dataset is empty.")
        return

    # Model
    model = NanoTextLM(m_conf).to(t_conf.device)
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimization: Compile
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer (Smart Weight Decay)
    optimizer = model.configure_optimizers(t_conf.weight_decay, t_conf.learning_rate, t_conf.device.type)
    scaler = torch.amp.GradScaler(enabled=(t_conf.device.type == 'cuda'))

    # Resume Logic
    iter_num = 0
    best_val_loss = 1e9
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=t_conf.device)
            
            # Handle both raw state_dict (legacy) and full checkpoint
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
                iter_num = checkpoint.get('iter_num', 0)
                best_val_loss = checkpoint.get('best_val_loss', 1e9)
                print(f"Resumed at step {iter_num} with val loss {best_val_loss:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded legacy model weights only. Resetting optimizer/steps.")
        else:
            print(f"Checkpoint {args.resume} not found. Starting from scratch.")

    # WandB Init
    if t_conf.wandb_project:
        print(f"Initializing WandB project: {t_conf.wandb_project}")
        wandb.init(
            project=t_conf.wandb_project, 
            name=t_conf.wandb_run_name or f"run_{int(time.time())}",
            config={**vars(m_conf), **vars(t_conf)},
            resume="allow",
            id=t_conf.wandb_run_name if args.resume else None
        )

    # Training Loop
    model.train()
    t0 = time.time()
    
    print(f"Starting training with Gradient Accumulation = {t_conf.gradient_accumulation_steps}")
    
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(t_conf.max_epochs):
        for micro_step, (x, y) in enumerate(train_loader):
            # Update Learning Rate
            lr = get_lr(iter_num, t_conf)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            x, y = x.to(t_conf.device), y.to(t_conf.device)
            
            # Forward pass with Mixed Precision
            ctx = torch.amp.autocast(device_type=t_conf.device.type, dtype=torch.float16) if t_conf.device.type == 'cuda' else torch.nullcontext()
            
            with ctx:
                logits, loss = model(x, y)
                # Scale loss for gradient accumulation
                loss = loss / t_conf.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Conditional Step (Gradient Accumulation)
            if (micro_step + 1) % t_conf.gradient_accumulation_steps == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_conf.grad_clip)
                
                # Step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Logging (only on step)
                if iter_num % t_conf.log_every == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    # Multiply loss back for display
                    loss_val = loss.item() * t_conf.gradient_accumulation_steps
                    print(f"Epoch {epoch+1} | Step {iter_num} | Loss: {loss_val:.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms")
                    
                    if wandb.run:
                        wandb.log({
                            "train/loss": loss_val,
                            "train/lr": lr,
                            "train/step_time_ms": dt*1000,
                            "epoch": epoch
                        }, step=iter_num)

                # Evaluation & Checkpointing
                if iter_num > 0 and iter_num % t_conf.eval_every == 0:
                    losses = estimate_loss(model, {'train': train_loader, 'val': val_loader}, t_conf.device)
                    print(f"Step {iter_num} Evaluation: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
                    
                    if wandb.run:
                        wandb.log({
                            "val/loss": losses['val'],
                            "train/eval_loss": losses['train']
                        }, step=iter_num)
                    
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        os.makedirs(t_conf.output_dir, exist_ok=True)
                        print(f"Saving best model (val_loss {best_val_loss:.4f})")
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': vars(m_conf)
                        }
                        torch.save(checkpoint, os.path.join(t_conf.output_dir, "best_model.pt"))
                
                iter_num += 1

    # Save final
    os.makedirs(t_conf.output_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': vars(m_conf)
    }
    torch.save(checkpoint, os.path.join(t_conf.output_dir, "final_model.pt"))
    print("Training complete.")

if __name__ == "__main__":
    seed_everything(1337)
    train()