from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_seq_len: int = 1024
    dropout: float = 0.1

@dataclass
class NanoConfig(ModelConfig):
    """Smaller config for testing or CPU training"""
    vocab_size: int = 5000  # Matches shakespeare tokenizer
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 256
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_epochs: int = 3
    weight_decay: float = 0.1
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 200  # Evaluate every N steps
    output_dir: str = "checkpoints"
    
    # Optimization
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 3e-5
    grad_clip: float = 1.0

    # Automatically select GPU if available, else CPU
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
