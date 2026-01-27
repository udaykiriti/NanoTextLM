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
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_epochs: int = 3
    weight_decay: float = 0.1
    log_every: int = 10
    save_every: int = 500
    output_dir: str = "checkpoints"

    # Automatically select GPU if available, else CPU
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
