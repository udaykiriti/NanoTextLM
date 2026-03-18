import os

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import NanoTextLM


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_checkpoint_path(project_root: str = PROJECT_ROOT) -> str:
    final_path = os.path.join(project_root, "checkpoints", "final_model.pt")
    if os.path.exists(final_path):
        return final_path
    return os.path.join(project_root, "checkpoints", "best_model.pt")


def load_checkpoint_state(model_path: str, device: str):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_inference_resources(compile_model: bool = True):
    device = get_device()
    model_path = resolve_checkpoint_path()
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run tokenizer training or restore tokenizer.json."
        )

    model = NanoTextLM(ModelConfig()).to(device)
    checkpoint_exists = os.path.exists(model_path)
    if checkpoint_exists:
        model.load_state_dict(load_checkpoint_state(model_path, device))
    model.eval()

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device, model_path, checkpoint_exists
