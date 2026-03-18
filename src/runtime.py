import os

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import NanoTextLM


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_env_override(name: str, env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_device(env: dict | None = None) -> str:
    override_device = _get_env_override("NANOTEXTLM_DEVICE", env)
    if override_device:
        device = override_device.lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("NANOTEXTLM_DEVICE must be 'cpu' or 'cuda'")
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def should_compile_model(device: str, env: dict | None = None) -> bool:
    compile_flag = _get_env_override("NANOTEXTLM_COMPILE", env)
    if compile_flag is None:
        compile_flag = "1"
    compile_flag = compile_flag.lower()
    return device == "cuda" and compile_flag not in {"0", "false", "no"}


def resolve_checkpoint_path(project_root: str | None = None) -> str:
    project_root = PROJECT_ROOT if project_root is None else project_root
    override_path = _get_env_override("NANOTEXTLM_CHECKPOINT")
    if override_path:
        return override_path
    final_path = os.path.join(project_root, "checkpoints", "final_model.pt")
    if os.path.exists(final_path):
        return final_path
    return os.path.join(project_root, "checkpoints", "best_model.pt")


def resolve_tokenizer_path(project_root: str | None = None) -> str:
    project_root = PROJECT_ROOT if project_root is None else project_root
    override_path = _get_env_override("NANOTEXTLM_TOKENIZER")
    if override_path:
        return override_path
    return os.path.join(project_root, "tokenizer.json")


def load_checkpoint_state(model_path: str, device: str):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def load_inference_resources(compile_model: bool = True):
    device = get_device()
    model_path = resolve_checkpoint_path()
    tokenizer_path = resolve_tokenizer_path()
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run tokenizer training or restore tokenizer.json."
        )

    model = NanoTextLM(ModelConfig()).to(device)
    checkpoint_exists = os.path.exists(model_path)
    if checkpoint_exists:
        model.load_state_dict(load_checkpoint_state(model_path, device))
    model.eval()

    if compile_model and should_compile_model(device) and hasattr(torch, "compile"):
        model = torch.compile(model)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device, model_path, checkpoint_exists
