import torch
import os
import argparse
import json
from huggingface_hub import HfApi, create_repo

def push_to_hub(checkpoint_path, repo_id, token=None):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict and config
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        config_dict = checkpoint['config']
    else:
        state_dict = checkpoint
        # Try to infer config or use default
        from config import ModelConfig
        config_dict = vars(ModelConfig())

    # Create a temporary directory for artifacts
    output_dir = "hf_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Model Weights
    print("Saving model weights...")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 2. Save Config
    # We map our config to a format HF might respect (Custom)
    print("Saving config...")
    # Add architectre
    config_dict["architectures"] = ["NanoTextLM"]
    config_dict["model_type"] = "nanotextlm"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 3. Save Tokenizer
    tokenizer_src = "tokenizer.json"
    if os.path.exists(tokenizer_src):
        print("Copying tokenizer...")
        import shutil
        shutil.copy(tokenizer_src, os.path.join(output_dir, "tokenizer.json"))
    
    # 4. Create Model Card
    print("Creating Model Card...")
    readme_content = f"""
---
tags:
- pytorch
- causal-lm
- nanotextlm
---

# NanoTextLM

This is a **NanoTextLM** model trained from scratch.

## Architecture
- **Type:** Decoder-only Transformer
- **Positional Embeddings:** Rotary (RoPE)
- **Normalization:** RMSNorm
- **Activation:** SwiGLU
- **Attention:** Flash Attention

## Usage

```python
# Coming soon: AutoModel integration code
```
    """
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content.strip())

    # Upload
    print(f"Uploading to Hugging Face Hub: {repo_id}...")
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Ensure you are logged in via `huggingface-cli login` or provided a valid token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--repo', type=str, required=True, help='HF Repo ID (e.g. username/nanotextlm)')
    parser.add_argument('--token', type=str, default=None, help='HF API Token')
    
    args = parser.parse_args()
    
    push_to_hub(args.checkpoint, args.repo, args.token)
