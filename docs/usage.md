# Usage Guide

## Makefile Shortcuts

The project includes a `Makefile` to simplify common commands:

- `make install`: Install Python dependencies.
- `make prepare`: Download and process the Shakespeare dataset.
- `make train`: Run full training on the default dataset.
- `make demo`: Run a fast training loop on Shakespeare.
- `make web`: Start the FastAPI web server.
- `make infer`: Start the CLI chat.
- `make test`: Run unit tests.
- `make evaluate`: Calculate Perplexity on the validation set.

## Training

### Resuming Training
To resume training from a checkpoint, use the `--resume` flag:

python src/train.py --resume checkpoints/final_model.pt

### Weights & Biases
To enable experiment tracking, set the configuration in `src/config.py`:

wandb_project: str = "nanotextlm"
wandb_run_name: str = "experiment-1"

## Inference

### Web Interface
The web interface (FastAPI) supports real-time streaming and parameter adjustment.
- **Temperature:** Controls randomness (higher = more creative).
- **Top-P:** Nucleus sampling threshold.
- **Max Tokens:** Length of generation.

### CLI Chat
The CLI maintains conversation history.
python src/inference.py

## Evaluation
Calculate the Perplexity (PPL) of your model to gauge its quality.

python scripts/evaluate.py --model checkpoints/final_model.pt --data data/processed/val.bin

## Deployment

### Hugging Face Hub
Upload your model to the Hugging Face Hub:

python scripts/push_to_hub.py --checkpoint checkpoints/final_model.pt --repo your-username/nanotextlm --token YOUR_TOKEN