# Usage Guide

## Data Preparation

Before training, you must prepare the dataset. NanoTextLM expects binary files (`.bin`) containing raw uint16 token IDs.

### Quick Start (Shakespeare Dataset)
To download and tokenize the Tiny Shakespeare dataset for a quick demo:

python scripts/prepare_shakespeare.py

### Custom Data (OpenWebText)
1. Download and extract raw text from Parquet files:
   python scripts/process_data.py

2. Train the BPE tokenizer on the raw text:
   python scripts/train_tokenizer.py

3. Tokenize the text into binary format:
   python scripts/tokenize_data.py

## Training

### Demo Mode
Train a small model on the Shakespeare dataset (CPU-friendly):

python src/train.py --demo

### Full Training
Train the full model on OpenWebText (GPU recommended):

python src/train.py

**Key Configuration:**
- Logs and checkpoints are saved to the `checkpoints/` directory.
- Training parameters (batch size, learning rate) can be modified in `src/config.py`.

## Inference

### Command Line Interface (CLI)
Run the interactive CLI to chat with the model. It supports streaming output.

python src/inference.py

### Web Interface
Launch the Flask web server to interact via a browser.

python src/app.py

Access the interface at http://localhost:5000.
