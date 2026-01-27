# NanoTextLM

NanoTextLM is a lightweight language model trained from scratch on the OpenWebText dataset. This project demonstrates the implementation of a decoder-only Transformer architecture, tokenizer training, and the complete training pipeline using PyTorch.

## Dataset

This project relies on a subset of the OpenWebText dataset.
Source: OpenWebText train-00074-of-00080.parquet

The dataset file should be placed at: data/raw/openwebtext/train-00074-of-00080.parquet

## Project Structure

- src/: Source code for the model, training, and inference.
- scripts/: Data processing and utility scripts.
- data/: Dataset storage (raw and processed).

## Usage

1. Install dependencies:
   pip install -r requirements.txt

2. Process data:
   python scripts/process_data.py

3. Train tokenizer:
   python src/tokenizer.py

4. Train model:
   python src/train.py

5. Run inference:
   python src/inference.py