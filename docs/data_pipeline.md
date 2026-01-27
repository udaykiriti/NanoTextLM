# Data Pipeline

NanoTextLM supports both simple datasets (Shakespeare) and large-scale sharded datasets (OpenWebText).

## Directory Structure
data/
├── raw/                  # Original files
├── processed/            # Single binary files (Shakespeare)
└── shards/               # Sharded binary files (Large Datasets)

## Sharded Tokenization (Scalable)

For datasets larger than RAM, use the sharded tokenizer script. This streams input text and writes output in chunks.

python scripts/tokenize_sharded.py

This will produce:
- `data/shards/train_000.bin`
- `data/shards/train_001.bin`
- ...

## Formats
- **.bin:** Raw uint16 arrays of token IDs.
- **tokenizer.json:** Hugging Face Tokenizer definition.