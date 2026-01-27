# Data Pipeline Guide

This document details the data organization and the commands used to populate it. NanoTextLM uses a binary format for efficient training.

## Directory Structure

The project expects the following directory structure for data management:

```
data/
├── raw/                  # Original downloaded files
│   ├── shakespeare.txt   # (Demo) Raw text file
│   └── openwebtext/      # (Full) Parquet files
│       └── train-*.parquet
│
└── processed/            # Files ready for training
    ├── train.bin         # Binary token IDs (uint16)
    ├── val.bin           # Validation split (optional)
    └── train.txt         # Intermediate text file (for tokenizer training)
```

## Workflows

### 1. Quick Start (Tiny Shakespeare)

This workflow downloads a small dataset, trains a tokenizer on it, and converts it to binary format in one go.

**Command:**
```bash
python scripts/prepare_shakespeare.py
```

**Result:**
- Creates `data/raw/shakespeare.txt`
- Creates `data/processed/train.bin` and `data/processed/val.bin`
- Saves `tokenizer.json` to the project root.

### 2. Full Training (OpenWebText / Custom)

For large datasets, the process is split into three steps to handle memory efficiency.

#### Step A: Raw Data -> Text
Extract text from raw formats (like Parquet) into a clean text file.

**Command:**
```bash
python scripts/process_data.py
```
*Input:* `data/raw/openwebtext/*.parquet`
*Output:* `data/processed/train.txt`

#### Step B: Train Tokenizer
Train a Byte-Pair Encoding (BPE) tokenizer on the text data.

**Command:**
```bash
python scripts/train_tokenizer.py
```
*Input:* `data/processed/train.txt`
*Output:* `tokenizer.json`

#### Step C: Text -> Binary
Encode the text file into a memory-mapped binary file using the trained tokenizer. This is the file used by the training loop.

**Command:**
```bash
python scripts/tokenize_data.py
```
*Input:* `data/processed/train.txt`
*Output:* `data/processed/train.bin`

## File Formats

- **.bin files:** These are raw `uint16` numpy arrays saved to disk. They contain the sequence of token IDs. We use `uint16` because our vocabulary size (50,257) fits within 65,535.
- **tokenizer.json:** The Hugging Face Tokenizers format containing the vocabulary and merge rules.
