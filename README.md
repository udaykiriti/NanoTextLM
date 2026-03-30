# NanoTextLM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/FastAPI-API-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

## Build Production-Ready Language Models Without the Bloat

**Fast training. Local iteration. Minimal overhead.**

NanoTextLM is a streamlined PyTorch framework for building, training, and deploying LLaMA-style transformers with modern techniques. No heavy frameworks. No unnecessary complexity. Just you, the model, and the data.

Built for researchers, students, and practitioners who want to understand and control every layer of their language model.

**Core Technologies**

```
RoPE Positional Embeddings  │  SwiGLU Activation  │  RMSNorm Normalization  │
Flash Attention Kernels     │  Mixed Precision    │  FastAPI Web Interface
```

## Why NanoTextLM?

| Use Case                         | Solution                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Learn Modern ML Architecture** | Study a production-grade LLaMA implementation in clean PyTorch—no framework abstractions hiding the details |
| **Run Local Experiments Fast**   | Train models on consumer hardware with minimal dependencies and pre-built demo datasets                     |
| **Deploy with Confidence**       | Full control over model loading, inference runtime, and serving infrastructure                              |

---

## What's Inside

**Model Architecture**

- RoPE rotary embeddings for better positional awareness
- SwiGLU activation gates for improved gradient flow
- RMSNorm for stable, efficient normalization
- Flash Attention for 2-3x faster inference

**Training Pipeline**

- Automatic mixed precision (AMP) to reduce memory footprint
- Gradient accumulation for effective larger batches
- Resume from checkpoints—no lost progress
- Optional WandB integration for experiment tracking

**Deployment Ready**

- CLI chat interface for quick testing
- FastAPI web app with streaming text generation
- Runtime configuration at inference time (device, tokenizer, model paths)
- Optional model compilation for maximum throughput

---

## Quick Start in 4 Commands

```bash
make install    # Install dependencies
make prepare    # Download and tokenize Tiny Shakespeare
make demo       # Train a small model
make web        # Launch the web interface
```

### 1. Install Dependencies

Requires Python 3.9+. Clone and set up the environment:

```bash
git clone https://github.com/udaykiriti/NanoTextLM.git
cd NanoTextLM
make install
```

For local testing with pytest:

```bash
pip install pytest
```

### 2. Prepare Your Dataset

Download and tokenize the Tiny Shakespeare dataset for a quick demonstration:

```bash
make prepare
```

### 3. Train on Demo Data

Verify everything works with a small training run:

```bash
make demo
```

### 4. Explore the Model

**Web Interface** — Launch the interactive web with real-time text generation:

```bash
make web
```

Access the UI at `http://localhost:5000`

**CLI Chat** — Quick command-line interaction:

```bash
make infer
```

**Run Tests** — Verify implementation correctness:

```bash
make test
```

---

## Configuration & Architecture

All model hyperparameters are in [src/config.py](src/config.py). Adjust these to match your hardware and research goals:

| Parameter     | Standard | Nano | What It Controls                            |
| ------------- | -------- | ---- | ------------------------------------------- |
| `d_model`     | 768      | 384  | Embedding and hidden dimensions             |
| `n_layers`    | 12       | 6    | Transformer depth                           |
| `n_heads`     | 12       | 6    | Attention heads (d_model must be divisible) |
| `max_seq_len` | 1024     | 256  | Maximum context window                      |

**Performance Profile:**

- **Nano Config:** Trains in minutes on consumer GPUs (3080, 4090, M1 Pro)
- **Standard Config:** Production-grade performance on datacenter hardware

---

## Repository Layout

```
NanoTextLM/
├── src/                  Core implementation
│   ├── model.py         # LLaMA-style transformer with RoPE, SwiGLU, Flash Attention
│   ├── train.py         # Training loop with AMP, gradient accumulation, WandB
│   ├── inference.py     # Inference runtime with tokenization
│   ├── app.py           # FastAPI web interface with streaming responses
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── config.py        # Centralized configuration (model, training, inference)
│   ├── runtime.py       # Shared loading utilities
│   └── templates/       # HTML frontend
│
├── scripts/              Utilities
│   ├── train_tokenizer.py      # Train BPE tokenizer
│   ├── prepare_shakespeare.py  # Download Tiny Shakespeare
│   ├── process_data.py         # Pre-process datasets
│   ├── tokenize_data.py        # Apply tokenizer to raw text
│   ├── evaluate.py             # Model evaluation
│   └── push_to_hub.py          # Hugging Face Model Hub upload
│
├── tests/                Unit and integration tests
│   ├── test_model.py
│   ├── test_train.py
│   ├── test_dataset.py
│   └── test_runtime.py
│
├── docs/                 Technical documentation
│   ├── architecture.md   # Deep dive: model design and components
│   ├── setup.md          # Detailed setup instructions
│   ├── usage.md          # End-to-end usage walkthrough
│   └── data_pipeline.md  # Dataset and tokenization pipeline
│
├── Makefile             # Common commands (install, train, serve)
├── Dockerfile           # Container build for reproducibility
├── requirements.txt     # Python dependencies
└── tokenizer.json       # Pre-trained tokenizer
```

---

## Learn More

Dive deeper into specific areas:

- **[Architecture Guide](docs/architecture.md)** — Understand the transformer design, custom kernels, and component interactions
- **[Setup Instructions](docs/setup.md)** — Detailed environment configuration and troubleshooting
- **[Usage & Training](docs/usage.md)** — End-to-end examples for training and inference
- **[Data Pipeline](docs/data_pipeline.md)** — Dataset preparation, tokenization, and benchmarking

---

## License

MIT License — See [LICENSE](LICENSE) for details. Use freely in research and production.
