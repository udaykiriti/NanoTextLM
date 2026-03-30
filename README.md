# NanoTextLM

<p align="center">
  <a href="https://github.com/udaykiriti/NanoTextLM/actions">
    <img src="https://github.com/udaykiriti/NanoTextLM/actions/workflows/tests.yml/badge.svg" alt="Python Tests">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-Web_UI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT">
  </a>
</p>

<p align="center">
  <strong>Compact transformer training. Fast local iteration. Minimal moving parts.</strong>
</p>

<p align="center">
  NanoTextLM is a lightweight PyTorch language-model project for learning, training, and serving a compact LLaMA-style transformer without a heavy framework stack.
</p>

<p align="center">
  <code>RoPE</code>
  <code>SwiGLU</code>
  <code>RMSNorm</code>
  <code>Flash Attention</code>
  <code>AMP</code>
  <code>FastAPI</code>
  <code>CLI Chat</code>
</p>

## Overview

| Built For | What You Get |
|-----------|---------------|
| Learning modern LM internals | A readable LLaMA-style implementation in plain PyTorch |
| Quick local experiments | Small configs, demo training, CLI chat, and a FastAPI app |
| Simple iteration | Minimal runtime helpers, clear scripts, and direct checkpoints |

## Snapshot

- **[Model]**: RoPE, RMSNorm, SwiGLU, Flash Attention.
- **[Training]**: AMP, gradient accumulation, optional WandB, checkpoint resume.
- **[Serving]**: CLI chat, web streaming, runtime overrides for device/checkpoint/tokenizer.

## Stack

| Layer | Tools |
|-------|-------|
| Model | PyTorch, Flash Attention, RMSNorm, RoPE, SwiGLU |
| Data | NumPy memmap datasets, tokenizer pipeline, Tiny Shakespeare demo data |
| Serving | FastAPI, Uvicorn, streaming text responses |
| Developer Workflow | Makefile commands, pytest, GitHub Actions, Docker build |

---

## Features

### Model
- **RoPE**: Rotary positional embeddings.
- **SwiGLU + RMSNorm**: Modern transformer building blocks.
- **Flash Attention**: Fast attention path on supported PyTorch builds.

### Training
- **AMP**: Mixed precision for lower memory use.
- **Gradient accumulation**: More effective batch size on smaller hardware.
- **Checkpoint resume**: Continue interrupted runs.

### Inference
- **CLI chat**: Quick local interaction.
- **FastAPI app**: Browser-based text generation.
- **Runtime overrides**: Device, tokenizer, checkpoint, and compile controls.

---

## Quick Start

```bash
make install
make prepare
make demo
make web
```

###  Install
Ensure you have Python 3.9+ installed. Clone the repo and install dependencies:
```bash
git clone https://github.com/udaykiriti/NanoTextLM.git
cd NanoTextLM
make install
```

If you want to run tests locally, make sure `pytest` is installed in your environment:
```bash
pip install pytest
```

###  Prepare Data
Download and tokenize the Tiny Shakespeare dataset for a quick demo:
```bash
make prepare
```

### Train A Demo Model
Start a small-scale training run to verify your setup:
```bash
make demo
```

### Run The App
Launch the interactive Web UI and start generating text:
```bash
make web
```
> [!NOTE]
> Access the web interface at `http://localhost:5000`.

Run the CLI chat interface instead:
```bash
make infer
```

Run the test suite:
```bash
make test
```

---

## Model Configuration

NanoTextLM is highly configurable. Key parameters in `src/config.py` include:

| Parameter | Default (Standard) | Default (Nano) | Description |
|-----------|--------------------|----------------|-------------|
| `d_model` | 768                | 384            | Embedding dimension |
| `n_layers`| 12                 | 6              | Number of transformer layers |
| `n_heads` | 12                 | 6              | Number of attention heads |
| `max_seq_len` | 1024           | 256            | Maximum context window |

---

## Project Structure

```text
NanoTextLM/
├── src/               # Core model and training logic
│   ├── model.py       # LLaMA-style Transformer implementation
│   ├── train.py       # Training loop with AMP and WandB
│   ├── app.py         # FastAPI Web backend
│   └── runtime.py     # Shared inference loading utilities
├── scripts/           # Data processing and utility scripts
├── docs/              # Detailed technical documentation
├── tests/             # Unit and integration tests
├── Makefile           # Convenient task automation
└── Dockerfile         # Optional local container build
```

---

## Docs

For a deeper dive into the architecture and setup, please refer to the following guides:

- **Architecture Guide:** [Detailed technical overview of the model](docs/architecture.md)
- **Setup Guide:** [Environment configuration and installation steps](docs/setup.md)
- **Usage Guide:** [Comprehensive instructions for training and inference](docs/usage.md)
- **Data Pipeline:** [Details on dataset preparation and tokenization](docs/data_pipeline.md)

---

## License

This project is licensed under the **MIT License**. For the full legal text, please refer to the [LICENSE](LICENSE) file.
