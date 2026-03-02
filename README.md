# NanoTextLM

[![Python Tests](https://github.com/udaykiriti/NanoTextLM/actions/workflows/tests.yml/badge.svg)](https://github.com/udaykiriti/NanoTextLM/actions)
[![Docker Build](https://github.com/udaykiriti/NanoTextLM/actions/workflows/docker.yml/badge.svg)](https://github.com/udaykiriti/NanoTextLM/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NanoTextLM** is a high-performance, lightweight language model built from scratch using PyTorch. Designed for efficiency and modularity, it implements a modern LLaMA-style architecture with features like RoPE, SwiGLU, and Flash Attention.

---

## Key Features

### Architecture
- **State-of-the-Art:** RoPE (Rotary Positional Embeddings), SwiGLU activation, and RMSNorm.
- **Speed:** Optimized with Flash Attention (PyTorch 2.0+) and fused AdamW optimizer.
- **Efficiency:** Gradient checkpointing and Automatic Mixed Precision (AMP) for low VRAM usage.

### Training & Evaluation
- **Scalable:** Built-in gradient accumulation for large batch training on small GPUs.
- **Observability:** Seamless Weights & Biases (WandB) integration for real-time logging.
- **Robust:** Support for resuming training from checkpoints and easy Hugging Face Hub integration.

### Inference & Deployment
- **Interactive:** Context-aware CLI chat and a modern FastAPI-powered Web UI.
- **Real-time:** Streaming support (SSE) for low-latency text generation.
- **Deployable:** Ready-to-use Docker containerization and GitHub Actions CI/CD.

---

## Quick Start

### 1. Installation
Ensure you have Python 3.9+ installed. Clone the repo and install dependencies:
```bash
git clone https://github.com/udaykiriti/NanoTextLM.git
cd NanoTextLM
make install
```

### 2. Prepare Data
Download and tokenize the Tiny Shakespeare dataset for a quick demo:
```bash
make prepare
```

### 3. Run Training (Demo)
Start a small-scale training run to verify your setup:
```bash
make demo
```

### 4. Chat with the Model
Launch the interactive Web UI and start generating text:
```bash
make web
```
> [!NOTE]
> Access the web interface at `http://localhost:5000`.

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
│   └── app.py         # FastAPI Web Backend
├── scripts/           # Data processing and utility scripts
├── docs/              # Detailed technical documentation
├── tests/             # Unit and integration tests
├── Makefile           # Convenient task automation
└── Dockerfile         # Containerized deployment
```

---

## Documentation

For a deeper dive into the architecture and setup, please refer to the following guides:

*   **Architecture Guide:** [Detailed technical overview of the model](docs/architecture.md)
*   **Setup Guide:** [Environment configuration and installation steps](docs/setup.md)
*   **Usage Guide:** [Comprehensive instructions for training and inference](docs/usage.md)
*   **Data Pipeline:** [Details on dataset preparation and tokenization](docs/data_pipeline.md)

---

## Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, adding a new feature, or updating documentation, follow these steps to contribute:

1.  **Fork** the repository to your own account.
2.  **Create** a new branch for your feature (`git checkout -b feature/amazing-feature`).
3.  **Commit** your changes with clear messages (`git commit -m 'Add amazing feature'`).
4.  **Push** your branch to your fork (`git push origin feature/amazing-feature`).
5.  **Submit** a Pull Request against our main branch.

---

## License

This project is licensed under the **MIT License**. For the full legal text, please refer to the [LICENSE](LICENSE) file.