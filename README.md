# NanoTextLM

![Python Tests](https://github.com/udaykiriti/NanoTextLM/actions/workflows/tests.yml/badge.svg)
![Docker Build](https://github.com/udaykiriti/NanoTextLM/actions/workflows/docker.yml/badge.svg)

NanoTextLM is a high-performance, lightweight language model trained from scratch. It implements a modern LLaMA-style architecture (RoPE, SwiGLU, RMSNorm) and features a complete training, evaluation, and deployment pipeline.

## Features

- **Architecture:** 
  - Rotary Positional Embeddings (RoPE)
  - SwiGLU Activation
  - RMSNorm (Root Mean Square Normalization)
  - Flash Attention (PyTorch 2.0)
  - No Biases in Linear Layers

- **Training:**
  - Automatic Mixed Precision (AMP)
  - Gradient Accumulation
  - Gradient Checkpointing (VRAM optimization)
  - Fused AdamW Optimizer
  - Weights & Biases (WandB) Logging
  - Resume Training Support

- **Inference:**
  - High-performance FastAPI Backend
  - Real-time Streaming (SSE)
  - Top-P (Nucleus) & Temperature Sampling
  - Context-Aware CLI Chat
  - Modern Web UI with Advanced Controls

- **Operations:**
  - Docker Containerization
  - GitHub Actions CI/CD
  - Makefile for Common Tasks
  - Hugging Face Hub Integration

## Quick Start

### 1. Install
pip install -r requirements.txt

### 2. Prepare Data (Shakespeare Demo)
make prepare

### 3. Train (Demo)
make demo

### 4. Chat (Web UI)
make web

Access the interface at http://localhost:5000.

## Documentation

- [Architecture Details](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [Usage Guide](docs/usage.md)
- [Data Pipeline](docs/data_pipeline.md)