# NanoTextLM

NanoTextLM is a lightweight, optimized language model trained from scratch on the OpenWebText dataset. It features a GPT-style decoder-only Transformer architecture with modern optimizations like Flash Attention and streaming inference.

## Features

- **Architecture:** 12-layer, 12-head Transformer (~85M parameters).
- **Optimization:** Flash Attention (PyTorch 2.0), Automatic Mixed Precision (AMP), Cosine Learning Rate Schedule.
- **Inference:** Streaming generation via CLI (Rich UI) and Web API (SSE).
- **UI:** Modern Dark Mode Web Interface.
- **Deployment:** Docker support.

## Project Structure

- `src/`: Source code (Model, Training, Inference, App).
- `scripts/`: Data processing scripts.
- `data/`: Dataset storage.
- `tests/`: Unit tests.

## Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Download and process the OpenWebText subset:

```bash
# 1. Process Parquet to Text
python scripts/process_data.py

# 2. Train Tokenizer
python src/tokenizer.py

# 3. Tokenize Data (Text to Binary)
python scripts/tokenize_data.py
```

### 3. Training

Train the model with optimizations (CUDA recommended):

```bash
python src/train.py
```

### 4. Inference

**CLI (Streaming):**
```bash
python src/inference.py
```

**Web App (Streaming UI):**
```bash
python src/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

## Testing

Run unit tests to verify the architecture:

```bash
pytest tests/
```

## Docker

Build and run the container:

```bash
docker build -t nanotextlm .
docker run -p 5000:5000 nanotextlm
```
