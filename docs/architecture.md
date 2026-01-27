# Architecture

NanoTextLM is based on the GPT (Generative Pre-trained Transformer) architecture, specifically a decoder-only Transformer model. It is designed to be lightweight yet efficient, utilizing modern deep learning optimizations.

## Core Components

### 1. NanoTextLM (The Model)
The main class `NanoTextLM` orchestrates the model. It consists of:
- **Embedding Layer (`wte`):** Maps input token IDs to dense vectors.
- **Positional Embedding (`wpe`):** Adds learnable position information to the token embeddings.
- **Transformer Blocks (`h`):** A stack of `n_layers` identical layers that process the information.
- **Layer Normalization (`ln_f`):** Final normalization before the output head.
- **Language Modeling Head (`lm_head`):** Projects the final hidden state back to the vocabulary size to predict the next token.

### 2. Transformer Block
Each block consists of two sub-layers:
- **Causal Self-Attention:** Allows the model to attend to past tokens.
- **Feed-Forward Network (MLP):** Processes the information independently at each position.
Both sub-layers are wrapped with Layer Normalization and residual connections.

### 3. Causal Self-Attention
We utilize `torch.nn.functional.scaled_dot_product_attention`, which leverages Flash Attention (if available) for memory-efficient and fast computation. The attention mechanism is masked to prevent the model from looking into the future (causal masking).

## Optimizations

### Flash Attention
By using PyTorch 2.0's native scaled dot product attention, we reduce memory bandwidth usage (HBM) and increase computational speed compared to standard attention implementations.

### Automatic Mixed Precision (AMP)
The training loop utilizes `torch.amp` to perform operations in `float16` or `bfloat16` where appropriate, reducing memory footprint and speeding up tensor core operations on NVIDIA GPUs.

### Torch Compile
The model graph is optimized using `torch.compile` (JIT), which fuses kernels and reduces Python overhead during execution.

### Fused AdamW
When training on CUDA devices, we employ the fused implementation of the AdamW optimizer, which batches element-wise updates to the parameters, further reducing training time.
