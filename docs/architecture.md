# Architecture

NanoTextLM implements a decoder-only Transformer architecture that closely mirrors state-of-the-art models like LLaMA and PaLM.

## Key Components

### 1. Rotary Positional Embeddings (RoPE)
Unlike standard GPT models that use absolute learned positional embeddings, NanoTextLM uses RoPE. This injects positional information by rotating the Query and Key vectors in the attention mechanism, allowing for better generalization to sequence lengths longer than those seen during training.

### 2. RMSNorm
We utilize Root Mean Square Normalization (RMSNorm) instead of standard LayerNorm. RMSNorm is computationally more efficient and provides better numerical stability during training.

### 3. SwiGLU Activation
The Feed-Forward Network (MLP) uses the SwiGLU activation function instead of GELU. SwiGLU (Swish Gated Linear Unit) involves three linear projections and has been shown to improve performance in models like LLaMA and PaLM.

### 4. Bias-Free Linear Layers
To optimize memory usage and parameter count, we have disabled biases in all linear layers (QKV projections and MLP layers), relying on RMSNorm for centering.

### 5. Flash Attention
The model leverages PyTorch 2.0's `scaled_dot_product_attention`, which uses IO-aware implementations (Flash Attention) to significantly reduce memory access and increase speed.

## Optimizations

### Training Optimizations
- **Automatic Mixed Precision (AMP):** Runs compute-heavy operations in float16/bfloat16.
- **Gradient Checkpointing:** Trades compute for memory by discarding intermediate activations during the forward pass and recomputing them during the backward pass.
- **Torch Compile:** Uses JIT compilation to fuse kernels.
- **Smart Weight Decay:** Applies weight decay only to 2D parameters (weights), excluding norms and embeddings.