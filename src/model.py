import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    # xq.shape = [B, T, H, D] -> We need to handle this shape carefully
    # Usually RoPE is applied on the head dimension.
    # Our q/k shape in attention is [B, T, n_heads, d_head] after transpose?
    # No, usually [B, n_heads, T, d_head] for standard attention, but we transpose.
    # Let's check CausalSelfAttention logic below.
    
    # xq: [B, T, n_heads, d_head]
    
    # Reshape for rotation (pairs)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    
    # Reshape freqs for broadcast
    # freqs_cos: [T, d_head/2] -> [1, T, 1, d_head/2]
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    
    # Apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    # Stack back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to [B, T, n_heads, d_head]
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # Transpose for Attention: [B, n_heads, T, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout.p if self.training else 0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU variant
        # We keep hidden dim same as standard transformer (4*d) for simplicity, 
        # though LLaMA uses 2/3 * 4*d to save params.
        hidden_dim = 4 * config.d_model
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False) # Feat
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=False) # Output
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: (SiLU(Gate) * Feat) -> Output
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.c_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.ln_1(x), freqs_cos, freqs_sin)
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoTextLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            # wpe removed (Absolute Positional Embedding)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        
        # Precompute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.max_seq_len * 2) # *2 margin
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
             torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        use_fused = (
            device_type == 'cuda'
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.99), **extra_args)
        return optimizer

    def forward(self, idx, targets=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()
        if t > self.config.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.config.max_seq_len}")
        
        # Slice RoPE frequencies
        freqs_cos = self.freqs_cos[:t]
        freqs_sin = self.freqs_sin[:t]
        
        # Input Embedding (No Positional Embedding added here!)
        x = self.transformer.drop(self.transformer.wte(idx))
        
        # Gradient Checkpointing
        if self.config.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(block, x, freqs_cos, freqs_sin, use_reentrant=False)
        else:
            for block in self.transformer.h: 
                x = block(x, freqs_cos, freqs_sin)
                
        x = self.transformer.ln_f(x)
        
        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
        return logits, loss

    def _sample_top_p(self, logits, temperature=1.0, top_k=None, top_p=None):
        if temperature < 1e-5:
            return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _validate_generation_args(self, max_new_tokens, temperature, top_k, top_p):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if top_p is not None and not 0 < top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1]")

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self._validate_generation_args(max_new_tokens, temperature, top_k, top_p)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            idx_next = self._sample_top_p(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.inference_mode()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self._validate_generation_args(max_new_tokens, temperature, top_k, top_p)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            idx_next = self._sample_top_p(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
