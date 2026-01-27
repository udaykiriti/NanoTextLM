import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More numerically stable and slightly faster than LayerNorm.
    Used in LLaMA, PaLM, etc.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # bias=False is a modern best practice (LLaMA, PaLM) to save parameters/compute
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
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
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replace LayerNorm with RMSNorm
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoTextLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Biases are generally disabled, but if any remain (e.g. in some layers if configured), zero them
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
             # RMSNorm has no bias, just weight
             torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Separate parameters into those that experience weight decay (matmuls, embeddings)
        and those that do not (biases, layernorms).
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Filter out parameters that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer
        use_fused = (device_type == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"Using AdamW optimizer (fused={use_fused})")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.99), **extra_args)
        return optimizer

    def forward(self, idx, targets=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        
        # Gradient Checkpointing (saves VRAM by recomputing activations during backward pass)
        if self.config.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h: 
                x = block(x)
                
        x = self.transformer.ln_f(x)
        
        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
        return logits, loss

    def _sample_top_p(self, logits, temperature=1.0, top_k=None, top_p=None):
        # Greedy decoding for temp -> 0
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            idx_next = self._sample_top_p(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            idx_next = self._sample_top_p(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next