import sys
import os
import torch
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import NanoTextLM
from config import ModelConfig

@pytest.fixture
def model_config():
    return ModelConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=32
    )

def test_model_initialization(model_config):
    model = NanoTextLM(model_config)
    assert model is not None
    # Check parameter count > 0
    assert sum(p.numel() for p in model.parameters()) > 0
    # Check if RMSNorm is used (just checking strict type might fail if we imported class locally, but logic holds)
    from model import RMSNorm
    assert isinstance(model.transformer.ln_f, RMSNorm)

def test_forward_pass(model_config):
    model = NanoTextLM(model_config)
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx)
    
    # Check shape: [batch, 1, vocab_size] (inference mode)
    assert logits.shape == (batch_size, 1, model_config.vocab_size)
    assert loss is None

def test_forward_pass_with_targets(model_config):
    model = NanoTextLM(model_config)
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx, targets=targets)
    
    assert logits.shape == (batch_size, seq_len, model_config.vocab_size)
    assert loss is not None
    assert isinstance(loss.item(), float)

def test_generate_stream(model_config):
    model = NanoTextLM(model_config)
    idx = torch.randint(0, model_config.vocab_size, (1, 5))
    generated = list(model.generate_stream(idx, max_new_tokens=3))
    assert len(generated) == 3
    assert all(token.shape == (1, 1) for token in generated)

def test_generate_zero_tokens_returns_input(model_config):
    model = NanoTextLM(model_config)
    idx = torch.randint(0, model_config.vocab_size, (1, 5))

    out = model.generate(idx, max_new_tokens=0)

    assert torch.equal(out, idx)

def test_generate_rejects_invalid_sampling_args(model_config):
    model = NanoTextLM(model_config)
    idx = torch.randint(0, model_config.vocab_size, (1, 5))

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, temperature=-0.1)

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, top_k=0)

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, top_p=1.5)

def test_forward_rejects_too_long_sequence(model_config):
    model = NanoTextLM(model_config)
    idx = torch.randint(0, model_config.vocab_size, (1, model_config.max_seq_len + 1))

    with pytest.raises(ValueError):
        model(idx)
