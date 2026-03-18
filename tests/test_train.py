import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import TrainingConfig
from train import get_lr


def test_get_lr_handles_zero_warmup():
    config = TrainingConfig(warmup_iters=0, lr_decay_iters=10, learning_rate=1e-3, min_lr=1e-4)

    lr = get_lr(0, config)

    assert lr == config.learning_rate


def test_get_lr_handles_non_increasing_decay_window():
    config = TrainingConfig(warmup_iters=10, lr_decay_iters=10, learning_rate=1e-3, min_lr=1e-4)

    lr = get_lr(10, config)

    assert lr == config.min_lr
