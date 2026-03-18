import os
import sys

import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dataset import TextDataset


def test_dataset_handles_too_small_validation_split(tmp_path):
    data_path = tmp_path / "tiny.bin"
    np.arange(4, dtype=np.uint16).tofile(data_path)

    dataset = TextDataset(str(data_path), block_size=8, split="val")

    assert len(dataset) == 0


def test_dataset_rejects_unknown_split(tmp_path):
    data_path = tmp_path / "tiny.bin"
    np.arange(16, dtype=np.uint16).tofile(data_path)

    with pytest.raises(ValueError):
        TextDataset(str(data_path), block_size=4, split="test")
