import os
import sys

import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import runtime


def test_load_inference_resources_requires_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "PROJECT_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
        runtime.load_inference_resources(compile_model=False)


def test_should_compile_model_only_on_cuda():
    assert runtime.should_compile_model("cpu", env={}) is False
    assert runtime.should_compile_model("cuda", env={}) is True
    assert runtime.should_compile_model("cuda", env={"NANOTEXTLM_COMPILE": "0"}) is False
