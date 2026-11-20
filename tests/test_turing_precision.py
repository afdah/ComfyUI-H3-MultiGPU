import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3vm.precision import (
    is_turing_or_older,
    cast_tensors_to_native_precision,
    sanitize_patcher_precision,
    sanitize_model_precision,
)


class DummyPatcher:
    def __init__(self, patches=None):
        self.patches = patches or {}
        self.force_cast_weights = False


def test_turing_detection():
    # GPU 0 on this system is RTX 2080 Ti (sm_75)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        assert is_turing_or_older("cuda:0") == (major < 8)
        assert is_turing_or_older("gpu:0") == (major < 8)


def test_tensor_casting():
    bf16_t = torch.randn(4, 4, dtype=torch.bfloat16)
    fp32_t = torch.randn(4, 4, dtype=torch.float32)
    fp16_t = torch.randn(4, 4, dtype=torch.float16)

    # Test single
    res, c = cast_tensors_to_native_precision(bf16_t)
    assert res.dtype == torch.float16 and c == 1

    res, c = cast_tensors_to_native_precision(fp32_t)
    assert res.dtype == torch.float16 and c == 1

    res, c = cast_tensors_to_native_precision(fp16_t)
    assert res.dtype == torch.float16 and c == 0

    # Test nested tuple (standard ComfyUI LoRA format)
    nested = (1.0, (torch.randn(2, 2, dtype=torch.bfloat16), torch.randn(2, 2, dtype=torch.float32)), 1.0, None, None)
    casted, c = cast_tensors_to_native_precision(nested)
    assert c == 2
    assert casted[1][0].dtype == torch.float16
    assert casted[1][1].dtype == torch.float16


def test_sanitize_patcher():
    patcher = DummyPatcher({
        "block.0.attn.qkv_proj.weight": [
            (1.0, (torch.randn(8, 8, dtype=torch.bfloat16), torch.randn(8, 8, dtype=torch.float32)), 1.0, None, None)
        ],
        "block.1.mlp.fc1.weight": [
            (0.8, torch.randn(16, 16, dtype=torch.bfloat16), 1.0, None, None)
        ]
    })

    converted = sanitize_model_precision(patcher, force=True)
    assert converted == 3
    assert patcher.force_cast_weights == torch.float16

    # Verify all tensors are now FP16
    qkv_patch = patcher.patches["block.0.attn.qkv_proj.weight"][0]
    assert qkv_patch[1][0].dtype == torch.float16
    assert qkv_patch[1][1].dtype == torch.float16

    mlp_patch = patcher.patches["block.1.mlp.fc1.weight"][0]
    assert mlp_patch[1].dtype == torch.float16


if __name__ == "__main__":
    test_turing_detection()
    test_tensor_casting()
    test_sanitize_patcher()
    print("Turing precision tests passed successfully!")
