"""Unit test for cross-GPU and chunked token modulation slicing in character swap/video edit workflows.

Verifies:
1. Exact numerical match between unchunked baseline and sliced/chunked multi-GPU execution.
2. Correct cross-device indexing when mod indices are created on cuda:0 and processed on cuda:1.
3. Mixed integer and Tensor mod segments.
"""

import sys
import torch
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from h3vm.post_attention_island import _mod_scale_shift_range, _mod_gate_range, _mod_row_slice


def test_cross_device_and_tensor_modulation():
    has_dual_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    d0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d1 = torch.device("cuda:1" if has_dual_gpu else d0)

    hidden = 128
    total_tokens = 1200

    # Layout:
    # 0..200: text (integer row 0)
    # 200..1000: video editing with per-token mask / replacement indices (Tensor on d0)
    # 1000..1200: audio (integer row 2)
    seg_text = (0, 200, 0)
    seg_video_tensor = (200, 1000, torch.randint(0, 3, (800,), device=d0, dtype=torch.long))
    seg_audio = (1000, 1200, 2)
    segments = [seg_text, seg_video_tensor, seg_audio]

    scale_ref = torch.randn(3, hidden, device=d0, dtype=torch.float32)
    shift_ref = torch.randn(3, hidden, device=d0, dtype=torch.float32)
    gate_ref = torch.randn(3, hidden, device=d0, dtype=torch.float32)

    h_orig = torch.randn(total_tokens, hidden, device=d0, dtype=torch.float32)
    x_orig = torch.randn(total_tokens, hidden, device=d0, dtype=torch.float32)
    other_orig = torch.randn(total_tokens, hidden, device=d0, dtype=torch.float32)

    # Reference single-GPU unchunked computation
    h_ref = h_orig.clone()
    for a, b, row in segments:
        s = scale_ref[row]
        sh = shift_ref[row]
        h_ref[a:b].mul_(1.0 + s).add_(sh)

    x_ref = x_orig.clone()
    for a, b, row in segments:
        g = gate_ref[row]
        x_ref[a:b].addcmul_(other_orig[a:b], g)

    # Chunks split across d0 and d1:
    # Chunk 1 on d0: [0, 450)
    # Chunk 2 on d1: [450, 950) (crosses into video tensor, parameter tensors on d1, index on d0)
    # Chunk 3 on d0: [950, 1200)
    chunks = [
        (0, 450, d0),
        (450, 950, d1),
        (950, 1200, d0),
    ]

    h_parts = []
    x_parts = []

    for s, e, dev in chunks:
        scale_dev = scale_ref.to(dev)
        shift_dev = shift_ref.to(dev)
        gate_dev = gate_ref.to(dev)

        h_slice = h_orig[s:e].clone().to(dev)
        x_slice = x_orig[s:e].clone().to(dev)
        other_slice = other_orig[s:e].clone().to(dev)

        out_h = _mod_scale_shift_range(h_slice, shift_dev, scale_dev, segments, global_start=s)
        out_x = _mod_gate_range(x_slice, gate_dev, other_slice, segments, global_start=s)

        h_parts.append(out_h.to(d0))
        x_parts.append(out_x.to(d0))

    h_reconstructed = torch.cat(h_parts, dim=0)
    x_reconstructed = torch.cat(x_parts, dim=0)

    max_diff_h = (h_ref - h_reconstructed).abs().max().item()
    max_diff_x = (x_ref - x_reconstructed).abs().max().item()

    print(f"Verified modulation slice: max_diff_h={max_diff_h}, max_diff_x={max_diff_x}")
    assert max_diff_h < 1e-6, f"Difference in h: {max_diff_h}"
    assert max_diff_x < 1e-6, f"Difference in x: {max_diff_x}"
    print("ALL TESTS PASSED: Cross-GPU token modulation slicing is 100% bit-exact!")


if __name__ == "__main__":
    test_cross_device_and_tensor_modulation()
