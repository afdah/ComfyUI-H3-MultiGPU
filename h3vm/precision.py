"""H3VM Turing sm_75 precision auto-caster and LoRA sanitizer.

RTX 2080 Ti and Turing GPUs (compute capability 7.5) feature full-speed FP16 and INT8 Tensor Cores,
but lack hardware BF16 and FP8 execution units. Running BF16 or FP32 weights on Turing forces
software emulation or doubles memory consumption.

This module inspects incoming ComfyUI models, LoRA patches, and weights, dynamically auto-casting
them to native FP16/INT8 for maximum Tensor Core throughput and minimum VRAM usage.
"""
from __future__ import annotations

import logging
import torch

LOG = logging.getLogger("h3vm.precision")


def is_turing_or_older(device: str | torch.device | None = None) -> bool:
    """Return True if device lacks native hardware BF16 Tensor Cores (sm_75 or older)."""
    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            dev = torch.device("cuda:0")
        elif isinstance(device, str):
            label = device.lower().strip()
            if label.startswith("gpu:"):
                dev = torch.device("cuda", int(label.split(":", 1)[1]))
            elif label.startswith("cuda:"):
                dev = torch.device(label)
            else:
                dev = torch.device("cuda:0")
        else:
            dev = device
        if dev.type != "cuda":
            return False
        idx = dev.index if dev.index is not None else 0
        major, _ = torch.cuda.get_device_capability(idx)
        # Turing is sm_75 (major=7, minor=5); Volta is sm_70; Pascal is sm_60.
        # Ampere (sm_80, RTX 30-series) and newer have native hardware BF16 Tensor Cores.
        return major < 8
    except Exception:
        return False


def cast_tensors_to_native_precision(item, target_dtype: torch.dtype = torch.float16) -> tuple[any, int]:
    """Recursively cast BF16 / FP32 tensors to target_dtype (FP16). Returns (casted_item, count)."""
    if torch.is_tensor(item):
        if item.dtype in (torch.bfloat16, torch.float32):
            return item.to(dtype=target_dtype), 1
        return item, 0
    if isinstance(item, tuple):
        res = [cast_tensors_to_native_precision(x, target_dtype) for x in item]
        return tuple(r[0] for r in res), sum(r[1] for r in res)
    if isinstance(item, list):
        res = [cast_tensors_to_native_precision(x, target_dtype) for x in item]
        return [r[0] for r in res], sum(r[1] for r in res)
    if isinstance(item, dict):
        new_dict = {}
        total = 0
        for k, v in item.items():
            val, c = cast_tensors_to_native_precision(v, target_dtype)
            new_dict[k] = val
            total += c
        return new_dict, total
    return item, 0


def _infer_target_dtype(patcher) -> torch.dtype:
    """Return the patcher model's preferred compute dtype, defaulting to FP16."""
    model = getattr(patcher, "model", None)
    if model is not None:
        manual_cast = getattr(model, "manual_cast_dtype", None)
        if isinstance(manual_cast, torch.dtype):
            return manual_cast
        model_dtype_fn = getattr(model, "model_dtype", None)
        if callable(model_dtype_fn):
            try:
                inferred = model_dtype_fn()
                if isinstance(inferred, torch.dtype) and inferred in (torch.float16, torch.bfloat16, torch.float32):
                    return inferred
            except Exception:
                pass
    return torch.float16


def sanitize_patcher_precision(patcher, device: str | torch.device | None = None, force: bool = False) -> int:
    """Inspect and cast attached LoRA patches to native FP16 on Turing sm_75 when appropriate.

    Respects models configured to compute in BF16/FP32 rather than blindly
    converting their patches to FP16. Returns the number of patched tensors
    converted.
    """
    if patcher is None:
        return 0
    if not force and not is_turing_or_older(device):
        return 0

    target_dtype = _infer_target_dtype(patcher)
    if target_dtype != torch.float16:
        return 0

    patches = getattr(patcher, "patches", {}) or {}
    if not patches:
        return 0

    total_converted = 0
    new_patches = {}
    for key, patch_list in patches.items():
        new_list = []
        for entry in patch_list:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                # Standard ComfyUI patch entry: (strength_patch, weight_data, strength_model, offset, function)
                v = entry[1]
                casted_v, count = cast_tensors_to_native_precision(v, target_dtype)
                if count > 0:
                    total_converted += count
                    entry = list(entry)
                    entry[1] = casted_v
                    entry = tuple(entry)
            new_list.append(entry)
        new_patches[key] = new_list

    patcher.patches = new_patches
    if total_converted > 0:
        print(
            f"[H3VM PRECISION] Turing sm_75: Auto-cast {total_converted} LoRA weight tensors to FP16 "
            f"for native Tensor Core acceleration",
            flush=True,
        )
    return total_converted


def sanitize_model_precision(patcher, device: str | torch.device | None = None, force: bool = False) -> int:
    """Ensure Turing LoRA patches match native FP16 when appropriate without corrupting force_cast_weights."""
    if patcher is None:
        return 0
    if not force and not is_turing_or_older(device):
        return 0

    # Repair non-bool force_cast_weights if set by older code; ComfyUI expects a bool here
    # (setting a torch.dtype makes bool(comfy_force_cast_weights) True and disables quantized kernels).
    if hasattr(patcher, "force_cast_weights") and not isinstance(getattr(patcher, "force_cast_weights"), bool):
        try:
            setattr(patcher, "force_cast_weights", False)
        except Exception:
            pass

    return sanitize_patcher_precision(patcher, device=device, force=force)
