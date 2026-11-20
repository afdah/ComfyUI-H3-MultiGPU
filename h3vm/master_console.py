"""H3VM Core policy helpers.

Pure/lightweight policy module.  No torch/comfy imports at module import time.
The public node entrypoint lives in the package __init__.py. This module stays
lightweight and contains policy/value translation only.
"""
from __future__ import annotations

import math
import os

MODES = (
    "SINGLE_GPU",
    "DUAL_QUIET",
    "DUAL_CAPACITY",
    "DUAL_SYNC_ACCEL",
    "DUAL_EXACT_SP",
)

MASTER_GPU_MODES = (
    "SINGLE_GPU | Single Card Baseline",
    "DUAL_QUIET | Dual-GPU Background",
    "DUAL_CAPACITY | Dual-GPU Capacity (OOM Prevention)",
    "DUAL_SYNC_ACCEL | Dual-GPU Speed (Fast)",
    "DUAL_EXACT_SP | Dual-GPU Exact SP (Preview)",
)

# Public product surface is intentionally small. Internal/legacy backend names
# remain accepted by normalize_mode() so old workflows and lab nodes keep working.
PRODUCT_MODES = ("Multi-GPU Fast", "Multi-GPU Capacity", "Multi-GPU Background")
LEGACY_PRODUCT_MODE_MAP = {
    "Single Card": "SINGLE_GPU",
    "Dual-GPU Cooperative": "DUAL_QUIET",
    "Dual-GPU Capacity": "DUAL_CAPACITY",
    "Dual-GPU Speed": "DUAL_SYNC_ACCEL",
    "Dual-GPU Exact SP": "DUAL_EXACT_SP",
}
PUBLIC_PRODUCT_MODE_MAP = {
    "Multi-GPU Fast": "DUAL_SYNC_ACCEL",
    "Multi-GPU Capacity": "DUAL_CAPACITY",
    "Multi-GPU Background": "DUAL_QUIET",
}
PARTICIPATION_PRESETS = ("Full | 100%", "High | 75%", "Medium | 50%", "Low | 25%", "Custom")
_PARTICIPATION_VALUES = {
    "Full | 100%": 100.0,
    "High | 75%": 75.0,
    "Medium | 50%": 50.0,
    "Low | 25%": 25.0,
}
_CAPACITY_PARTICIPATION_ANCHORS = (
    (25.0, "SAFE | Conservative Stable"),
    (50.0, "RESIDENT4 | Resident+ (More VRAM)"),
    (75.0, "RESIDENT8 | High Resident (Less Transfer)"),
    (100.0, "BALANCE | Balanced"),
)
STOCK_LORA = "None | Stock Model"
STOCK_STEPS = "20 Steps | Stock Model"
CUSTOM_STEPS = "Custom Steps"


def validate_sampling_contract(mode, turbo_lora, steps, style_lora_stack=None, custom_steps=8):
    normalize_mode(mode)
    del style_lora_stack
    if str(steps).strip() == CUSTOM_STEPS:
        requested = max(1, int(custom_steps))
    else:
        requested = step_value(steps)
    _, note = resolve_steps(turbo_lora, requested)
    return requested, note


def normalize_mode(value: str) -> str:
    text = str(value).strip()
    if text in PUBLIC_PRODUCT_MODE_MAP:
        return PUBLIC_PRODUCT_MODE_MAP[text]
    if text in LEGACY_PRODUCT_MODE_MAP:
        return LEGACY_PRODUCT_MODE_MAP[text]
    for mode in MODES:
        if text == mode or text.startswith(mode + "｜"):
            return mode
    raise ValueError(f"Unknown H3VM mode: {value}")


def resolve_public_mode(multi_gpu_enabled, gpu_mode) -> str:
    if not bool(multi_gpu_enabled):
        return "SINGLE_GPU"
    return normalize_mode(gpu_mode)


def resolve_participation_percent(preset, custom=100) -> float:
    key = str(preset).strip()
    if key in _PARTICIPATION_VALUES:
        return float(_PARTICIPATION_VALUES[key])
    if key == "Custom":
        return float(max(1.0, min(100.0, float(custom))))
    try:
        return float(max(1.0, min(100.0, float(key))))
    except Exception as exc:
        raise ValueError(f"Unknown H3VM participation preset: {preset}") from exc


def capacity_profile_for_participation(
    percent,
    primary_total_gb: float | None = None,
    secondary_total_gb: float | None = None,
) -> str:
    value = max(1.0, min(100.0, float(percent)))
    # For dual identical GPUs, auto-select DISPLAY_SAFE (55/45) so the monitor card
    # retains 2.2GB+ buffer and never lags the desktop.
    if primary_total_gb is not None and secondary_total_gb is not None:
        try:
            if abs(float(primary_total_gb) - float(secondary_total_gb)) <= 1.0 and value >= 80.0:
                return "DISPLAY_SAFE | 55/45 Monitor Protected"
        except Exception:
            pass
    return min(_CAPACITY_PARTICIPATION_ANCHORS, key=lambda item: (abs(item[0] - value), item[0]))[1]


def scale_secondary_work(base_count: int, percent, *, minimum: int = 1, maximum: int | None = None) -> int:
    base = max(0, int(base_count))
    if base == 0:
        return 0
    value = max(1.0, min(100.0, float(percent)))
    scaled = int(base * value / 100.0 + 0.5)
    if maximum is None:
        maximum = base
    return max(int(minimum), min(int(maximum), scaled))

STEP_PRESETS = ("4 Steps | Fast", "6 Steps | Balanced", "8 Steps | Quality", CUSTOM_STEPS)
ASPECTS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")

CAPACITY_VRAM_PROFILES = (
    "Auto",
    "DISPLAY_SAFE | 55/45 Monitor Protected",
    "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
    "BALANCE | Balanced (60/40)",
    "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "SAFE | Conservative Stable",
    "RESIDENT4 | Resident+ (More VRAM)",
    "RESIDENT8 | High Resident (Less Transfer)",
    "MLP36 | Secondary 36%",
    "MLP40 | Secondary 40% (High Load)",
    "HEAD20 | Attention+ (Secondary Extra Heads)",
    "MAX_TEST | Extreme",
)
ATTENTION_KERNELS = (
    "INT8_CURRENT (H3VM Native)",
    "OFFICIAL_OPTIMIZED (SageAttention)",
)
CAPACITY_CHUNK_SIZES = (
    "2048 (Recommended for 11GB-12GB)",
    "4096 (Standard for 16GB+)",
    "1024 (Ultra Low VRAM)",
)


def resolve_attention_kernel(value: str) -> str:
    raw = str(value).upper().strip()
    if "OFFICIAL" in raw or "SAGE" in raw:
        return "OFFICIAL_OPTIMIZED"
    return "INT8_CURRENT"


def resolve_chunk_size(value) -> int:
    try:
        digits = "".join(c for c in str(value).split()[0] if c.isdigit())
        if digits:
            return max(256, int(digits))
    except Exception:
        pass
    return 2048

_CAPACITY_VRAM_PROFILE_RATIOS = {
    "DISPLAY_SAFE | 55/45 Monitor Protected": {"primary_reserve_frac": 0.35, "secondary_reserve_frac": 0.40, "primary_cache_frac": 0.25, "secondary_cache_frac": 0.20, "trim_interval": 4, "mlp_primary_fraction": 0.56, "helper_heads": 24},
    "BALANCE | Balanced (60/40)": {"primary_reserve_frac": 0.37, "secondary_reserve_frac": 0.35, "primary_cache_frac": 0.22, "secondary_cache_frac": 0.28, "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 20},
    "SYMMETRIC | 50/50 Dual Identical (Headless)": {"primary_reserve_frac": 0.35, "secondary_reserve_frac": 0.35, "primary_cache_frac": 0.25, "secondary_cache_frac": 0.25, "trim_interval": 4, "mlp_primary_fraction": 0.50, "helper_heads": 28},
    "SAFE | Conservative Stable": {"primary_reserve_frac": 0.52, "secondary_reserve_frac": 0.55, "primary_cache_frac": 0.047, "secondary_cache_frac": 0.094, "trim_interval": 1, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "RESIDENT4 | Resident+ (More VRAM)": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24, "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "RESIDENT8 | High Resident (Less Transfer)": {"primary_reserve_frac": 0.36, "secondary_reserve_frac": 0.29, "primary_cache_frac": 0.24, "secondary_cache_frac": 0.34, "trim_interval": 8, "mlp_primary_fraction": 0.68, "helper_heads": 16},
    "MLP36 | Secondary 36%": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.38, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.24, "trim_interval": 4, "mlp_primary_fraction": 0.64, "helper_heads": 16},
    "MLP40 | Secondary 40% (High Load)": {"primary_reserve_frac": 0.41, "secondary_reserve_frac": 0.35, "primary_cache_frac": 0.18, "secondary_cache_frac": 0.27, "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 16},
    "HEAD20 | Attention+ (Secondary Extra Heads)": {"primary_reserve_frac": 0.43, "secondary_reserve_frac": 0.42, "primary_cache_frac": 0.16, "secondary_cache_frac": 0.20, "trim_interval": 4, "mlp_primary_fraction": 0.68, "helper_heads": 20},
    "BALANCE | Balanced": {"primary_reserve_frac": 0.37, "secondary_reserve_frac": 0.32, "primary_cache_frac": 0.22, "secondary_cache_frac": 0.30, "trim_interval": 4, "mlp_primary_fraction": 0.60, "helper_heads": 20},
    "MAX_TEST | Extreme": {"primary_reserve_frac": 0.28, "secondary_reserve_frac": 0.22, "primary_cache_frac": 0.30, "secondary_cache_frac": 0.42, "trim_interval": 8, "mlp_primary_fraction": 0.56, "helper_heads": 20},
}
_CAPACITY_PROFILE_ALIASES = {
    "CUSTOM": "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
    "CUSTOM | Dial Exact Ratio (e.g. 57/43)": "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
    "DISPLAY_SAFE": "DISPLAY_SAFE | 55/45 Monitor Protected",
    "DISPLAY_SAFE | 55/45 Monitor Protected": "DISPLAY_SAFE | 55/45 Monitor Protected",
    "55/45": "DISPLAY_SAFE | 55/45 Monitor Protected",
    "MONITOR_SAFE": "DISPLAY_SAFE | 55/45 Monitor Protected",
    "SYMMETRIC": "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "SYMMETRIC | 50/50 Dual Identical": "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "SYMMETRIC | 50/50 Dual Identical (Headless)": "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "50/50": "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "DUAL_IDENTICAL": "SYMMETRIC | 50/50 Dual Identical (Headless)",
    "BALANCE": "BALANCE | Balanced (60/40)",
    "BALANCE | Balanced": "BALANCE | Balanced (60/40)",
    "BALANCE | Balanced (60/40)": "BALANCE | Balanced (60/40)",
    "SAFE": "SAFE | Conservative Stable",
    "SAFE | Conservative Stable": "SAFE | Conservative Stable",
    "RESIDENT4": "RESIDENT4 | Resident+ (More VRAM)",
    "RESIDENT8": "RESIDENT8 | High Resident (Less Transfer)",
    "MLP36": "MLP36 | Secondary 36%",
    "MLP40": "MLP40 | Secondary 40% (High Load)",
    "HEAD20": "HEAD20 | Attention+ (Secondary Extra Heads)",
    "MAX_TEST": "MAX_TEST | Extreme",
}


def resolve_capacity_vram_profile(
    profile: str,
    primary_total_gb: float,
    secondary_total_gb: float,
    custom_split_primary: int = 57,
) -> dict:
    import re
    raw = str(profile).strip()
    key = _CAPACITY_PROFILE_ALIASES.get(raw, raw)

    is_custom = "CUSTOM" in key.upper() or "CUSTOM" in raw.upper()
    custom_match = re.search(r"(\d{2})\s*[/_:-]\s*(\d{2})", raw)
    if is_custom:
        # User selected Custom profile mode: always use the custom_split_primary integer widget,
        # never let example text like "(e.g. 57/43)" override the user's explicit dial setting.
        p_val = int(custom_split_primary)
        p_val = max(5, min(95, p_val))
        s_val = 100 - p_val
        p_frac = p_val / 100.0
        helper_heads = max(1, min(54, int(round(56.0 * (s_val / 100.0)))))
        key = f"CUSTOM | {p_val}/{s_val} Split"
        r = {
            "primary_reserve_frac": 0.35,
            "secondary_reserve_frac": 0.40,
            "primary_cache_frac": 0.25,
            "secondary_cache_frac": 0.20,
            "trim_interval": 4,
            "mlp_primary_fraction": p_frac,
            "helper_heads": helper_heads,
        }
    elif key in _CAPACITY_VRAM_PROFILE_RATIOS:
        r = _CAPACITY_VRAM_PROFILE_RATIOS[key]
    elif custom_match:
        # User passed a raw ratio string like "58/42" or "40/60" directly
        p_val = int(custom_match.group(1))
        p_val = max(5, min(95, p_val))
        s_val = 100 - p_val
        p_frac = p_val / 100.0
        helper_heads = max(1, min(54, int(round(56.0 * (s_val / 100.0)))))
        key = f"CUSTOM | {p_val}/{s_val} Split"
        r = {
            "primary_reserve_frac": 0.35,
            "secondary_reserve_frac": 0.40,
            "primary_cache_frac": 0.25,
            "secondary_cache_frac": 0.20,
            "trim_interval": 4,
            "mlp_primary_fraction": p_frac,
            "helper_heads": helper_heads,
        }
    else:
        if key not in _CAPACITY_VRAM_PROFILE_RATIOS:
            raise ValueError(f"Unknown Capacity profile: {profile}")
        r = _CAPACITY_VRAM_PROFILE_RATIOS[key]

    ptotal = max(1.0, float(primary_total_gb))
    stotal = max(1.0, float(secondary_total_gb))

    def _legal(total, reserve_frac, cache_frac):
        reserve = max(0.5, total * float(reserve_frac))
        reserve = min(reserve, max(0.5, total - 0.5))
        usable = max(0.0, total - reserve)
        cache = max(0.0, total * float(cache_frac))
        cache = min(cache, usable)
        return round(reserve, 3), round(cache, 3)

    p_reserve, p_cache = _legal(ptotal, r["primary_reserve_frac"], r["primary_cache_frac"])
    s_reserve, s_cache = _legal(stotal, r["secondary_reserve_frac"], r["secondary_cache_frac"])
    return {
        "profile": key,
        "requested_profile": raw,
        "primary_runtime_reserve_gb": p_reserve,
        "secondary_runtime_reserve_gb": s_reserve,
        "primary_hot_cache_gb": p_cache,
        "secondary_hot_cache_gb": s_cache,
        "single_root_trim_interval": int(r["trim_interval"]),
        "mlp_primary_fraction": float(r["mlp_primary_fraction"]),
        "capacity_helper_heads": int(r["helper_heads"]),
    }

RESOLUTION_PRESETS = (
    "0.2MP | Preview", "0.5MP | Fast", "0.7MP | Mid-High",
    "768P | Native HD", "1.5MP | Ultra HD", "2.0MP | 1080p", "CUSTOM | Custom"
)
_REFERENCE_16_9 = {
    "0.2MP | Preview": (608, 352), "0.5MP | Fast": (960, 544), "0.7MP | Mid-High": (1152, 640),
    "768P | Native HD": (1344, 768), "1.5MP | Ultra HD": (1664, 928), "2.0MP | 1080p": (1920, 1088),
}
_TARGET_MP = {
    "0.2MP | Preview": 0.20, "0.5MP | Fast": 0.50, "0.7MP | Mid-High": 0.70, "768P | Native HD": None,
    "1.5MP | Ultra HD": 1.50, "2.0MP | 1080p": 2.00,
}
_RATIO = {"16:9": 16/9, "9:16": 9/16, "1:1": 1.0, "4:3": 4/3, "3:4": 3/4, "21:9": 21/9}
_NATIVE_SHORT = 768

def _round32(x: float) -> int: return max(32, int(round(float(x) / 32.0)) * 32)
def _ceil32(x: float) -> int: return max(32, int(math.ceil(float(x) / 32.0)) * 32)
def resolve_resolution(aspect, preset, custom_width, custom_height):
    aspect = str(aspect); preset = str(preset)
    if aspect not in _RATIO: raise ValueError(f"Unsupported aspect ratio: {aspect}")
    if preset.startswith("CUSTOM"): return _round32(custom_width), _round32(custom_height)
    if aspect == "16:9" and preset in _REFERENCE_16_9: return _REFERENCE_16_9[preset]
    if aspect == "9:16" and preset in _REFERENCE_16_9:
        w, h = _REFERENCE_16_9[preset]; return h, w
    ratio = _RATIO[aspect]
    if preset == "768P | Native HD":
        if ratio >= 1.0: h = _NATIVE_SHORT; w = _round32(h * ratio)
        else: w = _NATIVE_SHORT; h = _round32(w / ratio)
        return int(w), int(h)
    mp = _TARGET_MP.get(preset)
    if mp is None: raise ValueError(f"Unsupported resolution preset: {preset}")
    pixels = float(mp) * 1_000_000.0; w = math.sqrt(pixels * ratio); h = w / ratio; return _ceil32(w), _ceil32(h)

def duration_to_length(seconds: float) -> int:
    frames = max(5, int(round(max(float(seconds), 0.05) * 24.0))); return int(frames + (5 - (frames % 17)) % 17)

def step_value(step_preset: str) -> int:
    if isinstance(step_preset, (int, float)): return max(1, int(step_preset))
    text = str(step_preset).strip()
    if text == STOCK_STEPS or text.startswith("20"): return 20
    if text.startswith("4"): return 4
    if text.startswith("6"): return 6
    if text.startswith("8"): return 8
    raise ValueError(f"Unknown H3 Turbo step preset: {step_preset}")

def lora_family(lora_name: str) -> str:
    if str(lora_name).strip() == STOCK_LORA: return "STOCK_H3"
    name = os.path.basename(str(lora_name)).lower()
    if "fl2v_turbo_8step_v1.0" in name: return "LIGHTX2V_V1_8STEP"
    if "fl2v_turbo_4step_v1.0_768p" in name: return "LIGHTX2V_V1_4STEP_768P"
    if "turbo_v4_step600" in name: return "LARRY_V4_600"
    if "turbo_4step" in name or "ckpt850" in name or "ckpt500" in name: return "LARRY_4STEP_LINE"
    return "CUSTOM_TURBO"

def resolve_steps(lora_name: str, step_preset: str):
    requested = step_value(step_preset); family = lora_family(lora_name)
    if family == "LIGHTX2V_V1_4STEP_768P": return (requested, "Current LoRA is labeled as 4-step; running as requested without forcing step count.") if requested != 4 else (4, None)
    if family == "LIGHTX2V_V1_8STEP" and requested == 6: return 6, "LightX2V v1.0 officially documents 8 steps (and 4-step use); 6-step is a compatibility experiment."
    if family == "LARRY_V4_600" and requested == 4: return 4, "Larry v4-600 supports 4 steps, but 6-8 is the current quality recommendation."
    return requested, None

def mode_summary(mode:str)->str:
    return {"SINGLE_GPU":"GPU0 single-card baseline","DUAL_QUIET":"dual-GPU background / reserved-compute mode","DUAL_CAPACITY":"dual-GPU capacity mode","DUAL_SYNC_ACCEL":"dual-GPU speed mode","DUAL_EXACT_SP":"internal exact 2-rank reference backend"}.get(str(mode),str(mode))

def apply_sigma_shift(model,shift_video:float=12.0,shift_audio:float=3.0):
    import comfy.model_sampling
    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST): pass
    original=model.get_model_object("model_sampling"); model_sampling=ModelSamplingAdvanced(model.model.model_config); model_sampling.set_parameters(shift=float(shift_video),audio_shift=float(shift_audio))
    if hasattr(original,"noise_scale"): model_sampling.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling",model_sampling); to=model.model_options["transformer_options"]=model.model_options.get("transformer_options",{}).copy(); to["minimax_h3_sigma_shift_video"]=float(shift_video); to["minimax_h3_sigma_shift_audio"]=float(shift_audio); return model
