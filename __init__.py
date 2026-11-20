"""ComfyUI-H3-VRAM-Master v0.21.1 focused public release.

Public surface:
    MODEL -> H3VM Core -> MODEL + H3VM_MODE + H3VM_VAE_POLICY
    LATENT + VAE + H3VM_MODE + H3VM_VAE_POLICY -> H3VM Video VAE Decode -> IMAGE

Installing the plugin is side-effect free. Runtime patches are activated lazily only
when an H3VM execution path actually runs. Standard ComfyUI CLIP/text encoder,
VAE loader, sampler, prompt, seed, duration, and resolution controls are untouched.
"""
from __future__ import annotations

from pathlib import Path

try:
    H3VM_VERSION = Path(__file__).resolve().with_name("VERSION").read_text(encoding="utf-8").strip()
except Exception:
    H3VM_VERSION = "unknown"
__version__ = H3VM_VERSION

from .h3vm.master_console import (
    PRODUCT_MODES,
    PARTICIPATION_PRESETS,
    CAPACITY_VRAM_PROFILES,
    ATTENTION_KERNELS,
    CAPACITY_CHUNK_SIZES,
)

MODE4_PREDICTORS = (
    "SPECTRAL | Fast",
    "LINEAR | Standard",
)
WEB_DIRECTORY = "./web"


class H3VMCoreEngine:
    """Pure MODEL -> MODEL H3 multi-GPU execution adapter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "multi_gpu_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Enabled",
                        "label_off": "Disabled",
                        "tooltip": "Disable for single-GPU execution. Enable to use selected H3VM dual-GPU strategy.",
                    },
                ),
                "dual_vae_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Enabled",
                        "label_off": "Disabled",
                        "tooltip": "Unified control for H3VM Video VAE dual-GPU decoding. Can be disabled if GPU speed gap is large.",
                    },
                ),
                "gpu_mode": (
                    PRODUCT_MODES,
                    {
                        "default": "Multi-GPU Fast",
                        "tooltip": "Multi-GPU Fast: speed-first; Multi-GPU Capacity: VRAM-first (prevents OOM); Multi-GPU Background: conservative cooperative.",
                    },
                ),
                "mode4_predictor": (
                    MODE4_PREDICTORS,
                    {
                        "default": "SPECTRAL | Fast",
                        "tooltip": "Only effective in Multi-GPU Fast mode. SPECTRAL is default; LINEAR uses standard predictor.",
                    },
                ),
                "gpu_participation": (
                    PARTICIPATION_PRESETS,
                    {
                        "default": "Full | 100%",
                        "tooltip": "Secondary GPU workload share. Automatically maps to validated safe parameters.",
                    },
                ),
                "custom_participation": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Effective only when Secondary GPU Load is set to 'Custom'.",
                    },
                ),
                "expected_steps_hint": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Scheduling guidance for prefetch/end-step hints only. Real sampling boundaries are detected by the sampler.",
                    },
                ),
                "telemetry": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "attention_kernel": (
                    ATTENTION_KERNELS,
                    {
                        "default": "INT8_CURRENT (H3VM Native)",
                        "tooltip": "Attention kernel in Capacity mode. Set to OFFICIAL_OPTIMIZED (SageAttention) if ComfyUI is launched with --use-sage-attention for 2x-3x faster attention on Turing/2080 Ti.",
                    },
                ),
                "capacity_profile": (
                    CAPACITY_VRAM_PROFILES,
                    {
                        "default": "Auto",
                        "tooltip": "VRAM residency profile for Capacity mode. 'Auto' selects DISPLAY_SAFE (55/45) for dual identical GPUs. Choose 'CUSTOM' to dial your exact ratio like 57/43.",
                    },
                ),
                "custom_split_primary": (
                    "INT",
                    {
                        "default": 57,
                        "min": 50,
                        "max": 90,
                        "step": 1,
                        "tooltip": "Primary GPU share % (50-90) when capacity_profile is set to 'CUSTOM | Dial Exact Ratio'. Example: 57 means 57% Primary (headless) / 43% Secondary (monitor card).",
                    },
                ),
                "chunk_size": (
                    CAPACITY_CHUNK_SIZES,
                    {
                        "default": "2048 (Recommended for 11GB-12GB)",
                        "tooltip": "Activation tensor micro-chunk rows. 2048 avoids activation spikes on 11GB-12GB cards like RTX 2080 Ti.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "H3VM_MODE", "H3VM_VAE_POLICY")
    RETURN_NAMES = ("model", "mode", "vae_policy")
    FUNCTION = "adapt"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "H3 VRAM Master Core: Schedules MODEL execution and VRAM / dual-GPU distribution. "
        "Does not alter Prompt, Seed, resolution, duration, Sampler, Sigmas, actual Steps, CLIP, or VAE Loader."
    )

    def adapt(
        self,
        model,
        multi_gpu_enabled=True,
        dual_vae_enabled=True,
        gpu_mode="Multi-GPU Fast",
        mode4_predictor="SPECTRAL | Fast",
        gpu_participation="Full | 100%",
        custom_participation=100,
        expected_steps_hint=20,
        telemetry=True,
        attention_kernel="INT8_CURRENT (H3VM Native)",
        capacity_profile="Auto",
        custom_split_primary=57,
        chunk_size="2048 (Recommended for 11GB-12GB)",
        ui_language=None,
        **kwargs,
    ):
        del ui_language
        from .h3vm import activate_runtime

        activate_runtime()
        from .h3vm.master_console import (
            resolve_public_mode,
            resolve_participation_percent,
            resolve_attention_kernel,
            resolve_chunk_size,
        )
        from .h3vm.core_adapter import H3VMCoreConfig, adapt_model

        mode = resolve_public_mode(multi_gpu_enabled, gpu_mode)
        participation = resolve_participation_percent(gpu_participation, custom_participation)
        actual_attn_kernel = resolve_attention_kernel(attention_kernel)
        actual_chunk_rows = resolve_chunk_size(chunk_size)

        out = adapt_model(
            model,
            config=H3VMCoreConfig(
                mode=mode,
                primary_device="gpu:0",
                secondary_device="gpu:1",
                capacity_vram_profile=str(capacity_profile),
                expected_steps=max(1, int(expected_steps_hint)),
                telemetry=bool(telemetry),
                mode4_predictor=str(mode4_predictor),
                secondary_participation=float(participation),
                capacity_attention_kernel=actual_attn_kernel,
                capacity_mlp_chunk_rows=actual_chunk_rows,
                capacity_outproj_chunk_rows=actual_chunk_rows,
                custom_split_primary=max(50, min(90, int(custom_split_primary))),
                public_controls=True,
            ),
        )
        vae_policy = {
            "dual_vae_enabled": bool(dual_vae_enabled) and mode != "SINGLE_GPU",
        }
        return out, mode, vae_policy


class H3VMModeAwareVideoVAEDecode:
    """MiniMax H3 Video VAE decoder controlled by the H3VM Core VAE policy."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "mode": ("H3VM_MODE",),
                "vae_policy": ("H3VM_VAE_POLICY",),
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {
                        "tooltip": "Video VAE file to load on secondary GPU. Must match the weights of the primary VAE.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/H3VM"
    DESCRIPTION = (
        "MiniMax H3 Video VAE optional dual-GPU acceleration. Managed by H3VM Core policy. "
        "Uses native single-card VAE decode when disabled or in single-GPU mode. "
        "In dual-GPU mode, vae_name must use the same weights as the primary connected VAE."
    )

    @staticmethod
    def _native_decode(vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def decode(
        self,
        samples,
        vae,
        mode,
        vae_policy,
        vae_name,
        ui_language=None,
        **kwargs,
    ):
        del ui_language
        mode_text = str(mode).strip()
        policy = vae_policy if isinstance(vae_policy, dict) else {}
        use_dual = bool(policy.get("dual_vae_enabled", False)) and not mode_text.startswith("SINGLE_GPU")
        if not use_dual:
            from .h3vm import runtime_active
            if runtime_active():
                from .h3vm.dual_video_vae import _cleanup_h3_runtime
                _cleanup_h3_runtime()
            return (self._native_decode(vae, samples),)

        from .h3vm import activate_runtime
        activate_runtime()
        from .h3vm.dual_video_vae import dual_decode_h3_video

        return (
            dual_decode_h3_video(
                vae=vae,
                samples=samples,
                vae_name=str(vae_name),
                primary_device="gpu:0",
                secondary_device="gpu:1",
                cleanup_h3=True,
                telemetry=True,
                post_cleanup=True,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "H3VMCoreEngine": H3VMCoreEngine,
    "H3VMModeAwareVideoVAEDecode": H3VMModeAwareVideoVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VMCoreEngine": "H3VM Core | Model Execution Engine",
    "H3VMModeAwareVideoVAEDecode": "H3VM Video VAE | Dual-GPU Decoder",
}

print(
    f"[ComfyUI-H3-VRAM-Master {H3VM_VERSION}] focused plugin ready | runtime lazy | Core + optional Dual Video VAE",
    flush=True,
)
