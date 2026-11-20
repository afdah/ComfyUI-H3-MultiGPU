# H3VM Dual-GPU Example Workflows

This folder contains pre-configured reference workflows for running MiniMax H3 on dual NVIDIA GPUs (including dual 11GB cards like 2x RTX 2080 Ti):

## Included Workflows

1. **`01_H3VM_DualGPU_Text2Video.json`**
   - Full Text-to-Video pipeline with prompt guider, INT8 DiT model, H3VM Core scheduling, and Dual-GPU Video VAE decoding.
   - Pre-configured with:
     - **Mode:** `Multi-GPU Capacity`
     - **Attention Kernel:** `OFFICIAL_OPTIMIZED (SageAttention)`
     - **Capacity Profile:** `Auto` (automatically switches to `SYMMETRIC | 50/50 Dual Identical` on identical GPUs)
     - **Activation Chunk Size:** `2048 (Recommended for 11GB-12GB)`

2. **`02_H3VM_DualGPU_Image2Video.json`**
   - Reference/Image-to-Video pipeline with image conditioning, H3VM Core scheduling, and Dual-GPU Video VAE decoding.
   - Pre-configured with identical dual-GPU capacity settings.

## How to Load in ComfyUI
- Simply drag and drop either `.json` file directly onto the ComfyUI browser canvas.
- Ensure ComfyUI was started with `--use-sage-attention` in `run_nvidia_gpu.bat` for optimal speed.
