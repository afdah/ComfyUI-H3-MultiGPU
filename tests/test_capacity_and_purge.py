"""Regression tests for H3VM TE/VAE purge, AIMDO VBAR watermarks, and CapacityFabric stream/memory fixes."""

import copy
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMFY_ROOT = ROOT.parents[1]
if (COMFY_ROOT / "comfy").is_dir() and str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _make_runtime_stub(primary, secondary=None, single_root=True, hot_cache=None):
    from h3vm.streaming_exact import StreamingExactRuntime

    rt = StreamingExactRuntime.__new__(StreamingExactRuntime)
    rt.primary_device = torch.device(primary)
    rt.secondary_device = torch.device(secondary) if secondary is not None else None
    rt.devices = {0: rt.primary_device, 1: rt.secondary_device}
    rt.single_root = bool(single_root)
    rt.hot_cache = hot_cache if hot_cache is not None else {0: 1.5, 1: 1.0}
    rt.patchers = {0: None, 1: None}
    rt.helper_patchers = {0: None, 1: None}
    rt._main_patcher_ref = None
    return rt


def test_purge_inactive_models_evicts_te_and_vae_keeps_h3():
    if not torch.cuda.is_available() or not (COMFY_ROOT / "comfy").is_dir():
        print("Skipping purge GPU test (CUDA or ComfyUI unavailable)")
        return

    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else None

    class FakeTE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(1024, 1024, dtype=torch.float16)

    class FakeVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(1024, 1024, dtype=torch.float16)

    class FakeH3(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(512, 512, dtype=torch.float16)

    te_p = ModelPatcher(FakeTE(), load_device=dev0, offload_device=torch.device("cpu"))
    vae_p = ModelPatcher(FakeVAE(), load_device=dev1 or dev0, offload_device=torch.device("cpu"))
    main_p = ModelPatcher(FakeH3(), load_device=dev0, offload_device=torch.device("cpu"))
    island_p = ModelPatcher(FakeH3(), load_device=dev0, offload_device=torch.device("cpu"))
    helper_p = ModelPatcher(FakeH3(), load_device=dev1 or dev0, offload_device=torch.device("cpu"))

    try:
        mm.load_models_gpu([te_p, main_p, island_p], force_full_load=True)
        mm.load_models_gpu([vae_p, helper_p], force_full_load=True)
        assert te_p.loaded_size() > 0
        assert vae_p.loaded_size() > 0

        rt = _make_runtime_stub(dev0, dev1)
        rt.bind_patchers(island_p, None, None, helper_p, main_patcher=main_p)
        rt._purge_inactive_models()

        loaded_models = [lm.model for lm in mm.current_loaded_models if lm.model is not None]
        assert te_p not in loaded_models and te_p.loaded_size() == 0
        assert vae_p not in loaded_models and vae_p.loaded_size() == 0
        assert main_p in loaded_models
        assert island_p in loaded_models
        assert helper_p in loaded_models
        print("Purge test passed: evicted inactive TE/VAE while preserving H3 patchers")
    finally:
        for p in (te_p, vae_p, main_p, island_p, helper_p):
            try:
                p.detach(True)
            except Exception:
                pass
        mm.cleanup_models()
        mm.soft_empty_cache()


def test_apply_vbar_watermarks_mock_and_real_aimdo():
    from h3vm.streaming_exact import GIB

    rt = _make_runtime_stub("cpu", "cpu", single_root=True, hot_cache={0: 2.0, 1: 1.5})

    root_vbar = MagicMock()
    root_vbar.max_size = 8 * GIB
    root_p = MagicMock()
    root_p._vbar_get.return_value = root_vbar

    helper_vbar = MagicMock()
    helper_vbar.max_size = 8 * GIB
    helper_p = MagicMock()
    helper_p._vbar_get.return_value = helper_vbar

    rt.bind_patchers(root_p, None, None, helper_p)
    rt._apply_vbar_watermarks(0)
    rt._apply_vbar_watermarks(1)

    root_vbar.set_watermark.assert_called_once_with(int(2.0 * GIB))
    helper_vbar.set_watermark.assert_called_once_with(int(1.5 * GIB))

    if torch.cuda.is_available():
        try:
            import importlib
            import comfy_aimdo.control as aimdo_ctrl
            if not getattr(aimdo_ctrl, "devctxs", None):
                aimdo_ctrl.init()
                aimdo_ctrl.init_device(0)
            import comfy_aimdo.model_vbar as aimdo_mv
            if aimdo_mv.lib is None and aimdo_ctrl.lib is not None:
                importlib.reload(aimdo_mv)
            ModelVBAR = aimdo_mv.ModelVBAR if aimdo_mv.lib is not None else None
        except Exception:
            aimdo_ctrl = None
            ModelVBAR = None

        if aimdo_ctrl is not None and ModelVBAR is not None:
            vbar = ModelVBAR(768 * 1024 * 1024, 0)
            try:
                real_p = MagicMock()
                real_p._vbar_get.return_value = vbar
                rt2 = _make_runtime_stub("cuda:0", None, single_root=True, hot_cache={0: 0.5, 1: 0.0})
                rt2.bind_patchers(real_p, None)
                initial_pages = vbar.get_watermark()
                rt2._apply_vbar_watermarks(0)
                capped_pages = vbar.get_watermark()
                assert capped_pages <= initial_pages
                assert capped_pages == (512 * 1024 * 1024) // (32 * 1024 * 1024)
                print(f"Real AIMDO VBAR watermark test passed: {initial_pages} -> {capped_pages} pages")
            finally:
                vbar.close()


def test_capacity_fabric_qkv_box_freed_and_dual_gpu_exactness():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not (COMFY_ROOT / "comfy").is_dir():
        print("Skipping dual-GPU CapacityFabric test (need >= 2 CUDA GPUs + ComfyUI)")
        return

    import comfy.ops
    import comfy_kitchen
    from h3vm.capacity_mode import CapacityFabric, _clone_attention_shared
    from h3vm.mlp_token_parallel import clone_mlp_shared
    from h3vm.post_attention_island import _clone_norm_shared, _mod_scale_shift_range, _mod_gate_range

    torch.manual_seed(0)
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    heads = 8
    hd = 64
    hidden = heads * hd
    ffn = 1024
    seq = 1024
    Lin = comfy.ops.disable_weight_init.Linear

    class SyntheticRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)

    class SyntheticAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = hd
            self.qkv_proj = Lin(hidden, 3 * hidden, bias=True, dtype=torch.float16)
            self.q_norm = SyntheticRMSNorm(hd)
            self.k_norm = SyntheticRMSNorm(hd)
            self.out_proj = Lin(hidden, hidden, bias=True, dtype=torch.float16)

    class SyntheticMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Lin(hidden, 2 * ffn, bias=True, dtype=torch.float16)
            self.fc2 = Lin(ffn, hidden, bias=True, dtype=torch.float16)

        def forward(self, x):
            return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    class SyntheticAdaLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Lin(hidden, 6 * hidden, bias=True, dtype=torch.float16)

        def forward(self, t_emb):
            return self.proj(t_emb).split(hidden, dim=-1)

    class SyntheticBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = SyntheticAdaLN()
            self.norm1 = SyntheticRMSNorm(hidden)
            self.attn = SyntheticAttn()
            self.norm2 = SyntheticRMSNorm(hidden)
            self.mlp = SyntheticMLP()

        def forward(self, x, t_emb, mod_segments, rope_freqs):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            h = self.norm1(x)
            _mod_scale_shift_range(h, shift_msa, scale_msa, mod_segments, 0)
            qkv = self.attn.qkv_proj(h)
            q, k, v = qkv.split(hidden, dim=-1)
            q = q.view(1, seq, heads, hd)
            k = k.view(1, seq, heads, hd)
            v = v.view(seq, heads, hd).transpose(0, 1).unsqueeze(0).contiguous()
            qw = self.attn.q_norm.weight
            kw = self.attn.k_norm.weight
            rot = int(rope_freqs.shape[-3]) * 2
            comfy_kitchen.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.attn.q_norm.eps, rot_dim=rot)
            q = q[0].transpose(0, 1).unsqueeze(0).contiguous()
            k = k[0].transpose(0, 1).unsqueeze(0).contiguous()
            from comfy.ldm.modules.attention import optimized_attention
            attn_heads = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True)
            flat = attn_heads.squeeze(0).transpose(0, 1).reshape(seq, hidden)
            attn_out = self.attn.out_proj(flat)
            x = _mod_gate_range(x.clone(), gate_msa, attn_out, mod_segments, 0)
            h2 = self.norm2(x)
            _mod_scale_shift_range(h2, shift_mlp, scale_mlp, mod_segments, 0)
            y2 = self.mlp(h2)
            return _mod_gate_range(x, gate_mlp, y2, mod_segments, 0)

    root_block = SyntheticBlock().to(dev0).eval().requires_grad_(False)
    helper_mlp = clone_mlp_shared(copy.deepcopy(root_block.mlp).to(dev1)).requires_grad_(False)
    helper_mlp.norm2_helper = _clone_norm_shared(copy.deepcopy(root_block.norm2).to(dev1)).requires_grad_(False)
    helper_attn = _clone_attention_shared(copy.deepcopy(root_block.attn).to(dev1)).requires_grad_(False)

    fabric = CapacityFabric(
        primary=dev0,
        secondary=dev1,
        owner_map={0: 0},
        helper_mlp_by_block={0: helper_mlp},
        helper_attn_by_block={0: helper_attn},
        primary_fraction=0.50,
        min_primary_fraction=0.50,
        max_primary_fraction=0.50,
        min_sequence_length=256,
        mlp_chunk_rows=512,
        outproj_chunk_rows=512,
        helper_heads=4,
        attention_kernel="OFFICIAL_OPTIMIZED",
    )
    fabric.set_sidecar_ready(True)
    fabric.begin_step(1)

    x = (torch.randn(seq, hidden, device=dev0, dtype=torch.float16) * 0.1)
    t_emb = (torch.randn(1, hidden, device=dev0, dtype=torch.float16) * 0.05)
    mod_segments = [(0, seq, 0)]
    rope_freqs = torch.zeros(1, seq, 1, hd // 2, 2, 2, device=dev0, dtype=torch.float16)
    rope_freqs[..., 0, 0] = 1.0
    rope_freqs[..., 1, 1] = 1.0

    # Verify _qkv_box clears raw QKV BEFORE prequantize_int8_attention runs
    qkv_box_len_inside_attn = []
    probe_qkv = [torch.randn(seq, 3 * 4 * hd, device=dev0, dtype=torch.float16)]
    orig_preq = comfy_kitchen.prequantize_int8_attention

    def spy_preq(q, k, v, scale=None, attn_mask=None):
        qkv_box_len_inside_attn.append(len(probe_qkv))
        return orig_preq(q, k, v, scale=scale, attn_mask=attn_mask)

    fabric_int8 = CapacityFabric(
        primary=dev0,
        secondary=dev1,
        owner_map={0: 0},
        helper_mlp_by_block={0: helper_mlp},
        helper_attn_by_block={0: helper_attn},
        helper_heads=4,
        attention_kernel="INT8_CURRENT",
    )
    comfy_kitchen.prequantize_int8_attention = spy_preq
    try:
        with torch.no_grad():
            out_qkv = fabric_int8._qkv_norm_rope_attention(
                root_block.attn, probe_qkv[0], rope_freqs, {}, 4, _qkv_box=probe_qkv
            )
        assert torch.isfinite(out_qkv).all()
        assert qkv_box_len_inside_attn == [0], f"Expected _qkv_box cleared before attention, got {qkv_box_len_inside_attn}"
        del out_qkv
    finally:
        comfy_kitchen.prequantize_int8_attention = orig_preq
        fabric_int8.close()

    with torch.no_grad():
        ref = root_block(x.clone(), t_emb, mod_segments, rope_freqs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            out = fabric.execute_block(0, root_block, x.clone(), t_emb, mod_segments, rope_freqs)
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)

    stream_warnings = [w for w in caught if "original creation stream matches" in str(w.message)]
    assert not stream_warnings, f"Unexpected same-stream record_stream warnings: {[str(w.message) for w in stream_warnings]}"
    assert torch.isfinite(out).all()
    max_diff = float((out - ref).abs().max().item())
    assert max_diff < 5e-2, f"Dual-GPU CapacityFabric output diverged from reference: max_diff={max_diff}"
    fabric.close()
    print(f"Dual-GPU CapacityFabric exactness + zero-stream-warning test passed! max_diff={max_diff:.6f}")


def test_streaming_runtime_and_capacity_end_to_end():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not (COMFY_ROOT / "comfy").is_dir():
        print("Skipping end-to-end StreamingExactRuntime test (need >= 2 CUDA GPUs + ComfyUI)")
        return

    import comfy.model_management as mm
    import comfy.ops
    from comfy.model_patcher import ModelPatcher
    from h3vm.capacity_mode import CapacityFabric, build_capacity_helper_maps
    from h3vm.master_console import resolve_capacity_vram_profile
    from h3vm.post_attention_island import _mod_scale_shift_range, _mod_gate_range
    from h3vm.streaming_exact import StreamingExactRuntime

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    cpu = torch.device("cpu")
    heads, hd, ffn, seq = 8, 64, 512, 512
    hidden = heads * hd
    Lin = comfy.ops.disable_weight_init.Linear

    class Norm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16), requires_grad=False)
            self.eps = 1e-5

        def forward(self, x):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = hd
            self.qkv_proj = Lin(hidden, 3 * hidden, bias=True, dtype=torch.float16)
            self.q_norm = Norm(hd)
            self.k_norm = Norm(hd)
            self.out_proj = Lin(hidden, hidden, bias=True, dtype=torch.float16)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Lin(hidden, 2 * ffn, bias=True, dtype=torch.float16)
            self.fc2 = Lin(ffn, hidden, bias=True, dtype=torch.float16)

    class AdaLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Lin(hidden, 6 * hidden, bias=True, dtype=torch.float16)

        def forward(self, t_emb):
            return self.proj(t_emb).split(hidden, dim=-1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = AdaLN()
            self.norm1 = Norm(hidden)
            self.attn = Attn()
            self.norm2 = Norm(hidden)
            self.mlp = MLP()

    torch.manual_seed(1)
    blocks_list = [Block().eval().requires_grad_(False), Block().eval().requires_grad_(False)]
    owner_map = {0: 0, 1: 0}
    _, sec_helpers, helper_mlp, helper_attn, helper_pkt = build_capacity_helper_maps(
        blocks_list, owner_map, [0, 1]
    )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = Lin(256, 256, bias=False, dtype=torch.float16)

    te_p = ModelPatcher(FakeModel().eval().requires_grad_(False), load_device=dev0, offload_device=cpu)
    vae_p = ModelPatcher(FakeModel().eval().requires_grad_(False), load_device=dev1, offload_device=cpu)
    main_p = ModelPatcher(FakeModel().eval().requires_grad_(False), load_device=dev0, offload_device=cpu)
    root_p = ModelPatcher(nn.ModuleDict({"0": blocks_list[0], "1": blocks_list[1]}), load_device=dev0, offload_device=cpu)
    helper_p = ModelPatcher(nn.ModuleDict({str(k): v for k, v in sec_helpers.items()}), load_device=dev1, offload_device=cpu)

    plan = resolve_capacity_vram_profile("CUSTOM", 11.0, 11.0, custom_split_primary=45)
    runtime = StreamingExactRuntime(
        dev0, dev1, {0: blocks_list[0], 1: blocks_list[1]}, owner_map,
        space=None,
        expected_steps=1,
        primary_runtime_reserve_gb=plan["primary_runtime_reserve_gb"],
        secondary_runtime_reserve_gb=plan["secondary_runtime_reserve_gb"],
        primary_hot_cache_gb=plan["primary_hot_cache_gb"],
        secondary_hot_cache_gb=plan["secondary_hot_cache_gb"],
        one_ahead_prefetch=False,
        trim_on_stripe_boundary=True,
        telemetry=False,
        hard_cleanup_after_sample=True,
        single_root=True,
        single_root_trim_interval=plan["single_root_trim_interval"],
    )
    fabric = CapacityFabric(
        primary=dev0, secondary=dev1, owner_map=owner_map,
        helper_mlp_by_block=helper_mlp, helper_attn_by_block=helper_attn, helper_packet_by_block=helper_pkt,
        primary_fraction=plan["mlp_primary_fraction"],
        min_primary_fraction=plan["mlp_primary_fraction"],
        max_primary_fraction=plan["mlp_primary_fraction"],
        min_sequence_length=128,
        mlp_chunk_rows=256, outproj_chunk_rows=256,
        helper_heads=plan["capacity_helper_heads"],
        attention_kernel="INT8_CURRENT",
        telemetry=False,
    )
    runtime.bind_patchers(root_p, None, None, helper_p, main_patcher=main_p)
    runtime.bind_mlp_fabric(fabric)

    try:
        mm.load_models_gpu([te_p, main_p], force_full_load=True)
        mm.load_models_gpu([vae_p], force_full_load=True)
        assert te_p.loaded_size() > 0 and vae_p.loaded_size() > 0

        runtime.sampling_begin()
        loaded = [lm.model for lm in mm.current_loaded_models if lm.model is not None]
        assert te_p not in loaded and te_p.loaded_size() == 0
        assert vae_p not in loaded and vae_p.loaded_size() == 0
        assert main_p in loaded

        # Force immediate weights onto assigned devices if non-dynamic test patcher left tensors on CPU
        root_p.model.to(dev0)
        helper_p.model.to(dev1)

        fabric.set_sidecar_ready(True)
        fabric.begin_step(1)
        x = torch.randn(seq, hidden, device=dev0, dtype=torch.float16) * 0.05
        t_emb = torch.randn(1, hidden, device=dev0, dtype=torch.float16) * 0.05
        rope_freqs = torch.zeros(1, seq, 1, hd // 2, 2, 2, device=dev0, dtype=torch.float16)
        rope_freqs[..., 0, 0] = 1.0
        rope_freqs[..., 1, 1] = 1.0
        with torch.no_grad():
            for idx, blk in enumerate(blocks_list):
                x = fabric.execute_block(idx, blk, x, t_emb, [(0, seq, 0)], rope_freqs)
                fabric.on_block_complete(idx)
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)
        assert torch.isfinite(x).all()
        runtime.sampling_end()
        print("End-to-end StreamingExactRuntime + CapacityFabric + purge test passed!")
    finally:
        fabric.close()
        runtime.close()
        for p in (te_p, vae_p, main_p, root_p, helper_p):
            try:
                p.detach(True)
            except Exception:
                pass
        mm.cleanup_models()
        mm.soft_empty_cache()


if __name__ == "__main__":
    test_purge_inactive_models_evicts_te_and_vae_keeps_h3()
    test_apply_vbar_watermarks_mock_and_real_aimdo()
    test_capacity_fabric_qkv_box_freed_and_dual_gpu_exactness()
    test_streaming_runtime_and_capacity_end_to_end()
    print("ALL CAPACITY & PURGE TESTS PASSED!")
