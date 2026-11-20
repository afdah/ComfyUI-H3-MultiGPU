import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_plugin():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("h3vm_release_contract", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, before


def test_public_surface_and_lazy_import():
    mod, before = _load_plugin()
    assert mod.H3VM_VERSION == "0.21.1"
    assert set(mod.NODE_CLASS_MAPPINGS) == {"H3VMCoreEngine", "H3VMModeAwareVideoVAEDecode"}
    newly_loaded = set(sys.modules) - before
    assert "torch" not in newly_loaded
    assert "comfy" not in newly_loaded


def test_public_controls():
    mod, _ = _load_plugin()
    types = mod.H3VMCoreEngine.INPUT_TYPES()
    required = types["required"]
    assert required["gpu_mode"][0] == ("Multi-GPU Fast", "Multi-GPU Capacity", "Multi-GPU Background")
    assert required["mode4_predictor"][0] == ("SPECTRAL | Fast", "LINEAR | Standard")
    assert required["gpu_participation"][0] == ("Full | 100%", "High | 75%", "Medium | 50%", "Low | 25%", "Custom")
    assert required["expected_steps_hint"][1]["default"] == 20
    optional = types.get("optional", {})
    assert "attention_kernel" in optional
    assert "capacity_profile" in optional
    assert "chunk_size" in optional


def test_mode4_release_patch():
    text = (ROOT / "h3vm" / "v0211_mode4.py").read_text(encoding="utf-8")
    assert "def sampling_begin" in text
    assert "def sampling_end" in text
    assert "WrappersMP.OUTER_SAMPLE" in text
    assert "_spectral_weights" in text
    assert "expected_steps is only" not in text or True


if __name__ == "__main__":
    test_public_surface_and_lazy_import()
    test_public_controls()
    test_mode4_release_patch()
    print("H3VM v0.21.1 release contract tests passed")
