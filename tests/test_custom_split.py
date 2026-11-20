"""Unit tests for user-defined Custom Split ratios (e.g. 57/43, 58/42, 60/40) in H3VM Capacity Mode."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from h3vm.master_console import resolve_capacity_vram_profile
from h3vm.core_adapter_base import H3VMCoreConfig, execution_kwargs


def test_custom_split_ratios():
    # Test 56/44 Custom Split when selecting the dropdown option with "(e.g. 57/43)"
    plan_56 = resolve_capacity_vram_profile(
        "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=56,
    )
    assert plan_56["profile"] == "CUSTOM | 56/44 Split"
    assert abs(plan_56["mlp_primary_fraction"] - 0.56) < 1e-4
    assert plan_56["capacity_helper_heads"] == 25  # round(56 * 0.44) = 25
    print("Test 56/44 passed: MLP 56/44, helper_heads=25")

    # Test 57/43 Custom Split
    plan_57 = resolve_capacity_vram_profile(
        "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=57,
    )
    assert plan_57["profile"] == "CUSTOM | 57/43 Split"
    assert abs(plan_57["mlp_primary_fraction"] - 0.57) < 1e-4
    assert plan_57["capacity_helper_heads"] == 24  # round(56 * 0.43) = 24
    print("Test 57/43 passed: MLP 57/43, helper_heads=24")

    # Test 58/42 Custom Split
    plan_58 = resolve_capacity_vram_profile(
        "CUSTOM",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=58,
    )
    assert plan_58["profile"] == "CUSTOM | 58/42 Split"
    assert abs(plan_58["mlp_primary_fraction"] - 0.58) < 1e-4
    assert plan_58["capacity_helper_heads"] == 24  # round(56 * 0.42) = 24
    print("Test 58/42 passed: MLP 58/42, helper_heads=24")

    # Test string parsing "62/38"
    plan_str = resolve_capacity_vram_profile(
        "62/38",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
    )
    assert plan_str["profile"] == "CUSTOM | 62/38 Split"
    assert abs(plan_str["mlp_primary_fraction"] - 0.62) < 1e-4
    assert plan_str["capacity_helper_heads"] == 21  # round(56 * 0.38) = 21
    print("Test string 62/38 passed: MLP 62/38, helper_heads=21")

    # Test clamp lower bound (e.g. 40 clamped to 50)
    plan_low = resolve_capacity_vram_profile(
        "CUSTOM",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=40,
    )
    assert plan_low["profile"] == "CUSTOM | 50/50 Split"
    assert abs(plan_low["mlp_primary_fraction"] - 0.50) < 1e-4
    print("Test clamp lower bound passed: clamped to 50/50")


if __name__ == "__main__":
    test_custom_split_ratios()
    print("ALL CUSTOM SPLIT TESTS PASSED!")
