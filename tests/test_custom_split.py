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

    # Test ratios below 50: 40/60 Custom Split
    plan_40 = resolve_capacity_vram_profile(
        "CUSTOM | Dial Exact Ratio (e.g. 57/43)",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=40,
    )
    assert plan_40["profile"] == "CUSTOM | 40/60 Split"
    assert abs(plan_40["mlp_primary_fraction"] - 0.40) < 1e-4
    assert plan_40["capacity_helper_heads"] == 34  # round(56 * 0.60) = 34
    print("Test 40/60 passed: MLP 40/60, helper_heads=34")

    # Test 43/57 Custom Split (mirror of 57/43)
    plan_43 = resolve_capacity_vram_profile(
        "CUSTOM",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=43,
    )
    assert plan_43["profile"] == "CUSTOM | 43/57 Split"
    assert abs(plan_43["mlp_primary_fraction"] - 0.43) < 1e-4
    assert plan_43["capacity_helper_heads"] == 32  # round(56 * 0.57) = 32
    print("Test 43/57 passed: MLP 43/57, helper_heads=32")

    # Test string parsing "45/55"
    plan_str_low = resolve_capacity_vram_profile(
        "45/55",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
    )
    assert plan_str_low["profile"] == "CUSTOM | 45/55 Split"
    assert abs(plan_str_low["mlp_primary_fraction"] - 0.45) < 1e-4
    assert plan_str_low["capacity_helper_heads"] == 31  # round(56 * 0.55) = 31
    print("Test string 45/55 passed: MLP 45/55, helper_heads=31")

    # Test clamp lower bound (e.g. 2 clamped to 5)
    plan_low = resolve_capacity_vram_profile(
        "CUSTOM",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=2,
    )
    assert plan_low["profile"] == "CUSTOM | 5/95 Split"
    assert abs(plan_low["mlp_primary_fraction"] - 0.05) < 1e-4
    assert plan_low["capacity_helper_heads"] == 53  # round(56 * 0.95) = 53
    print("Test clamp lower bound passed: clamped to 5/95")

    # Test clamp upper bound (e.g. 99 clamped to 95)
    plan_high = resolve_capacity_vram_profile(
        "CUSTOM",
        primary_total_gb=11.0,
        secondary_total_gb=11.0,
        custom_split_primary=99,
    )
    assert plan_high["profile"] == "CUSTOM | 95/5 Split"
    assert abs(plan_high["mlp_primary_fraction"] - 0.95) < 1e-4
    assert plan_high["capacity_helper_heads"] == 3  # round(56 * 0.05) = 3
    print("Test clamp upper bound passed: clamped to 95/5")


if __name__ == "__main__":
    test_custom_split_ratios()
    print("ALL CUSTOM SPLIT TESTS PASSED!")
