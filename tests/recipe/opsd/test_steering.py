import pytest
import torch

from recipe.opsd.steering import (
    find_transformer_layers,
    parse_layer_fraction_spec,
    resolve_fractional_layer_modules,
)


class _LayeredModule(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(num_layers)]
        )


def test_parse_layer_fraction_spec_resolves_qwen3_1p7b_production_band():
    assert parse_layer_fraction_spec("0.31-0.37", total_layers=28) == [9, 10]


def test_parse_layer_fraction_spec_resolves_singletons_and_ranges():
    assert parse_layer_fraction_spec("0.25-0.5,0.75", total_layers=8) == [2, 3, 4, 6]


def test_parse_layer_fraction_spec_clamps_fraction_one_to_last_layer():
    assert parse_layer_fraction_spec("1.0", total_layers=8) == [7]


def test_parse_layer_fraction_spec_rejects_descending_range():
    with pytest.raises(ValueError, match="descending"):
        parse_layer_fraction_spec("0.75-0.25", total_layers=8)


def test_find_and_resolve_hf_style_layers():
    module = _LayeredModule(num_layers=4)
    assert sorted(find_transformer_layers(module)) == [0, 1, 2, 3]
    selected = resolve_fractional_layer_modules(module, "0,0.5,1")
    assert sorted(selected) == [0, 2, 3]
    assert selected[2] is module.model.layers[2]


def test_resolve_rejects_unexpected_model_depth():
    module = _LayeredModule(num_layers=27)
    with pytest.raises(ValueError, match="depth violates"):
        resolve_fractional_layer_modules(
            module,
            "0.31-0.37",
            expected_total_layers=28,
            expected_layer_indices=[9, 10],
        )


def test_resolve_rejects_unexpected_layer_indices():
    module = _LayeredModule(num_layers=28)
    with pytest.raises(ValueError, match="layer resolution violates"):
        resolve_fractional_layer_modules(
            module,
            "0.31-0.37",
            expected_total_layers=28,
            expected_layer_indices=[8, 9],
        )


def test_resolve_accepts_exact_qwen3_1p7b_contract():
    module = _LayeredModule(num_layers=28)
    selected = resolve_fractional_layer_modules(
        module,
        "0.31-0.37",
        expected_total_layers=28,
        expected_layer_indices=[9, 10],
    )
    assert sorted(selected) == [9, 10]
