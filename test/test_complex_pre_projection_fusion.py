from __future__ import annotations

import math

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_pre_projection_fusion import ComplexPreProjectionFusion
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexPreProjectionFusionConfig,
)
from test.complex_fixtures import write_geometry_npz


def _inputs(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    batch_size = 2
    base_response = torch.tensor(
        [
            [[2.0, 4.0, 0.5], [1.0, 2.0, 3.0]],
            [[1.0, -2.0, 0.25], [0.5, 1.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]],
        dtype=torch.float64,
    )
    x_amplitude = torch.tensor(
        [[2.0, 4.0], [1.0, 3.0]],
        dtype=torch.float64,
    )
    y_amplitude = torch.tensor(
        [[3.0, 5.0, 7.0], [2.0, 4.0, 6.0]],
        dtype=torch.float64,
    )
    assert base_response.shape == (batch_size, 2, geometry.num_points)
    return geometry, base_response, rhs, x_amplitude, y_amplitude


def _module(
    *,
    gate: float = 0.05,
    hidden_dim: int = 8,
    depth: int = 1,
    mode: str = "residual_correction",
    combination: str = "convex_average",
    nonlinear_final_init_scale: float = 0.0,
) -> ComplexPreProjectionFusion:
    return ComplexPreProjectionFusion(
        ComplexPreProjectionFusionConfig(
            enabled=True,
            mode=mode,
            combination=combination,
            nonlinear_hidden_dim=hidden_dim,
            nonlinear_depth=depth,
            nonlinear_final_init_scale=nonlinear_final_init_scale,
            gate_initial_value=gate,
        ),
        activation="tanh",
        use_bias=True,
        dtype=torch.float64,
    )


def test_pre_projection_fusion_is_identity_at_initialization(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(gate=0.2)

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    torch.testing.assert_close(result.fused_response, base_response)
    torch.testing.assert_close(
        result.blended_correction,
        torch.zeros_like(result.blended_correction),
    )
    torch.testing.assert_close(
        result.fused_physical,
        result.base_physical,
    )
    assert float(result.gate.item()) == pytest.approx(0.2)


def test_pre_projection_fusion_changes_only_physical_difference_mode(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(gate=0.25)
    with torch.no_grad():
        module.linear_correction.weight.copy_(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        )

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    expected_correction = (1.0 - result.gate) * result.base_difference
    torch.testing.assert_close(result.blended_correction, expected_correction)
    torch.testing.assert_close(
        result.fused_difference,
        result.base_difference + expected_correction,
    )
    torch.testing.assert_close(
        result.fused_physical.sum(dim=1),
        result.base_physical.sum(dim=1),
    )
    x_sigma = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    y_sigma = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    torch.testing.assert_close(
        result.fused_response[:, 0],
        x_sigma * result.fused_physical[:, 0],
    )
    torch.testing.assert_close(
        result.fused_response[:, 1],
        y_sigma * result.fused_physical[:, 1],
    )


def test_pre_projection_fusion_preserves_zero_source_scale_exactly(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module()
    with torch.no_grad():
        module.linear_correction.weight.fill_(2.0)
        final_layer = module.nonlinear_correction[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.weight.fill_(1.0)
        assert final_layer.bias is not None
        final_layer.bias.fill_(3.0)

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=torch.zeros_like(x_amplitude),
        y_source_amplitude=torch.zeros_like(y_amplitude),
    )

    torch.testing.assert_close(
        result.source_scale,
        torch.zeros_like(result.source_scale),
    )
    torch.testing.assert_close(
        result.blended_correction,
        torch.zeros_like(result.blended_correction),
    )
    torch.testing.assert_close(result.fused_response, base_response)
    assert torch.isfinite(result.fused_response).all()


def test_pre_projection_fusion_geometry_feature_contract(tmp_path):
    geometry, *_rest = _inputs(tmp_path)
    module = _module()

    features = module._geometry_features(
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    x_length = geometry.x_lengths_for_valid_points()
    y_length = geometry.y_lengths_for_valid_points()
    reference = torch.tensor(1.0, dtype=torch.float64)
    expected_kappa = (
        4.0
        * x_length.square()
        * y_length.square()
        / (x_length.square() + y_length.square()).square()
    )
    assert features.shape == (geometry.num_points, 6)
    torch.testing.assert_close(features[:, 0], geometry.x_local_t)
    torch.testing.assert_close(features[:, 1], geometry.y_local_t)
    torch.testing.assert_close(features[:, 2], torch.log(x_length / reference))
    torch.testing.assert_close(features[:, 3], torch.log(y_length / reference))
    torch.testing.assert_close(features[:, 4], torch.log(x_length / y_length))
    torch.testing.assert_close(features[:, 5], expected_kappa)


def test_absolute_linear_candidate_is_base_difference_for_positive_scale(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.5,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.0,
    )

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    torch.testing.assert_close(
        module.linear_correction.weight,
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(result.linear_component, result.base_difference)
    torch.testing.assert_close(result.combined_component, result.linear_component)
    torch.testing.assert_close(result.fused_difference, result.base_difference)


def test_absolute_linear_candidate_is_zero_for_zero_source_scale(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.5,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.0,
    )

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=torch.zeros_like(x_amplitude),
        y_source_amplitude=torch.zeros_like(y_amplitude),
    )

    torch.testing.assert_close(
        result.linear_component,
        torch.zeros_like(result.linear_component),
    )
    torch.testing.assert_close(
        result.fused_difference,
        torch.zeros_like(result.fused_difference),
    )
    assert torch.isfinite(result.fused_response).all()


def test_absolute_linear_plus_nonlinear_has_no_outer_base_residual(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.25,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.0,
    )
    with torch.no_grad():
        module.linear_correction.weight.copy_(
            torch.tensor([[0.5, -0.25]], dtype=torch.float64)
        )
        final_layer = module.nonlinear_correction[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.weight.zero_()
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.8)

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    expected = result.linear_component + result.gate * result.nonlinear_component
    torch.testing.assert_close(result.combined_component, expected)
    torch.testing.assert_close(result.fused_difference, expected)
    assert not torch.allclose(
        result.fused_difference,
        result.base_difference + expected,
    )
    torch.testing.assert_close(
        result.fused_physical.sum(dim=1),
        result.base_physical.sum(dim=1),
    )


def test_absolute_convex_average_uses_two_absolute_candidates(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.25,
        mode="absolute_difference",
        combination="convex_average",
        nonlinear_final_init_scale=0.0,
    )
    with torch.no_grad():
        module.linear_correction.weight.copy_(
            torch.tensor([[0.25, 0.5]], dtype=torch.float64)
        )
        final_layer = module.nonlinear_correction[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.weight.zero_()
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.75)

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    expected = (
        1.0 - result.gate
    ) * result.linear_component + result.gate * result.nonlinear_component
    torch.testing.assert_close(result.combined_component, expected)
    torch.testing.assert_close(result.fused_difference, expected)


@pytest.mark.parametrize("scale", [0.0, 0.01, 1.0])
def test_absolute_nonlinear_final_initialization_scale(scale):
    torch.manual_seed(1729)
    standard = _module(
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=1.0,
    )
    torch.manual_seed(1729)
    scaled = _module(
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=scale,
    )

    standard_final = standard.nonlinear_correction[-1]
    scaled_final = scaled.nonlinear_correction[-1]
    assert isinstance(standard_final, torch.nn.Linear)
    assert isinstance(scaled_final, torch.nn.Linear)
    torch.testing.assert_close(
        scaled_final.weight,
        scale * standard_final.weight,
    )
    assert standard_final.bias is not None
    assert scaled_final.bias is not None
    torch.testing.assert_close(
        scaled_final.bias,
        scale * standard_final.bias,
    )


def test_absolute_small_final_initialization_reaches_hidden_layer_on_first_backward(
    tmp_path,
):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    torch.manual_seed(7)
    module = _module(
        gate=0.5,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.01,
    )

    fused = module(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )
    fused.square().mean().backward()

    first_layer = module.nonlinear_correction[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.weight.grad is not None
    assert torch.isfinite(first_layer.weight.grad).all()
    assert torch.count_nonzero(first_layer.weight.grad) > 0


def test_absolute_standard_final_initialization_with_small_gate_is_finite(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.05,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=1.0,
    )

    result = module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )

    assert float(result.gate.item()) == pytest.approx(0.05)
    assert torch.isfinite(result.linear_component).all()
    assert torch.isfinite(result.nonlinear_component).all()
    assert torch.isfinite(result.fused_response).all()


def test_absolute_fusion_preserves_projection_balance_and_output_shape(tmp_path):
    geometry, base_response, rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module(
        gate=0.5,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.01,
    )

    fused = module(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )
    projection = apply_complex_balance_projection(
        fused,
        rhs,
        geometry,
        BalanceProjectionConfig(enabled=True, mode="physical_symmetric"),
    )

    assert fused.shape == base_response.shape
    torch.testing.assert_close(
        projection.projected_physical.sum(dim=1),
        rhs,
    )


def test_residual_and_absolute_modes_keep_checkpoint_tensor_surface():
    residual = _module()
    absolute = _module(
        gate=0.5,
        mode="absolute_difference",
        combination="linear_plus_nonlinear",
        nonlinear_final_init_scale=0.01,
    )

    residual_state = residual.state_dict()
    absolute_state = absolute.state_dict()
    assert residual_state.keys() == absolute_state.keys()
    for key in residual_state:
        assert residual_state[key].shape == absolute_state[key].shape


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nonlinear_hidden_dim": 0}, "nonlinear_hidden_dim"),
        ({"nonlinear_depth": 0}, "nonlinear_depth"),
        ({"mode": "unsupported"}, "mode"),
        ({"combination": "unsupported"}, "combination"),
        (
            {
                "mode": "residual_correction",
                "combination": "linear_plus_nonlinear",
            },
            "requires.*convex_average",
        ),
        ({"nonlinear_final_init_scale": -1.0}, "nonlinear_final_init_scale"),
        ({"nonlinear_final_init_scale": math.inf}, "nonlinear_final_init_scale"),
        ({"gate_initial_value": 0.0}, "gate_initial_value"),
        ({"gate_initial_value": 1.0}, "gate_initial_value"),
        ({"gate_initial_value": math.inf}, "gate_initial_value"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_pre_projection_fusion_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ComplexPreProjectionFusionConfig(**kwargs)


def test_pre_projection_fusion_config_rejects_unknown_keys():
    with pytest.raises(TypeError, match="unknown keys"):
        ComplexPreProjectionFusionConfig.from_raw({"enabled": True, "unknown": 1})
