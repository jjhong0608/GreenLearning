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
    hidden_dim: int = 8,
    depth: int = 1,
    use_bias: bool = True,
) -> ComplexPreProjectionFusion:
    return ComplexPreProjectionFusion(
        ComplexPreProjectionFusionConfig(
            enabled=True,
            hidden_dim=hidden_dim,
            depth=depth,
        ),
        activation="tanh",
        use_bias=use_bias,
        dtype=torch.float64,
    )


def _diagnostics(module, inputs):
    geometry, base_response, rhs, x_amplitude, y_amplitude = inputs
    return module.forward_with_diagnostics(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )


def test_pre_projection_fusion_has_single_two_input_residual_mlp():
    module = _module(hidden_dim=8, depth=2)

    linear_layers = [
        layer for layer in module.residual_mlp if isinstance(layer, torch.nn.Linear)
    ]

    assert module.INPUT_FEATURE_NAMES == (
        "normalized_difference",
        "normalized_rhs",
    )
    assert [layer.in_features for layer in linear_layers] == [2, 8, 8]
    assert [layer.out_features for layer in linear_layers] == [8, 8, 1]
    assert not hasattr(module, "linear_correction")
    assert not hasattr(module, "nonlinear_correction")
    assert not hasattr(module, "gate_logit")
    assert not hasattr(module, "_geometry_features")


def test_pre_projection_fusion_zero_initializes_output_layer():
    module = _module()
    final_layer = module.residual_mlp[-1]

    assert isinstance(final_layer, torch.nn.Linear)
    torch.testing.assert_close(
        final_layer.weight,
        torch.zeros_like(final_layer.weight),
    )
    assert final_layer.bias is not None
    torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))


def test_pre_projection_fusion_is_projected_identity_at_initialization(tmp_path):
    inputs = _inputs(tmp_path)
    geometry, base_response, rhs, *_ = inputs
    result = _diagnostics(_module(), inputs)
    projection_config = BalanceProjectionConfig(
        enabled=True,
        mode="physical_symmetric",
    )
    base_projection = apply_complex_balance_projection(
        base_response,
        rhs,
        geometry,
        projection_config,
    )
    fused_projection = apply_complex_balance_projection(
        result.fused_response,
        rhs,
        geometry,
        projection_config,
    )

    torch.testing.assert_close(
        result.normalized_residual,
        torch.zeros_like(result.normalized_residual),
    )
    torch.testing.assert_close(
        result.physical_residual,
        torch.zeros_like(result.physical_residual),
    )
    torch.testing.assert_close(result.fused_difference, result.base_difference)
    torch.testing.assert_close(
        fused_projection.projected_response,
        base_projection.projected_response,
    )
    torch.testing.assert_close(
        fused_projection.projected_physical,
        base_projection.projected_physical,
    )


def test_pre_projection_fusion_uses_normalized_physical_inputs(tmp_path):
    inputs = _inputs(tmp_path)
    result = _diagnostics(_module(), inputs)
    geometry, _base_response, rhs, x_amplitude, y_amplitude = inputs
    x_amplitude_valid = x_amplitude[:, geometry.x_segment_id]
    y_amplitude_valid = y_amplitude[:, geometry.y_segment_id]
    expected_scale = torch.sqrt(
        0.5 * (x_amplitude_valid.square() + y_amplitude_valid.square())
    )

    torch.testing.assert_close(result.source_scale, expected_scale)
    torch.testing.assert_close(
        result.normalized_difference,
        result.base_difference / result.safe_source_scale,
    )
    torch.testing.assert_close(
        result.normalized_rhs,
        rhs / result.safe_source_scale,
    )


def test_pre_projection_fusion_applies_fixed_identity_skip(tmp_path):
    inputs = _inputs(tmp_path)
    module = _module()
    final_layer = module.residual_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        final_layer.weight.zero_()
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.25)

    result = _diagnostics(module, inputs)
    expected_residual = 0.25 * result.safe_source_scale

    torch.testing.assert_close(
        result.normalized_residual,
        torch.full_like(result.normalized_residual, 0.25),
    )
    torch.testing.assert_close(result.physical_residual, expected_residual)
    torch.testing.assert_close(
        result.fused_difference,
        result.base_difference + expected_residual,
    )


def test_pre_projection_fusion_constructs_balanced_physical_pair(tmp_path):
    inputs = _inputs(tmp_path)
    geometry, _base_response, rhs, *_ = inputs
    module = _module()
    final_layer = module.residual_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.5)

    result = _diagnostics(module, inputs)
    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)

    torch.testing.assert_close(result.fused_physical.sum(dim=1), rhs)
    torch.testing.assert_close(
        result.fused_physical[:, 0] - result.fused_physical[:, 1],
        result.fused_difference,
    )
    torch.testing.assert_close(
        result.pre_projection_balance_residual,
        torch.zeros_like(rhs),
    )
    torch.testing.assert_close(
        result.fused_response[:, 0],
        sigma_x * result.fused_physical[:, 0],
    )
    torch.testing.assert_close(
        result.fused_response[:, 1],
        sigma_y * result.fused_physical[:, 1],
    )


def test_pre_projection_fusion_preserves_zero_source_homogeneity(tmp_path):
    geometry, base_response, _rhs, x_amplitude, y_amplitude = _inputs(tmp_path)
    module = _module()
    final_layer = module.residual_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        final_layer.weight.fill_(1.0)
        assert final_layer.bias is not None
        final_layer.bias.fill_(3.0)

    zeros = torch.zeros_like(base_response[:, 0])
    result = module.forward_with_diagnostics(
        base_response=torch.zeros_like(base_response),
        rhs_phys=zeros,
        geometry=geometry,
        x_source_amplitude=torch.zeros_like(x_amplitude),
        y_source_amplitude=torch.zeros_like(y_amplitude),
    )

    torch.testing.assert_close(result.source_scale, torch.zeros_like(zeros))
    torch.testing.assert_close(result.physical_residual, torch.zeros_like(zeros))
    torch.testing.assert_close(result.fused_difference, torch.zeros_like(zeros))
    torch.testing.assert_close(result.fused_response, torch.zeros_like(base_response))
    assert torch.isfinite(result.normalized_residual).all()
    assert torch.isfinite(result.fused_response).all()


def test_pre_projection_fusion_hidden_gradient_follows_zero_head_warm_start(tmp_path):
    inputs = _inputs(tmp_path)
    module = _module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    geometry, base_response, rhs, x_amplitude, y_amplitude = inputs

    first_output = module(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )
    first_output.square().mean().backward()
    first_layer = module.residual_mlp[0]
    final_layer = module.residual_mlp[-1]
    assert isinstance(first_layer, torch.nn.Linear)
    assert isinstance(final_layer, torch.nn.Linear)
    assert first_layer.weight.grad is not None
    assert torch.count_nonzero(first_layer.weight.grad) == 0
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_output = module(
        base_response=base_response,
        rhs_phys=rhs,
        geometry=geometry,
        x_source_amplitude=x_amplitude,
        y_source_amplitude=y_amplitude,
    )
    second_output.square().mean().backward()

    assert first_layer.weight.grad is not None
    assert torch.count_nonzero(first_layer.weight.grad) > 0
    assert torch.isfinite(first_layer.weight.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"hidden_dim": True}, "hidden_dim"),
        ({"depth": 0}, "depth"),
        ({"depth": True}, "depth"),
        ({"eps": 0.0}, "eps"),
        ({"eps": math.inf}, "eps"),
    ],
)
def test_pre_projection_fusion_config_rejects_invalid_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ComplexPreProjectionFusionConfig(**kwargs)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "mode",
        "combination",
        "nonlinear_hidden_dim",
        "nonlinear_depth",
        "nonlinear_final_init_scale",
        "gate_initial_value",
    ],
)
def test_pre_projection_fusion_config_rejects_legacy_keys(legacy_key):
    with pytest.raises(TypeError, match="unknown keys"):
        ComplexPreProjectionFusionConfig.from_raw(
            {
                "enabled": True,
                legacy_key: 1,
            }
        )


def test_pre_projection_fusion_config_rejects_unknown_keys():
    with pytest.raises(TypeError, match="unknown keys"):
        ComplexPreProjectionFusionConfig.from_raw({"enabled": True, "unknown": 1})
