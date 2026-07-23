from __future__ import annotations

import math

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_pre_projection_fusion import ComplexPreProjectionFusion
from greenonet.config import ComplexPreProjectionFusionConfig
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
) -> ComplexPreProjectionFusion:
    return ComplexPreProjectionFusion(
        ComplexPreProjectionFusionConfig(
            enabled=True,
            nonlinear_hidden_dim=hidden_dim,
            nonlinear_depth=depth,
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nonlinear_hidden_dim": 0}, "nonlinear_hidden_dim"),
        ({"nonlinear_depth": 0}, "nonlinear_depth"),
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
