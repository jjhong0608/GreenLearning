from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.coupling_model import CouplingNet, FiveStencilSourceLift
from greenonet.coupling_trainer import CouplingTrainer
from greenonet.config import (
    CouplingBestRelSolCheckpointConfig,
    CouplingLossTermConfig,
    CouplingLossesConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
    SourceStencilLiftConfig,
)
from greenonet.numerics import line_operator_fd


def _make_npz(tmp_path, name: str = "sample.npz"):
    grid = np.linspace(0.0, 1.0, 257)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    sol = xx + yy
    rhs = xx * 0 + 1.0
    uxx = np.ones((255, 255))
    uyy = np.ones((255, 255))
    path = tmp_path / name
    np.savez(path, sol=sol, rhs=rhs, uxx=uxx, uyy=uyy)
    return path


class _DummyFluxModel(nn.Module):
    def __init__(self, projected_flux: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_projected_flux", projected_flux)

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
        return self._projected_flux.clone()


class _InspectModeFluxModel(nn.Module):
    def __init__(self, projected_flux: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_projected_flux", projected_flux)
        self.training_states: list[bool] = []
        self.grad_enabled_states: list[bool] = []

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
        self.training_states.append(self.training)
        self.grad_enabled_states.append(torch.is_grad_enabled())
        return self._projected_flux.clone()


class _TrainableFluxModel(nn.Module):
    def __init__(self, projected_flux: torch.Tensor) -> None:
        super().__init__()
        self.projected_flux = nn.Parameter(projected_flux.clone())

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
        return self.projected_flux


class _RawSumFromRhsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, b_vals, c_vals, rhs_tilde, rhs_norm
        projected_flux = torch.zeros_like(rhs_raw)
        rhs_inner = rhs_raw[:, 0, :, 1:-1]
        projected_flux[:, 0, :, 1:-1] = 0.25 * rhs_inner + 0.0 * self.anchor
        projected_flux[:, 1, :, 1:-1] = (0.75 * rhs_inner).transpose(
            -1, -2
        ) + 0.0 * self.anchor
        return projected_flux


def _flux_from_q_common(
    q: torch.Tensor, common: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Build balanced flux tensors from interior gauge/common parts.

    q: (B, n, n), common: (B, n, n)
    returns flux: (B, 2, n, n+2) with zero boundaries.
    """
    if common is None:
        common = torch.zeros_like(q)
    if q.shape != common.shape:
        raise ValueError("q and common must have identical shapes")
    bsz, n_lines, n_inner = q.shape
    if n_lines != n_inner:
        raise ValueError("Expected square interior grid: q.shape[-2] == q.shape[-1]")
    m_points = n_lines + 2
    flux = torch.zeros((bsz, 2, n_lines, m_points), dtype=q.dtype)
    phi_int = common + q
    psi_int_t = common - q
    flux[:, 0, :, 1:-1] = phi_int
    flux[:, 1, :, 1:-1] = psi_int_t.transpose(-1, -2)
    return flux


def _make_zero_dirichlet_grid(m_points: int) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, m_points, dtype=torch.float64)
    x = axis.view(1, -1)
    y = axis.view(-1, 1)
    return (x * (1.0 - x) * y * (1.0 - y)).unsqueeze(0)


def _interior_second_difference(lines: torch.Tensor, spacing: float) -> torch.Tensor:
    return -(lines[..., 2:] - 2.0 * lines[..., 1:-1] + lines[..., :-2]) / (spacing**2)


def _sequential_integrator(outputs: list[torch.Tensor]):
    def fake_integrate(
        green: torch.Tensor, values: torch.Tensor, axis: torch.Tensor
    ) -> torch.Tensor:
        del green, values, axis
        return outputs.pop(0)

    return fake_integrate


def _make_step_loss_inputs(
    batch: int = 1, n_lines: int = 1, m_points: int = 5
) -> tuple[torch.Tensor, ...]:
    x_axis = torch.linspace(0.0, 1.0, m_points, dtype=torch.float64)
    y_axis = torch.linspace(0.0, 1.0, m_points, dtype=torch.float64)

    coords = torch.zeros((2, n_lines, m_points, 2), dtype=torch.float64)
    coords[0, :, :, 0] = x_axis.unsqueeze(0).expand(n_lines, -1)
    coords[0, :, :, 1] = 0.5
    coords[1, :, :, 0] = 0.5
    coords[1, :, :, 1] = y_axis.unsqueeze(0).expand(n_lines, -1)

    rhs_raw = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    rhs_tilde = rhs_raw.clone()
    rhs_norm = torch.ones((batch, 2, n_lines), dtype=torch.float64)
    sol = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    flux_target = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    a_vals = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    b_vals = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    c_vals = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
    return (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )


def _make_coupling_dataset_item(
    *,
    n_lines: int = 3,
    m_points: int = 5,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    inputs = _make_step_loss_inputs(batch=1, n_lines=n_lines, m_points=m_points)
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = inputs
    a_vals[:] = 1.0
    item = (
        coords,
        rhs_raw[0],
        rhs_tilde[0],
        rhs_norm[0],
        sol[0],
        flux_target[0],
        a_vals[0],
        b_vals[0],
        c_vals[0],
        torch.zeros_like(a_vals[0]),
    )
    return inputs, item


def _losses_config(
    *,
    l2_enabled: bool = True,
    l2_weight: float = 1.0,
    energy_enabled: bool = True,
    energy_weight: float = 1.0,
    cross_enabled: bool = True,
    cross_weight: float = 1.0,
) -> CouplingLossesConfig:
    return CouplingLossesConfig(
        l2_consistency=CouplingLossTermConfig(
            enabled=l2_enabled,
            weight=l2_weight,
        ),
        energy_consistency=CouplingLossTermConfig(
            enabled=energy_enabled,
            weight=energy_weight,
        ),
        cross_consistency=CouplingLossTermConfig(
            enabled=cross_enabled,
            weight=cross_weight,
        ),
    )


def _axis_field_from_canonical_full(full_field: torch.Tensor) -> torch.Tensor:
    bsz, m_points, m_points_2 = full_field.shape
    if m_points != m_points_2:
        raise ValueError("full_field must be square.")
    n_lines = m_points - 2
    values = torch.zeros((bsz, 2, n_lines, m_points), dtype=full_field.dtype)
    values[:, 0] = full_field[:, 1:-1, :]
    values[:, 1] = full_field[:, :, 1:-1].transpose(-1, -2)
    return values


def _rhs_raw_from_canonical_full(full_source: torch.Tensor) -> torch.Tensor:
    return _axis_field_from_canonical_full(full_source)


def _make_linear_source_lift(
    *, use_g_normalization: bool = False
) -> FiveStencilSourceLift:
    lift = FiveStencilSourceLift(
        SourceStencilLiftConfig(
            enabled=True,
            encoder_type="linear",
            hidden_dim=1,
            depth=1,
            activation="relu",
            use_bias=False,
            use_g_normalization=use_g_normalization,
        )
    ).to(dtype=torch.float64)
    source = lift.source_encoder[0]
    coefficient = lift.coefficient_encoder[0]
    assert len(lift.source_encoder) == 1
    assert len(lift.coefficient_encoder) == 1
    assert isinstance(source, nn.Linear)
    assert isinstance(coefficient, nn.Linear)
    with torch.no_grad():
        source.weight.fill_(1.0)
        coefficient.weight.fill_(1.0)
    return lift


def _make_optimizer_test_coupling_net(*, source_lift_enabled: bool) -> CouplingNet:
    return CouplingNet(
        CouplingModelConfig(
            branch_input_dim=5,
            trunk_input_dim=2,
            hidden_dim=8,
            depth=1,
            dtype=torch.float64,
            source_stencil_lift=SourceStencilLiftConfig(
                enabled=source_lift_enabled,
                hidden_dim=4,
            ),
        )
    )


def test_source_stencil_lift_default_encoder_is_mlp():
    cfg = SourceStencilLiftConfig(enabled=True)
    lift = FiveStencilSourceLift(cfg)

    source_linear_layers = [
        module for module in lift.source_encoder if isinstance(module, nn.Linear)
    ]
    coefficient_linear_layers = [
        module for module in lift.coefficient_encoder if isinstance(module, nn.Linear)
    ]

    assert lift.encoder_type == "mlp"
    assert lift.coefficient_normalization == "rms"
    assert lift.coefficient_tanh_beta == 1.0
    assert len(source_linear_layers) == cfg.depth + 1
    assert len(coefficient_linear_layers) == cfg.depth + 1
    assert source_linear_layers[0].in_features == lift.stencil_features
    assert coefficient_linear_layers[0].in_features == lift.stencil_features
    assert source_linear_layers[0].out_features == cfg.hidden_dim
    assert coefficient_linear_layers[0].out_features == cfg.hidden_dim
    assert source_linear_layers[-1].in_features == cfg.hidden_dim
    assert coefficient_linear_layers[-1].in_features == cfg.hidden_dim
    assert source_linear_layers[-1].out_features == 1
    assert coefficient_linear_layers[-1].out_features == 1
    assert lift.source_encoder is not lift.coefficient_encoder


def test_source_stencil_lift_mlp_encoder_accepts_uppercase_config():
    lift = FiveStencilSourceLift(
        SourceStencilLiftConfig(enabled=True, encoder_type="MLP")
    )

    assert lift.encoder_type == "mlp"


def test_source_stencil_lift_linear_encoder_has_single_affine_layer():
    torch.set_default_dtype(torch.float64)
    lift = FiveStencilSourceLift(
        SourceStencilLiftConfig(
            enabled=True,
            encoder_type="linear",
            hidden_dim=7,
            depth=3,
            use_bias=False,
        )
    ).to(dtype=torch.float64)
    source = lift.source_encoder[0]
    coefficient = lift.coefficient_encoder[0]

    assert lift.encoder_type == "linear"
    assert len(lift.source_encoder) == 1
    assert len(lift.coefficient_encoder) == 1
    assert isinstance(source, nn.Linear)
    assert isinstance(coefficient, nn.Linear)
    assert source.in_features == lift.stencil_features
    assert coefficient.in_features == lift.stencil_features
    assert source.out_features == 1
    assert coefficient.out_features == 1
    assert source.bias is None
    assert coefficient.bias is None
    assert source is not coefficient

    full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    lifted = lift(
        _rhs_raw_from_canonical_full(full),
        _axis_field_from_canonical_full(full + 2.0),
    )

    assert lifted.shape == (1, 2, 3, 3)


def test_source_stencil_lift_rejects_invalid_encoder_type():
    with pytest.raises(ValueError, match="encoder_type"):
        FiveStencilSourceLift(SourceStencilLiftConfig(enabled=True, encoder_type="cnn"))


def test_source_stencil_lift_rejects_invalid_coefficient_normalization():
    with pytest.raises(ValueError, match="coefficient_normalization"):
        FiveStencilSourceLift(
            SourceStencilLiftConfig(
                enabled=True,
                coefficient_normalization="softmax",  # type: ignore[arg-type]
            )
        )


def test_source_stencil_lift_rejects_non_positive_coefficient_tanh_beta():
    with pytest.raises(ValueError, match="coefficient_tanh_beta"):
        FiveStencilSourceLift(
            SourceStencilLiftConfig(
                enabled=True,
                coefficient_normalization="tanh",
                coefficient_tanh_beta=0.0,
            )
        )


def test_source_stencil_lift_shape_and_g_normalization():
    torch.set_default_dtype(torch.float64)
    full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    rhs_raw = _rhs_raw_from_canonical_full(full)
    a_vals = _axis_field_from_canonical_full(2.0 + full)
    lift = _make_linear_source_lift(use_g_normalization=True)

    lifted = lift(rhs_raw, a_vals)
    lifted_source, lifted_coefficient = lift.forward_components(rhs_raw, a_vals)
    diagnostics = lift.lift_with_diagnostics(rhs_raw, a_vals)[1]

    assert lifted.shape == (1, 2, 3, 3)
    assert lifted_source.shape == (1, 2, 3, 3)
    assert lifted_coefficient.shape == (1, 2, 3, 3)
    torch.testing.assert_close(lifted[:, 1], lifted[:, 0].transpose(-1, -2))
    torch.testing.assert_close(
        lifted[:, 0],
        lifted_source[:, 0] * lifted_coefficient[:, 0],
    )
    assert torch.sqrt(torch.mean(lifted_source[:, 0].square())).item() == pytest.approx(
        1.0
    )
    assert torch.sqrt(
        torch.mean(lifted_coefficient[:, 0].square())
    ).item() == pytest.approx(1.0)
    assert torch.isfinite(diagnostics["source_lift_g_rms"]).item()


def test_source_stencil_lift_tanh_coefficient_normalization_uses_raw_encoder_output():
    torch.set_default_dtype(torch.float64)
    source_full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    a_full = torch.linspace(-1.0, 1.0, 25, dtype=torch.float64).reshape(1, 5, 5)
    beta = 2.5
    lift = FiveStencilSourceLift(
        SourceStencilLiftConfig(
            enabled=True,
            encoder_type="linear",
            use_bias=False,
            use_g_normalization=True,
            coefficient_normalization="tanh",
            coefficient_tanh_beta=beta,
        )
    ).to(dtype=torch.float64)
    source_layer = lift.source_encoder[0]
    coefficient_layer = lift.coefficient_encoder[0]
    assert isinstance(source_layer, nn.Linear)
    assert isinstance(coefficient_layer, nn.Linear)
    with torch.no_grad():
        source_layer.weight.fill_(1.0)
        coefficient_layer.weight.zero_()
        coefficient_layer.weight[0, 0] = 0.75

    rhs_raw = _rhs_raw_from_canonical_full(source_full)
    a_vals = _axis_field_from_canonical_full(a_full)
    lifted_source, lifted_coefficient = lift.forward_components(rhs_raw, a_vals)

    source_stencil, _f_hat = lift._build_source_stencil_features(rhs_raw)
    coefficient_stencil = lift._build_coefficient_stencil_features(
        a_vals,
        expected_shape=lift._validate_rhs_raw(rhs_raw),
    )
    source_raw = torch.sum(source_stencil, dim=-1)
    expected_source = source_raw / torch.sqrt(
        torch.mean(source_raw.square(), dim=(-1, -2), keepdim=True) + lift.eps
    )
    r_coef = 0.75 * coefficient_stencil[..., 0]
    expected_coefficient = beta * torch.tanh(r_coef)

    torch.testing.assert_close(lifted_source[:, 0], expected_source)
    torch.testing.assert_close(lifted_coefficient[:, 0], expected_coefficient)
    torch.testing.assert_close(
        lifted_coefficient[:, 1],
        expected_coefficient.transpose(-1, -2),
    )
    assert torch.sqrt(torch.mean(lifted_source[:, 0].square())).item() == pytest.approx(
        1.0
    )
    assert torch.sqrt(torch.mean(lifted_coefficient[:, 0].square())).item() != (
        pytest.approx(1.0)
    )


def test_source_stencil_lift_feature_contract_uses_normalized_f_and_raw_coefficient():
    torch.set_default_dtype(torch.float64)
    source_full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    a_full = torch.arange(101, 126, dtype=torch.float64).reshape(1, 5, 5)
    lift = _make_linear_source_lift(use_g_normalization=False)
    rhs_raw = _rhs_raw_from_canonical_full(source_full)
    a_vals = _axis_field_from_canonical_full(a_full)

    source_stencil, f_hat = lift._build_source_stencil_features(rhs_raw)
    coefficient_stencil = lift._build_coefficient_stencil_features(
        a_vals,
        expected_shape=lift._validate_rhs_raw(rhs_raw),
    )
    features, combined_f_hat = lift._build_stencil_features(rhs_raw, a_vals)

    f_int = source_full[:, 1:-1, 1:-1]
    scale = torch.sqrt(torch.mean(f_int * f_int, dim=(-1, -2), keepdim=True) + lift.eps)
    expected_f_stencil = lift._gather_5_stencil(source_full / scale)
    expected_a_stencil = lift._gather_5_stencil(a_full)
    expected_features = torch.cat((expected_f_stencil, expected_a_stencil), dim=-1)

    assert source_stencil.shape == (1, 3, 3, 5)
    assert coefficient_stencil.shape == (1, 3, 3, 5)
    torch.testing.assert_close(source_stencil, expected_f_stencil)
    torch.testing.assert_close(coefficient_stencil, expected_a_stencil)
    torch.testing.assert_close(features, expected_features)
    torch.testing.assert_close(f_hat, f_int / scale)
    torch.testing.assert_close(combined_f_hat, f_hat)


def test_source_stencil_lift_multiplies_source_and_coefficient_encodings():
    torch.set_default_dtype(torch.float64)
    source_full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    a_full = torch.arange(101, 126, dtype=torch.float64).reshape(1, 5, 5)
    lift = FiveStencilSourceLift(
        SourceStencilLiftConfig(
            enabled=True,
            encoder_type="linear",
            use_bias=False,
            use_g_normalization=False,
        )
    ).to(dtype=torch.float64)
    source_layer = lift.source_encoder[0]
    coefficient_layer = lift.coefficient_encoder[0]
    assert isinstance(source_layer, nn.Linear)
    assert isinstance(coefficient_layer, nn.Linear)
    with torch.no_grad():
        source_layer.weight.zero_()
        source_layer.weight[0, 0] = 1.0
        coefficient_layer.weight.zero_()
        coefficient_layer.weight[0, 0] = 2.0

    lifted = lift(
        _rhs_raw_from_canonical_full(source_full),
        _axis_field_from_canonical_full(a_full),
    )
    lifted_source, lifted_coefficient = lift.forward_components(
        _rhs_raw_from_canonical_full(source_full),
        _axis_field_from_canonical_full(a_full),
    )

    source_stencil, _f_hat = lift._build_source_stencil_features(
        _rhs_raw_from_canonical_full(source_full)
    )
    coefficient_stencil = lift._build_coefficient_stencil_features(
        _axis_field_from_canonical_full(a_full),
        expected_shape=lift._validate_rhs_raw(
            _rhs_raw_from_canonical_full(source_full)
        ),
    )
    expected = source_stencil[..., 0] * (2.0 * coefficient_stencil[..., 0])
    torch.testing.assert_close(lifted_source[:, 0], source_stencil[..., 0])
    torch.testing.assert_close(
        lifted_coefficient[:, 0],
        2.0 * coefficient_stencil[..., 0],
    )
    torch.testing.assert_close(lifted[:, 0], expected)
    torch.testing.assert_close(lifted[:, 1], expected.transpose(-1, -2))


def test_source_stencil_lift_uses_boundaries_but_ignores_corners():
    torch.set_default_dtype(torch.float64)
    full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    a_full = full + 10.0
    lift = _make_linear_source_lift(use_g_normalization=False)

    corner_changed = full.clone()
    corner_changed[:, 0, 0] = 1.0e6
    corner_changed[:, 0, -1] = -1.0e6
    corner_changed[:, -1, 0] = 2.0e6
    corner_changed[:, -1, -1] = -2.0e6

    boundary_changed = full.clone()
    boundary_changed[:, 1, 0] = boundary_changed[:, 1, 0] + 50.0

    a_corner_changed = a_full.clone()
    a_corner_changed[:, 0, 0] = 1.0e6
    a_boundary_changed = a_full.clone()
    a_boundary_changed[:, 1, 0] = a_boundary_changed[:, 1, 0] + 50.0

    a_vals = _axis_field_from_canonical_full(a_full)
    base = lift(_rhs_raw_from_canonical_full(full), a_vals)
    from_corners = lift(_rhs_raw_from_canonical_full(corner_changed), a_vals)
    from_boundary = lift(_rhs_raw_from_canonical_full(boundary_changed), a_vals)
    from_a_corners = lift(
        _rhs_raw_from_canonical_full(full),
        _axis_field_from_canonical_full(a_corner_changed),
    )
    from_a_boundary = lift(
        _rhs_raw_from_canonical_full(full),
        _axis_field_from_canonical_full(a_boundary_changed),
    )

    torch.testing.assert_close(base, from_corners)
    torch.testing.assert_close(base, from_a_corners)
    assert not torch.allclose(base[:, 0, 0, 0], from_boundary[:, 0, 0, 0])
    assert not torch.allclose(base[:, 0, 0, 0], from_a_boundary[:, 0, 0, 0])
    torch.testing.assert_close(base[:, 0, :, 1:], from_boundary[:, 0, :, 1:])
    torch.testing.assert_close(base[:, 0, :, 1:], from_a_boundary[:, 0, :, 1:])


def test_source_stencil_lift_uses_raw_a_channel_without_normalization():
    torch.set_default_dtype(torch.float64)
    source_full = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) + 1.0
    a_full = torch.ones((1, 5, 5), dtype=torch.float64) * 2.0
    lift = _make_linear_source_lift(use_g_normalization=False)
    rhs_raw = _rhs_raw_from_canonical_full(source_full)

    base = lift(rhs_raw, _axis_field_from_canonical_full(a_full))
    changed = lift(rhs_raw, _axis_field_from_canonical_full(a_full + 3.0))
    scaled = lift(rhs_raw, _axis_field_from_canonical_full(4.0 * a_full))

    assert not torch.allclose(base, changed)
    assert not torch.allclose(base, scaled)


def test_coupling_net_source_branch_input_dimension_changes_only_when_enabled():
    disabled = CouplingNet(CouplingModelConfig(branch_input_dim=5, hidden_dim=8))
    enabled = CouplingNet(
        CouplingModelConfig(
            branch_input_dim=5,
            hidden_dim=8,
            source_stencil_lift=SourceStencilLiftConfig(enabled=True),
        )
    )
    disabled_rhs_first = disabled.branch_rhs.net[0]
    enabled_rhs_first = enabled.branch_rhs.net[0]
    enabled_a_first = enabled.branch_a.net[0]
    assert enabled.branch_coefficient is not None
    enabled_coefficient_branch_first = enabled.branch_coefficient.net[0]
    enabled_lift_source_first = enabled.source_stencil_lift.source_encoder[0]
    enabled_lift_coefficient_first = enabled.source_stencil_lift.coefficient_encoder[0]

    assert isinstance(disabled_rhs_first, nn.Linear)
    assert isinstance(enabled_rhs_first, nn.Linear)
    assert isinstance(enabled_a_first, nn.Linear)
    assert isinstance(enabled_coefficient_branch_first, nn.Linear)
    assert enabled.source_stencil_lift is not None
    assert isinstance(enabled_lift_source_first, nn.Linear)
    assert isinstance(enabled_lift_coefficient_first, nn.Linear)
    assert disabled_rhs_first.in_features == 5
    assert enabled_rhs_first.in_features == 3
    assert enabled_a_first.in_features == 5
    assert enabled_coefficient_branch_first.in_features == 3
    assert enabled_lift_source_first.in_features == 5
    assert enabled_lift_coefficient_first.in_features == 5


def test_coupling_net_source_lift_rejects_too_short_branch_input():
    with pytest.raises(ValueError, match="branch_input_dim > 2"):
        CouplingNet(
            CouplingModelConfig(
                branch_input_dim=2,
                source_stencil_lift=SourceStencilLiftConfig(enabled=True),
            )
        )


def test_coupling_net_source_lift_keeps_physical_rhs_for_projection():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        source_stencil_lift=SourceStencilLiftConfig(enabled=True, hidden_dim=4),
    )
    model = CouplingNet(cfg)
    coords = torch.randn(2, n, m, 2)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    rhs_tilde = torch.randn(bsz, 2, n, m)
    rhs_norm = torch.rand(bsz, 2, n) + 0.1

    out = model(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )

    phi = out[:, 0, :, 1:-1]
    psi = out[:, 1, :, 1:-1].transpose(-1, -2)
    torch.testing.assert_close(phi + psi, rhs_raw[:, 0, :, 1:-1])


def test_coupling_net_enabled_source_lift_uses_separate_source_and_coefficient_branches():
    class _InspectSourceLift(nn.Module):
        def forward_components(
            self,
            rhs_raw: torch.Tensor,
            a_vals: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert rhs_raw.shape == a_vals.shape
            bsz, axis, n_lines, m_points = rhs_raw.shape
            source = torch.full(
                (bsz, axis, n_lines, m_points - 2),
                2.0,
                dtype=rhs_raw.dtype,
                device=rhs_raw.device,
            )
            coefficient = torch.full(
                (bsz, axis, n_lines, m_points - 2),
                3.0,
                dtype=rhs_raw.dtype,
                device=rhs_raw.device,
            )
            return source, coefficient

        def forward(self, rhs_raw: torch.Tensor, a_vals: torch.Tensor) -> torch.Tensor:
            raise AssertionError("CouplingNet should use forward_components")

    class _FailBranch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError(
                "branch_a should not be used when source lift is enabled"
            )

    class _ExpectBranch(nn.Module):
        def __init__(self, expected_value: float, output_value: float, hidden_dim: int):
            super().__init__()
            self.expected_value = expected_value
            self.output_value = output_value
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch.testing.assert_close(
                x,
                torch.full_like(x, self.expected_value),
            )
            return torch.full(
                (x.shape[0], self.hidden_dim),
                self.output_value,
                dtype=x.dtype,
                device=x.device,
            )

    class _OnesBranch(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                (x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device
            )

    torch.set_default_dtype(torch.float64)
    bsz = 1
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        source_stencil_lift=SourceStencilLiftConfig(enabled=True, hidden_dim=4),
    )
    model = CouplingNet(cfg)
    model.source_stencil_lift = _InspectSourceLift()  # type: ignore[assignment]
    model.branch_a = _FailBranch()
    model.branch_rhs = _ExpectBranch(2.0, 5.0, cfg.hidden_dim)
    model.branch_coefficient = _ExpectBranch(3.0, 7.0, cfg.hidden_dim)  # type: ignore[assignment]
    model.trunk = _OnesBranch(cfg.hidden_dim)

    coords = torch.zeros((2, n, m, 2), dtype=torch.float64)
    a_vals = torch.ones((bsz, 2, n, m), dtype=torch.float64)
    zeros = torch.zeros_like(a_vals)
    rhs_raw = torch.zeros_like(a_vals)
    rhs_norm = torch.ones((bsz, 2, n), dtype=torch.float64)

    out = model(
        coords=coords,
        a_vals=a_vals,
        b_vals=zeros,
        c_vals=zeros,
        rhs_raw=rhs_raw,
        rhs_tilde=zeros,
        rhs_norm=rhs_norm,
    )

    assert out.shape == (bsz, 2, n, m)


def test_coupling_dataset_shapes(tmp_path):
    _make_npz(tmp_path)
    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    coords, rhs_raw, rhs_tilde, rhs_norm, sol, flux, kappa, b_vals, c_vals, ap = ds[0]
    assert coords.shape[0] == 2
    assert rhs_raw.shape[0] == 2 and rhs_raw.shape[-1] == 3
    assert rhs_tilde.shape == rhs_raw.shape
    assert rhs_norm.shape == (2, rhs_raw.shape[1])
    assert sol.shape == rhs_raw.shape
    assert flux.shape == rhs_raw.shape
    assert kappa.shape == rhs_raw.shape
    assert b_vals.shape == rhs_raw.shape
    assert c_vals.shape == rhs_raw.shape
    assert ap.shape == rhs_raw.shape


def test_coupling_dataset_trapezoid_rhs_norm(tmp_path):
    grid = np.linspace(0.0, 1.0, 257)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    sol = xx + yy
    rhs = yy**2
    uxx = np.ones((255, 255))
    uyy = np.ones((255, 255))
    path = tmp_path / "sample_quad.npz"
    np.savez(path, sol=sol, rhs=rhs, uxx=uxx, uyy=uyy)

    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
        integration_rule="trapezoid",
    )

    _coords, rhs_raw, _rhs_tilde, rhs_norm, *_rest = ds[0]
    assert rhs_raw.shape == (2, 1, 3)
    assert rhs_norm[0, 0].item() == pytest.approx(np.sqrt(0.28125), rel=1e-6)


def test_coupling_model_forward():
    cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    coords = torch.zeros((2, 1, 3, 2))
    kappa = torch.randn(1, 2, 1, 3)
    b_vals = torch.randn(1, 2, 1, 3)
    c_vals = torch.randn(1, 2, 1, 3)
    rhs_raw = torch.randn(1, 2, 1, 3)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1, keepdim=False).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)
    projected_flux = model(
        coords=coords,
        a_vals=kappa,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )
    assert isinstance(projected_flux, torch.Tensor)
    assert projected_flux.shape == (1, 2, 1, 3)
    phi_x = projected_flux[:, 0]
    psi_y = projected_flux[:, 1]
    assert phi_x.shape == rhs_raw[:, 0].shape
    assert psi_y.shape == rhs_raw[:, 1].shape


def test_coupling_model_backprop_through_shared_encoder():
    cfg = CouplingModelConfig(
        branch_input_dim=3,
        hidden_dim=8,
        depth=2,
    )
    model = CouplingNet(cfg)

    coords = torch.zeros((2, 1, 3, 2), dtype=torch.float64)
    a_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    b_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    c_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_raw = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1, keepdim=False).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    projected_flux = model(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )
    loss = projected_flux.square().sum()
    loss.backward()

    def _has_grad(module: nn.Module) -> bool:
        return any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in module.parameters()
        )

    assert projected_flux.shape == (1, 2, 1, 3)
    assert _has_grad(model.branch_a)
    assert _has_grad(model.branch_rhs)
    assert _has_grad(model.trunk)


def test_coupling_model_enabled_source_lift_backprop_skips_branch_a():
    cfg = CouplingModelConfig(
        branch_input_dim=3,
        hidden_dim=8,
        depth=2,
        source_stencil_lift=SourceStencilLiftConfig(enabled=True, hidden_dim=4),
    )
    model = CouplingNet(cfg)

    coords = torch.zeros((2, 1, 3, 2), dtype=torch.float64)
    a_vals = torch.rand(1, 2, 1, 3, dtype=torch.float64) + 1.0
    b_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    c_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_raw = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1, keepdim=False).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    projected_flux = model(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )
    loss = projected_flux.square().sum()
    loss.backward()

    def _has_grad(module: nn.Module) -> bool:
        return any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in module.parameters()
        )

    assert projected_flux.shape == (1, 2, 1, 3)
    assert model.source_stencil_lift is not None
    assert model.branch_coefficient is not None
    assert _has_grad(model.source_stencil_lift)
    assert _has_grad(model.branch_rhs)
    assert _has_grad(model.branch_coefficient)
    assert _has_grad(model.trunk)
    assert not _has_grad(model.branch_a)


def _make_projection_coords(n_lines: int, dtype: torch.dtype = torch.float64):
    m_points = n_lines + 2
    axis = torch.linspace(0.0, 1.0, m_points, dtype=dtype)
    y_inner = axis[1:-1]
    coords = torch.zeros((2, n_lines, m_points, 2), dtype=dtype)
    coords[0, :, :, 0] = axis.unsqueeze(0).expand(n_lines, -1)
    coords[0, :, :, 1] = y_inner.view(-1, 1).expand(-1, m_points)
    coords[1, :, :, 0] = y_inner.view(-1, 1).expand(-1, m_points)
    coords[1, :, :, 1] = axis.unsqueeze(0).expand(n_lines, -1)
    return coords


def test_coupling_balance_projection_uses_fixed_half_split():
    class _ZeroBranch(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device
            )

    cfg = CouplingModelConfig(branch_input_dim=5, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    model.branch_a = _ZeroBranch(cfg.hidden_dim)
    model.branch_b = _ZeroBranch(cfg.hidden_dim)
    model.branch_c = _ZeroBranch(cfg.hidden_dim)
    model.branch_rhs = _ZeroBranch(cfg.hidden_dim)
    model.trunk = _ZeroBranch(cfg.hidden_dim)

    coords = torch.zeros((2, 3, 5, 2), dtype=torch.float64)
    zeros = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    rhs_raw = zeros.clone()
    rhs_x_int = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.5, 1.5, 2.5]]],
        dtype=torch.float64,
    )
    rhs_raw[:, 0, :, 1:-1] = rhs_x_int
    rhs_norm = torch.ones((1, 2, 3), dtype=torch.float64)

    projected_flux = model(
        coords=coords,
        a_vals=zeros,
        b_vals=zeros,
        c_vals=zeros,
        rhs_raw=rhs_raw,
        rhs_tilde=zeros,
        rhs_norm=rhs_norm,
    )

    expected_phi = 0.5 * rhs_x_int
    expected_psi = 0.5 * rhs_x_int.transpose(-1, -2)
    assert torch.allclose(
        projected_flux[:, 0, :, 1:-1], expected_phi, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        projected_flux[:, 1, :, 1:-1], expected_psi, atol=1e-12, rtol=1e-12
    )


def test_symmetric_balance_projection_matches_old_formula_directly():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(branch_input_dim=m, balance_projection="symmetric")
    model = CouplingNet(cfg)
    flux_int = torch.randn(2, 2, n, n, dtype=torch.float64)
    rhs_raw = torch.randn(2, 2, n, m, dtype=torch.float64)
    coords = _make_projection_coords(n)

    projected = model._apply_balance_projection(flux_int, rhs_raw, coords)

    phi0 = flux_int[:, 0]
    psi0 = flux_int[:, 1].transpose(-1, -2)
    f = rhs_raw[:, 0, :, 1:-1]
    residual = f - (phi0 + psi0)
    expected_phi = phi0 + 0.5 * residual
    expected_psi = psi0 + 0.5 * residual
    torch.testing.assert_close(projected[:, 0], expected_phi)
    torch.testing.assert_close(projected[:, 1].transpose(-1, -2), expected_psi)


def test_smooth_mask_balance_projection_preserves_balance_shape_dtype_and_device():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
    )
    model = CouplingNet(cfg)
    flux_int = torch.randn(2, 2, n, n, dtype=torch.float64)
    rhs_raw = torch.randn(2, 2, n, m, dtype=torch.float64)
    coords = _make_projection_coords(n)

    projected = model._apply_balance_projection(flux_int, rhs_raw, coords)

    assert projected.shape == flux_int.shape
    assert projected.dtype == flux_int.dtype
    assert projected.device == flux_int.device
    phi = projected[:, 0]
    psi = projected[:, 1].transpose(-1, -2)
    torch.testing.assert_close(
        phi + psi,
        rhs_raw[:, 0, :, 1:-1],
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_smooth_mask_default_powers_match_original_formula():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
        smooth_mask_power=1.0,
        smooth_mask_diff_power=1.0,
    )
    model = CouplingNet(cfg)
    flux_int = torch.randn(2, 2, n, n, dtype=torch.float64)
    rhs_raw = torch.randn(2, 2, n, m, dtype=torch.float64)
    coords = _make_projection_coords(n)

    projected = model._apply_balance_projection(flux_int, rhs_raw, coords)

    phi0 = flux_int[:, 0]
    psi0 = flux_int[:, 1].transpose(-1, -2)
    f = rhs_raw[:, 0, :, 1:-1]
    m_phi, m_psi = model._smooth_mask_components(coords, flux_int)
    denom = m_phi + m_psi
    w_phi = m_phi / denom
    w_psi = m_psi / denom
    alpha = (m_phi * m_psi) / denom
    expected_phi = w_phi * f + alpha * (phi0 - psi0)
    expected_psi = w_psi * f - alpha * (phi0 - psi0)

    torch.testing.assert_close(projected[:, 0], expected_phi)
    torch.testing.assert_close(projected[:, 1].transpose(-1, -2), expected_psi)


def test_smooth_mask_components_use_expected_axes_and_normalization():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
    )
    model = CouplingNet(cfg)
    coords = _make_projection_coords(n)
    flux_int = torch.zeros((1, 2, n, n), dtype=torch.float64)

    m_phi, m_psi = model._smooth_mask_components(coords, flux_int)

    expected = torch.tensor([0.75, 1.0, 0.75], dtype=torch.float64)
    assert m_phi.shape == (1, n, 1)
    assert m_psi.shape == (1, 1, n)
    torch.testing.assert_close(m_phi[0, :, 0], expected)
    torch.testing.assert_close(m_psi[0, 0, :], expected)
    assert torch.all(m_phi + m_psi > 0.0)
    assert torch.max(m_phi).item() == pytest.approx(1.0)
    assert torch.max(m_psi).item() == pytest.approx(1.0)


def test_smooth_mask_components_apply_power_without_moving_center():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
        smooth_mask_power=0.5,
    )
    model = CouplingNet(cfg)
    coords = _make_projection_coords(n)
    flux_int = torch.zeros((1, 2, n, n), dtype=torch.float64)

    m_phi, m_psi = model._smooth_mask_components(coords, flux_int)

    base = torch.tensor([0.75, 1.0, 0.75], dtype=torch.float64)
    expected = torch.sqrt(base)
    torch.testing.assert_close(m_phi[0, :, 0], expected)
    torch.testing.assert_close(m_psi[0, 0, :], expected)
    assert m_phi[0, 0, 0].item() > base[0].item()
    assert m_phi[0, 1, 0].item() == pytest.approx(1.0)
    assert m_psi[0, 0, 1].item() == pytest.approx(1.0)


def test_smooth_mask_diff_power_keeps_split_and_increases_boundary_difference():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
        smooth_mask_diff_power=0.5,
    )
    model = CouplingNet(cfg)
    coords = _make_projection_coords(n)
    flux_int = torch.zeros((1, 2, n, n), dtype=torch.float64)
    phi0 = torch.tensor(
        [[[1.0, 2.0, 3.0], [1.5, 0.5, -0.5], [0.25, -1.0, 2.5]]],
        dtype=torch.float64,
    )
    psi0 = torch.tensor(
        [[[0.5, -1.0, 2.0], [0.25, 1.25, -1.5], [1.0, 0.0, -0.25]]],
        dtype=torch.float64,
    )
    flux_int[:, 0] = phi0
    flux_int[:, 1] = psi0.transpose(-1, -2)
    rhs_raw = torch.zeros((1, 2, n, m), dtype=torch.float64)
    f = torch.tensor(
        [[[2.0, 1.0, -1.0], [0.5, 3.0, -0.75], [1.25, -0.25, 0.75]]],
        dtype=torch.float64,
    )
    rhs_raw[:, 0, :, 1:-1] = f

    projected = model._apply_balance_projection(flux_int, rhs_raw, coords)

    m_phi, m_psi = model._smooth_mask_components(coords, flux_int)
    denom = m_phi + m_psi
    w_phi = m_phi / denom
    w_psi = m_psi / denom
    alpha_soft = (m_phi * m_psi) / denom
    beta = 0.5 * torch.sqrt(2.0 * alpha_soft)
    expected_phi = w_phi * f + beta * (phi0 - psi0)
    expected_psi = w_psi * f - beta * (phi0 - psi0)
    old_beta = alpha_soft

    torch.testing.assert_close(projected[:, 0], expected_phi)
    torch.testing.assert_close(projected[:, 1].transpose(-1, -2), expected_psi)
    assert beta[0, 0, 0].item() > old_beta[0, 0, 0].item()
    torch.testing.assert_close(w_phi + w_psi, torch.ones_like(w_phi + w_psi))


def test_smooth_mask_power_and_diff_power_preserve_balance():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        balance_projection="smooth_mask",
        smooth_mask_normalize=True,
        smooth_mask_power=0.5,
        smooth_mask_diff_power=0.5,
    )
    model = CouplingNet(cfg)
    flux_int = torch.randn(2, 2, n, n, dtype=torch.float64)
    rhs_raw = torch.randn(2, 2, n, m, dtype=torch.float64)
    coords = _make_projection_coords(n)

    projected = model._apply_balance_projection(flux_int, rhs_raw, coords)

    phi = projected[:, 0]
    psi = projected[:, 1].transpose(-1, -2)
    torch.testing.assert_close(
        phi + psi,
        rhs_raw[:, 0, :, 1:-1],
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_smooth_mask_projection_config_validation():
    with pytest.raises(ValueError, match="Unsupported balance_projection"):
        CouplingNet(CouplingModelConfig(balance_projection="bad"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smooth_mask_eps"):
        CouplingNet(CouplingModelConfig(smooth_mask_eps=0.0))
    with pytest.raises(ValueError, match="smooth_mask_power"):
        CouplingNet(CouplingModelConfig(smooth_mask_power=0.0))
    with pytest.raises(ValueError, match="smooth_mask_diff_power"):
        CouplingNet(CouplingModelConfig(smooth_mask_diff_power=0.0))


def test_coupling_trainer_runs(tmp_path):
    _make_npz(tmp_path)
    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=2,
            batch_size=1,
            log_interval=1,
            use_lr_schedule=True,
            warmup_epochs=1,
            min_lr=1e-5,
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )
    trainer.train(ds)
    metrics = trainer.evaluate(ds)
    assert "loss" in metrics and "rel_flux" in metrics and "rel_sol" in metrics
    assert "loss_l2_consistency" in metrics
    assert "loss_energy_consistency" in metrics
    assert "loss_cross_consistency" in metrics


def test_coupling_trainer_validation_uses_eval_mode_and_restores_training(tmp_path):
    inputs, item = _make_coupling_dataset_item()
    flux_target = inputs[5]
    model = _InspectModeFluxModel(projected_flux=torch.zeros_like(flux_target))
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    trainer.model.train()
    loader = trainer._make_loader([item], shuffle=False)  # type: ignore[arg-type]

    metrics = trainer._evaluate_loader(loader)

    assert "loss" in metrics
    assert trainer.model.training is True
    assert model.training_states == [False]
    assert model.grad_enabled_states == [False]


def test_coupling_trainer_evaluate_uses_eval_mode_and_restores_training(tmp_path):
    inputs, item = _make_coupling_dataset_item()
    flux_target = inputs[5]
    model = _InspectModeFluxModel(projected_flux=torch.zeros_like(flux_target))
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    trainer.model.train()

    metrics = trainer.evaluate([item])  # type: ignore[arg-type]

    assert "loss" in metrics
    assert trainer.model.training is True
    assert model.training_states == [False]
    assert model.grad_enabled_states == [False]


def test_coupling_validation_metrics_are_deterministic_with_dropout(tmp_path):
    _inputs, item = _make_coupling_dataset_item()
    model_cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=1,
        dropout=0.9,
        dtype=torch.float64,
    )
    trainer = CouplingTrainer(
        model=CouplingNet(model_cfg),
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        model_cfg=model_cfg,
    )
    loader = trainer._make_loader([item], shuffle=False)  # type: ignore[arg-type]
    trainer.model.train()

    metrics_a = trainer._evaluate_loader(loader)
    metrics_b = trainer._evaluate_loader(loader)

    assert trainer.model.training is True
    for key in (
        "loss",
        "loss_l2_consistency",
        "loss_energy_consistency",
        "loss_cross_consistency",
        "rel_flux",
        "rel_sol",
    ):
        assert metrics_a[key] == pytest.approx(metrics_b[key])


def test_coupling_evaluator_uses_eval_mode_no_grad_and_restores_training(
    tmp_path,
    monkeypatch,
):
    inputs, item = _make_coupling_dataset_item()
    flux_target = inputs[5]
    model = _InspectModeFluxModel(projected_flux=torch.zeros_like(flux_target))
    model.train()
    evaluator = CouplingEvaluator(
        model=model,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        device=torch.device("cpu"),
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        "greenonet.coupling_evaluator._render_heatmap_task",
        lambda task: None,
    )

    evaluator.evaluate([item], batch_size=1, plot_workers=1)  # type: ignore[arg-type]

    assert model.training is True
    assert model.training_states == [False]
    assert model.grad_enabled_states == [False]
    assert (tmp_path / "test" / "metrics.csv").exists()


def test_coupling_trainer_compile_enabled_calls_torch_compile(tmp_path, monkeypatch):
    compiled = {"count": 0}

    def fake_compile(model):
        compiled["count"] += 1
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = CouplingNet(CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2))
    config = CouplingTrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
    )
    config.compile.enabled = True
    CouplingTrainer(
        model=model,
        config=config,
        work_dir=tmp_path,
        green_kernel=torch.zeros((2, 1, 3, 3), dtype=torch.float64),
        model_cfg=CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2),
    )

    assert compiled["count"] == 1


def test_coupling_trainer_saves_periodic_adam_checkpoints(tmp_path):
    _make_npz(tmp_path)
    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=3,
            batch_size=1,
            log_interval=1,
            periodic_checkpoint=CouplingPeriodicCheckpointConfig(
                enabled=True,
                every_epochs=2,
            ),
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )

    trainer.train(ds)

    assert (tmp_path / "coupling_model_epoch_0002.safetensors").exists()
    assert not (tmp_path / "coupling_model_epoch_0001.safetensors").exists()
    assert not (tmp_path / "coupling_model_epoch_0003.safetensors").exists()
    assert not (tmp_path / "coupling_model_lbfgs_best_rel_sol.safetensors").exists()


def test_coupling_trainer_saves_best_validation_rel_sol_checkpoint(tmp_path):
    _make_npz(tmp_path, "train_sample.npz")
    train_ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    _make_npz(val_dir, "val_sample.npz")
    val_ds = CouplingDataset(
        data_dir=val_dir,
        step_size=0.5,
        n_points_per_line=3,
    )

    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=2,
            batch_size=1,
            log_interval=1,
            best_rel_sol_checkpoint=CouplingBestRelSolCheckpointConfig(enabled=True),
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )

    trainer.train(train_ds, val_ds)

    assert (tmp_path / "coupling_model_adam_best_rel_sol.safetensors").exists()


def test_coupling_trainer_skips_best_validation_checkpoint_when_disabled(tmp_path):
    _make_npz(tmp_path, "train_sample.npz")
    train_ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    _make_npz(val_dir, "val_sample.npz")
    val_ds = CouplingDataset(
        data_dir=val_dir,
        step_size=0.5,
        n_points_per_line=3,
    )

    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            best_rel_sol_checkpoint=CouplingBestRelSolCheckpointConfig(enabled=False),
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )

    trainer.train(train_ds, val_ds)

    assert not (tmp_path / "coupling_model_adam_best_rel_sol.safetensors").exists()


def test_coupling_trainer_skips_best_validation_checkpoint_without_val_dataset(
    tmp_path,
):
    _make_npz(tmp_path)
    train_ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            best_rel_sol_checkpoint=CouplingBestRelSolCheckpointConfig(enabled=True),
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )

    trainer.train(train_ds)

    assert not (tmp_path / "coupling_model_adam_best_rel_sol.safetensors").exists()


def test_coupling_trainer_rejects_invalid_periodic_checkpoint_interval(tmp_path):
    _make_npz(tmp_path)
    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            periodic_checkpoint=CouplingPeriodicCheckpointConfig(
                enabled=True,
                every_epochs=0,
            ),
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
        model_cfg=model_cfg,
    )

    with pytest.raises(ValueError, match="every_epochs"):
        trainer.train(ds)


def test_coupling_evaluator_batched(tmp_path, monkeypatch):
    _make_npz(tmp_path, "sample0.npz")
    _make_npz(tmp_path, "sample1.npz")
    ds = CouplingDataset(
        data_dir=tmp_path,
        step_size=0.5,
        n_points_per_line=3,
    )
    green_kernel = torch.ones((2, 1, 3, 3), dtype=torch.float64)
    model_cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(model_cfg)
    evaluator = CouplingEvaluator(
        model=model,
        green_kernel=green_kernel,
        device=torch.device("cpu"),
        work_dir=tmp_path,
    )

    def _fake_render(task):
        base_path = Path(task["base_path"])
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.with_suffix(".png").write_text("stub")

    class _DummyFuture:
        def __init__(self, fn, arg):
            self._fn = fn
            self._arg = arg

        def result(self):
            return self._fn(self._arg)

    class _DummyExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def submit(self, fn, arg):
            return _DummyFuture(fn, arg)

        def shutdown(self, wait=True):
            return None

    monkeypatch.setattr(
        "greenonet.coupling_evaluator._render_heatmap_task", _fake_render
    )
    monkeypatch.setattr(
        "greenonet.coupling_evaluator.ProcessPoolExecutor", _DummyExecutor
    )

    evaluator.evaluate(ds, dataset_name="test", batch_size=2, plot_workers=2)
    csv_path = tmp_path / "test" / "metrics.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert "sample0" in lines[1] or "sample0" in lines[2]
    assert "sample1" in lines[1] or "sample1" in lines[2]
    assert (tmp_path / "test" / "sample0_sol_exact.png").exists()
    assert (tmp_path / "test" / "sample1_sol_exact.png").exists()
    assert (tmp_path / "test" / "sample0_null_sol_x.png").exists()
    assert (tmp_path / "test" / "sample0_null_sol_y.png").exists()
    assert (tmp_path / "test" / "sample0_null_sol_residual.png").exists()
    assert (tmp_path / "test" / "sample0_closure_phi_residual.png").exists()
    assert (tmp_path / "test" / "sample0_closure_psi_residual.png").exists()


def test_step_loss_ignores_raw_split_sum_without_balance_loss(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    rhs_raw[:, 0, :, 1:-1] = 3.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    trainer = CouplingTrainer(
        model=_DummyFluxModel(projected_flux=projected_flux),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    loss, metrics = trainer._step_loss(
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )

    assert abs(loss.item()) < 1e-12


def test_single_stage_optimization_resolution_uses_shared_config(tmp_path):
    trainer = CouplingTrainer(
        model=_RawSumFromRhsModel(),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            learning_rate=1.25,
            use_lr_schedule=True,
            warmup_epochs=9,
            min_lr=1e-3,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    single_cfg = trainer._optimization_config()

    assert single_cfg.learning_rate == 1.25
    assert single_cfg.warmup_epochs == 9
    assert single_cfg.min_lr == 1e-3


def test_coupling_optimizer_splits_source_lift_parameter_group(tmp_path):
    model = _make_optimizer_test_coupling_net(source_lift_enabled=True)
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_learning_rate=1.0e-4,
            weight_decay=1.0e-2,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    optimizer = trainer._build_optimizer(trainer.config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    main_group = next(
        group for group in optimizer.param_groups if group["name"] == "main"
    )
    source_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "source_stencil_lift"
    )
    assert main_group["lr"] == pytest.approx(1.0e-3)
    assert source_group["lr"] == pytest.approx(1.0e-4)
    assert main_group["weight_decay"] == pytest.approx(1.0e-2)
    assert source_group["weight_decay"] == pytest.approx(1.0e-2)

    source_lift = trainer.model.source_stencil_lift
    assert source_lift is not None
    source_expected_ids = {
        id(param) for param in source_lift.parameters() if param.requires_grad
    }
    source_group_ids = {id(param) for param in source_group["params"]}
    main_group_ids = {id(param) for param in main_group["params"]}
    trainable_ids = {
        id(param) for param in trainer.model.parameters() if param.requires_grad
    }

    assert source_group_ids == source_expected_ids
    assert main_group_ids.isdisjoint(source_group_ids)
    assert main_group_ids | source_group_ids == trainable_ids


def test_coupling_optimizer_source_weight_decay_override_splits_group(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=1.0e-2,
            source_stencil_lift_weight_decay=2.0e-3,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    optimizer = trainer._build_optimizer(trainer.config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    main_group = next(
        group for group in optimizer.param_groups if group["name"] == "main"
    )
    source_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "source_stencil_lift"
    )
    assert main_group["lr"] == pytest.approx(1.0e-3)
    assert source_group["lr"] == pytest.approx(1.0e-3)
    assert main_group["weight_decay"] == pytest.approx(1.0e-2)
    assert source_group["weight_decay"] == pytest.approx(2.0e-3)


def test_coupling_optimizer_honors_source_lr_and_weight_decay(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_learning_rate=1.0e-4,
            weight_decay=1.0e-2,
            source_stencil_lift_weight_decay=2.0e-3,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    optimizer = trainer._build_optimizer(trainer.config)
    source_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "source_stencil_lift"
    )

    assert source_group["lr"] == pytest.approx(1.0e-4)
    assert source_group["weight_decay"] == pytest.approx(2.0e-3)


def test_coupling_optimizer_keeps_single_group_without_source_lift_lr(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=False),
        config=CouplingTrainingConfig(learning_rate=2.0e-3, weight_decay=3.0e-2),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    optimizer = trainer._build_optimizer(trainer.config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(3.0e-2)


def _manual_global_grad_norm(module: nn.Module) -> torch.Tensor:
    grads = [
        param.grad.reshape(-1)
        for param in module.parameters()
        if param.grad is not None
    ]
    if not grads:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.linalg.vector_norm(torch.cat(grads), ord=2)


def test_coupling_trainer_default_gradient_clipping_uses_max_norm_one(tmp_path):
    model = nn.Linear(2, 1, bias=False).to(dtype=torch.float64)
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    for param in model.parameters():
        param.grad = torch.full_like(param, 10.0)

    returned_norm = trainer._clip_gradients_if_enabled(trainer.config)

    assert returned_norm is not None
    assert returned_norm.item() > 1.0
    assert _manual_global_grad_norm(model).item() <= 1.0 + 1.0e-12


def test_coupling_trainer_custom_gradient_clipping_max_norm(tmp_path):
    model = nn.Linear(3, 1, bias=False).to(dtype=torch.float64)
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(gradient_clip_max_norm=0.25),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    for param in model.parameters():
        param.grad = torch.full_like(param, 5.0)

    returned_norm = trainer._clip_gradients_if_enabled(trainer.config)

    assert returned_norm is not None
    assert returned_norm.item() > 0.25
    assert _manual_global_grad_norm(model).item() <= 0.25 + 1.0e-12


def test_coupling_trainer_gradient_clipping_none_leaves_gradients_unchanged(tmp_path):
    model = nn.Linear(2, 1, bias=False).to(dtype=torch.float64)
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(gradient_clip_max_norm=None),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    for param in model.parameters():
        param.grad = torch.full_like(param, 3.0)
    before = _manual_global_grad_norm(model)

    returned_norm = trainer._clip_gradients_if_enabled(trainer.config)

    assert returned_norm is None
    torch.testing.assert_close(_manual_global_grad_norm(model), before)


def test_coupling_trainer_rejects_non_positive_gradient_clip_max_norm(tmp_path):
    trainer = CouplingTrainer(
        model=nn.Linear(2, 1).to(dtype=torch.float64),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(gradient_clip_max_norm=0.0),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="gradient_clip_max_norm"):
        trainer._clip_gradients_if_enabled(trainer.config)


def test_coupling_optimizer_rejects_source_lift_lr_without_source_lift(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=False),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_learning_rate=1.0e-4,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="source_stencil_lift"):
        trainer._build_optimizer(trainer.config)


def test_coupling_optimizer_rejects_source_weight_decay_without_source_lift(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=False),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_weight_decay=1.0e-2,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="source_stencil_lift"):
        trainer._build_optimizer(trainer.config)


def test_coupling_optimizer_rejects_non_positive_source_lift_lr(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_learning_rate=0.0,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="source_stencil_lift_learning_rate"):
        trainer._build_optimizer(trainer.config)


def test_coupling_optimizer_rejects_non_positive_main_lr(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(learning_rate=0.0),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="learning_rate"):
        trainer._build_optimizer(trainer.config)


def test_coupling_optimizer_rejects_negative_weight_decay(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=-1.0e-2,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="weight_decay"):
        trainer._build_optimizer(trainer.config)


def test_coupling_optimizer_rejects_negative_source_weight_decay(tmp_path):
    trainer = CouplingTrainer(
        model=_make_optimizer_test_coupling_net(source_lift_enabled=True),
        config=CouplingTrainingConfig(
            learning_rate=1.0e-3,
            source_stencil_lift_weight_decay=-1.0e-2,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="source_stencil_lift_weight_decay"):
        trainer._build_optimizer(trainer.config)


def test_energy_consistency_loss_zero_for_identical_branch_reconstructions(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    base_grid = _make_zero_dirichlet_grid(m_points=5)
    u_phi_x = base_grid
    u_psi_y = base_grid.transpose(-1, -2)
    a_vals = torch.ones((1, 2, 3, 5), dtype=torch.float64)

    loss_energy = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_vals,
        x_axis=axis,
        y_axis=axis,
    )

    assert loss_energy.item() >= 0.0
    assert loss_energy.item() == pytest.approx(0.0, abs=1e-12)


def test_energy_consistency_physical_energy_constant_residual_is_zero(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    u_phi_x = torch.full((1, 5, 5), 3.0, dtype=torch.float64)
    u_psi_y = torch.zeros((1, 5, 5), dtype=torch.float64)
    a_vals = torch.ones((1, 2, 3, 5), dtype=torch.float64)

    loss_energy = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_vals,
        x_axis=axis,
        y_axis=axis,
    )

    assert loss_energy.item() == pytest.approx(0.0, abs=1e-12)


def test_energy_consistency_physical_energy_linear_x_residual(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    x_grid = axis.view(1, 1, -1).expand(1, 5, 5)
    u_phi_x = x_grid.clone()
    u_psi_y = torch.zeros((1, 5, 5), dtype=torch.float64)
    a_vals = torch.ones((1, 2, 3, 5), dtype=torch.float64)

    loss_energy = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_vals,
        x_axis=axis,
        y_axis=axis,
    )

    assert torch.isfinite(loss_energy)
    assert loss_energy.item() > 0.0
    # Only x-faces contribute: n interior rows * (m - 1) x-faces * hx * hy.
    assert loss_energy.item() == pytest.approx(3.0 * 4.0 * 0.25 * 0.25)


def test_energy_consistency_physical_energy_scales_linearly_with_coefficient(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    x_grid = axis.view(1, 1, -1).expand(1, 5, 5)
    u_phi_x = x_grid.clone()
    u_psi_y = torch.zeros((1, 5, 5), dtype=torch.float64)
    a_ones = torch.ones((1, 2, 3, 5), dtype=torch.float64)
    a_twos = 2.0 * a_ones

    loss_unit_coeff = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_ones,
        x_axis=axis,
        y_axis=axis,
    )
    loss_double_coeff = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_twos,
        x_axis=axis,
        y_axis=axis,
    )

    assert loss_double_coeff.item() == pytest.approx(2.0 * loss_unit_coeff.item())


def test_energy_consistency_physical_energy_is_scalar_and_differentiable(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((2, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    x_grid = axis.view(1, 1, -1).expand(2, 5, 5).clone()
    y_grid = axis.view(1, -1, 1).expand(2, 5, 5).clone()
    u_phi_x = (x_grid.square() + 0.25 * y_grid).clone().requires_grad_(True)
    u_psi_y = torch.zeros_like(u_phi_x).requires_grad_(True)
    a_vals = torch.ones((2, 2, 3, 5), dtype=torch.float64)

    loss_energy = trainer._energy_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_vals,
        x_axis=axis,
        y_axis=axis,
    )
    loss_energy.backward()

    assert loss_energy.ndim == 0
    assert u_phi_x.grad is not None
    assert u_psi_y.grad is not None
    assert torch.linalg.norm(u_phi_x.grad).item() > 0.0
    assert torch.linalg.norm(u_psi_y.grad).item() > 0.0


def test_energy_consistency_physical_energy_shape_mismatch_errors(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    u_phi_x = torch.zeros((1, 5, 5), dtype=torch.float64)
    u_psi_y_bad = torch.zeros((1, 4, 5), dtype=torch.float64)
    a_vals = torch.ones((1, 2, 3, 5), dtype=torch.float64)

    with pytest.raises(ValueError, match="same common grid"):
        trainer._energy_consistency_loss(
            u_phi_x=u_phi_x,
            u_psi_y=u_psi_y_bad,
            a_vals=a_vals,
            x_axis=axis,
            y_axis=axis,
        )

    with pytest.raises(ValueError, match="x-face coefficient shape"):
        trainer._energy_consistency_loss(
            u_phi_x=u_phi_x,
            u_psi_y=torch.zeros((1, 5, 5), dtype=torch.float64),
            a_vals=torch.ones((1, 2, 2, 5), dtype=torch.float64),
            x_axis=axis,
            y_axis=axis,
        )


def test_cross_consistency_loss_uses_cross_terms_not_self_closure(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    base_grid = _make_zero_dirichlet_grid(m_points=5)
    u_phi_x = base_grid
    u_psi_y = (2.0 * base_grid).transpose(-1, -2)
    a_vals = torch.ones((1, 2, 3, 5), dtype=torch.float64)
    b_vals = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    c_vals = torch.zeros((1, 2, 3, 5), dtype=torch.float64)

    u_psi_x_view = u_psi_y.transpose(-1, -2)
    phi_x = torch.zeros((1, 3, 5), dtype=torch.float64)
    phi_x[:, :, 1:-1] = line_operator_fd(
        u_lines=u_psi_x_view[:, 1:-1, :],
        a_lines=a_vals[:, 0],
        b_lines=b_vals[:, 0],
        c_lines=c_vals[:, 0],
        axis_coords=axis,
    )

    u_phi_y_view = u_phi_x.transpose(-1, -2)
    psi_y = torch.zeros((1, 3, 5), dtype=torch.float64)
    psi_y[:, :, 1:-1] = line_operator_fd(
        u_lines=u_phi_y_view[:, 1:-1, :],
        a_lines=a_vals[:, 1],
        b_lines=b_vals[:, 1],
        c_lines=c_vals[:, 1],
        axis_coords=axis,
    )

    loss_cross = trainer._cross_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        phi_x=phi_x,
        psi_y=psi_y,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        x_axis=axis,
        y_axis=axis,
    )

    self_x = line_operator_fd(
        u_lines=u_phi_x[:, 1:-1, :],
        a_lines=a_vals[:, 0],
        b_lines=b_vals[:, 0],
        c_lines=c_vals[:, 0],
        axis_coords=axis,
    )
    self_y = line_operator_fd(
        u_lines=u_psi_y[:, 1:-1, :],
        a_lines=a_vals[:, 1],
        b_lines=b_vals[:, 1],
        c_lines=c_vals[:, 1],
        axis_coords=axis,
    )
    self_mismatch = (self_x - phi_x[:, :, 1:-1]).pow(2).sum() + (
        self_y - psi_y[:, :, 1:-1]
    ).pow(2).sum()

    assert loss_cross.item() == pytest.approx(0.0, abs=1e-12)
    assert self_mismatch.item() > 0.0


def test_step_loss_ignores_energy_consistency_when_disabled(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    model = _DummyFluxModel(projected_flux=projected_flux)
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=False,
                energy_enabled=False,
                energy_weight=1.0,
                cross_enabled=False,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    loss, metrics = trainer._step_loss(
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )

    assert metrics["loss_energy_consistency"] > 0.0
    assert abs(loss.item()) < 1e-12


def test_energy_consistency_loss_backpropagates_to_projected_flux(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    model = _TrainableFluxModel(projected_flux=projected_flux)
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=False,
                energy_enabled=True,
                energy_weight=1.0,
                cross_enabled=False,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    loss, metrics = trainer._step_loss(
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )
    loss.backward()

    assert metrics["loss_energy_consistency"] > 0.0
    assert model.projected_flux.grad is not None
    assert torch.linalg.norm(model.projected_flux.grad).item() > 0.0


def test_step_loss_matches_weighted_sum_of_enabled_losses(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    trainer = CouplingTrainer(
        model=_DummyFluxModel(projected_flux=projected_flux),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=True,
                l2_weight=0.5,
                energy_enabled=True,
                energy_weight=2.0,
                cross_enabled=True,
                cross_weight=3.0,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    loss, metrics = trainer._step_loss(
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )

    expected = (
        0.5 * metrics["loss_l2_consistency"]
        + 2.0 * metrics["loss_energy_consistency"]
        + 3.0 * metrics["loss_cross_consistency"]
    )
    assert loss.item() == pytest.approx(expected)


def test_source_lift_metrics_are_finite_when_enabled(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    rhs_raw[:, 0, :, 1:-1] = torch.arange(1, 10, dtype=torch.float64).reshape(1, 3, 3)
    rhs_tilde = rhs_raw.clone()
    rhs_norm = torch.ones((1, 2, 3), dtype=torch.float64)
    a_vals[:] = 1.0
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        source_stencil_lift=SourceStencilLiftConfig(enabled=True, hidden_dim=4),
    )
    trainer = CouplingTrainer(
        model=CouplingNet(cfg),
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=False,
                energy_enabled=False,
                cross_enabled=False,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    _loss, metrics = trainer._step_loss(
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    )

    for key in (
        "source_lift_corr_g_f",
        "source_lift_rel_diff_g_f",
        "source_lift_g_rms",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_evaluate_reports_all_three_loss_components(tmp_path):
    (
        coords,
        rhs_raw,
        rhs_tilde,
        rhs_norm,
        sol,
        flux_target,
        a_vals,
        b_vals,
        c_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    trainer = CouplingTrainer(
        model=_DummyFluxModel(projected_flux=projected_flux),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    dataset = [
        (
            coords,
            rhs_raw[0],
            rhs_tilde[0],
            rhs_norm[0],
            sol[0],
            flux_target[0],
            a_vals[0],
            b_vals[0],
            c_vals[0],
            torch.zeros_like(a_vals[0]),
        )
    ]

    metrics = trainer.evaluate(dataset)  # type: ignore[arg-type]

    assert "loss" in metrics
    assert "loss_l2_consistency" in metrics
    assert "loss_energy_consistency" in metrics
    assert "loss_cross_consistency" in metrics
