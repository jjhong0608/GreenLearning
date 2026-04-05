from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch
from torch import nn

from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.coupling_model import CouplingNet
from greenonet.coupling_trainer import CouplingTrainer
from greenonet.config import (
    CouplingBestRelSolCheckpointConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
)


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


def _set_structured_baseline_context(
    model: CouplingNet,
    *,
    n_lines: int,
    m_points: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    integration_rule: str = "simpson",
) -> torch.Tensor:
    green_kernel = torch.ones((2, n_lines, m_points, m_points), dtype=torch.float64)
    green_kernel[0] *= scale_x
    green_kernel[1] *= scale_y
    model.set_structured_baseline_context(
        green_kernel=green_kernel,
        integration_rule=integration_rule,
    )
    return green_kernel


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
        projected_flux[:, 1, :, 1:-1] = (
            0.75 * rhs_inner
        ).transpose(-1, -2) + 0.0 * self.anchor
        return projected_flux


def _flux_from_q_common(q: torch.Tensor, common: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build post-projection style flux tensors from interior gauge/common parts.

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
    return -(
        lines[..., 2:] - 2.0 * lines[..., 1:-1] + lines[..., :-2]
    ) / (spacing**2)


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
    _set_structured_baseline_context(model, n_lines=1, m_points=3)
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
    _set_structured_baseline_context(model, n_lines=1, m_points=3)

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
    assert _has_grad(model.branch_b)
    assert _has_grad(model.branch_c)
    assert _has_grad(model.branch_rhs)
    assert _has_grad(model.branch_fuser)
    assert _has_grad(model.trunk)


def test_coupling_has_no_global_baseline_lambda():
    model = CouplingNet(CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2))

    assert not hasattr(model, "lambda_logit")
    assert not hasattr(model, "baseline_lambda")
    assert all("lambda" not in name for name, _ in model.named_parameters())


def test_coupling_computes_local_pointwise_green_responses_on_common_grid():
    cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    _set_structured_baseline_context(
        model,
        n_lines=3,
        m_points=5,
        scale_x=1.0,
        scale_y=2.0,
        integration_rule="trapezoid",
    )

    coords = torch.zeros((2, 3, 5, 2), dtype=torch.float64)
    axis = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    coords[0, :, :, 0] = axis.unsqueeze(0).expand(3, -1)
    coords[0, :, :, 1] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 0] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 1] = axis.unsqueeze(0).expand(3, -1)
    rhs_raw = torch.zeros((1, 2, 3, 5), dtype=torch.float64)

    response_x_lines = torch.tensor(
        [
            [
                [10.0, 1.0, 2.0, 3.0, 20.0],
                [11.0, 4.0, 5.0, 6.0, 21.0],
                [12.0, 7.0, 8.0, 9.0, 22.0],
            ]
        ],
        dtype=torch.float64,
    )
    response_y_lines = torch.tensor(
        [
            [
                [30.0, 13.0, 14.0, 15.0, 31.0],
                [32.0, 16.0, 17.0, 18.0, 33.0],
                [34.0, 19.0, 20.0, 21.0, 35.0],
            ]
        ],
        dtype=torch.float64,
    )
    integrate_seq = _sequential_integrator([response_x_lines, response_y_lines])

    def _fake_integrate_green_lines(
        self,
        green: torch.Tensor,
        values: torch.Tensor,
        axis_coords: torch.Tensor,
    ) -> torch.Tensor:
        return integrate_seq(green, values, axis_coords)

    model._integrate_green_lines = MethodType(_fake_integrate_green_lines, model)

    response_x_local, response_y_local = model._compute_local_green_response_fields(
        coords, rhs_raw
    )

    assert torch.allclose(response_x_local, response_x_lines[:, :, 1:-1])
    assert torch.allclose(
        response_y_local, response_y_lines[:, :, 1:-1].transpose(-1, -2)
    )


def test_coupling_structured_baseline_uses_local_pointwise_smoothed_inverse_weight():
    cfg = CouplingModelConfig(branch_input_dim=5, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    _set_structured_baseline_context(model, n_lines=3, m_points=5, integration_rule="trapezoid")

    coords = torch.zeros((2, 3, 5, 2), dtype=torch.float64)
    axis = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    coords[0, :, :, 0] = axis.unsqueeze(0).expand(3, -1)
    coords[0, :, :, 1] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 0] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 1] = axis.unsqueeze(0).expand(3, -1)
    rhs_raw = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    rhs_common = torch.tensor(
        [[[2.0, 4.0, 6.0], [3.0, 5.0, 7.0], [1.0, 8.0, 9.0]]],
        dtype=torch.float64,
    )
    rhs_raw[:, 0, :, 1:-1] = rhs_common
    rhs_raw[:, 1, :, 1:-1] = rhs_common.transpose(-1, -2)

    response_x_local = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 0.0], [1.5, 0.5, 2.5]]],
        dtype=torch.float64,
    )
    response_y_local = torch.tensor(
        [[[0.0, 5.0, 1.0], [2.0, 1.0, 4.0], [3.0, 2.5, 0.5]]],
        dtype=torch.float64,
    )

    def _fake_local_responses(
        self, coords: torch.Tensor, rhs_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del coords, rhs_raw
        return response_x_local, response_y_local

    model._compute_local_green_response_fields = MethodType(
        _fake_local_responses, model
    )

    baseline_int = model._build_structured_baseline(coords, rhs_raw)

    assert model.BASELINE_EPS == pytest.approx(1e-12)
    expected_mag_x = torch.sqrt(response_x_local.pow(2) + model.BASELINE_EPS**2)
    expected_mag_y = torch.sqrt(response_y_local.pow(2) + model.BASELINE_EPS**2)
    expected_weight = expected_mag_y / (expected_mag_x + expected_mag_y)
    weight_with_old_denominator_eps = expected_mag_y / (
        expected_mag_x + expected_mag_y + model.BASELINE_EPS
    )
    centered = (expected_weight - 0.5) * rhs_common
    expected_phi = centered
    expected_psi_common = -centered

    assert expected_weight[0, 0, 0].item() == pytest.approx(0.5)
    assert weight_with_old_denominator_eps[0, 0, 0].item() == pytest.approx(1.0 / 3.0)
    assert not torch.allclose(expected_weight, weight_with_old_denominator_eps)
    assert torch.allclose(baseline_int[:, 0], expected_phi)
    assert torch.allclose(baseline_int[:, 1], expected_psi_common.transpose(-1, -2))
    assert torch.allclose(
        baseline_int[:, 0] + baseline_int[:, 1].transpose(-1, -2),
        torch.zeros_like(rhs_common),
    )


def test_coupling_zero_correction_reduces_to_projected_plus_centered_structured_baseline():
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
    _set_structured_baseline_context(
        model,
        n_lines=3,
        m_points=5,
        scale_x=1.0,
        scale_y=2.0,
    )
    model.branch_a = _ZeroBranch(cfg.hidden_dim)
    model.branch_b = _ZeroBranch(cfg.hidden_dim)
    model.branch_c = _ZeroBranch(cfg.hidden_dim)
    model.branch_rhs = _ZeroBranch(cfg.hidden_dim)
    model.trunk = _ZeroBranch(cfg.hidden_dim)
    with torch.no_grad():
        model.branch_fuser.weight.zero_()
        model.branch_fuser.bias.zero_()

    coords = torch.zeros((2, 3, 5, 2), dtype=torch.float64)
    axis = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    coords[0, :, :, 0] = axis.unsqueeze(0).expand(3, -1)
    coords[0, :, :, 1] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 0] = torch.linspace(0.25, 0.75, 3, dtype=torch.float64).unsqueeze(-1)
    coords[1, :, :, 1] = axis.unsqueeze(0).expand(3, -1)
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

    baseline_int = model._build_structured_baseline(coords, rhs_raw)
    projected_phi = 0.5 * rhs_x_int
    projected_psi = 0.5 * rhs_x_int.transpose(-1, -2)
    expected_phi = projected_phi + baseline_int[:, 0]
    expected_psi = projected_psi + baseline_int[:, 1]
    assert torch.allclose(
        projected_flux[:, 0, :, 1:-1], expected_phi, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        projected_flux[:, 1, :, 1:-1], expected_psi, atol=1e-12, rtol=1e-12
    )


def test_coupling_learned_branch_uses_a_b_c_and_f():
    torch.manual_seed(0)
    cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    _set_structured_baseline_context(
        model,
        n_lines=1,
        m_points=3,
        scale_x=0.0,
        scale_y=0.0,
    )
    coords = torch.zeros((2, 1, 3, 2), dtype=torch.float64)
    coords[0, 0, :, 0] = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    coords[1, 0, :, 1] = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    rhs_raw = torch.zeros((1, 2, 1, 3), dtype=torch.float64)
    rhs_norm = torch.ones((1, 2, 1), dtype=torch.float64)

    a_base = torch.ones((1, 2, 1, 3), dtype=torch.float64)
    b_base = torch.zeros_like(a_base)
    c_base = torch.zeros_like(a_base)
    rhs_tilde_base = torch.zeros_like(a_base)
    rhs_tilde_alt = rhs_tilde_base.clone()
    rhs_tilde_alt[:, :, :, 1] = 1.0

    out_base = model(
        coords=coords,
        a_vals=a_base,
        b_vals=b_base,
        c_vals=c_base,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde_base,
        rhs_norm=rhs_norm,
    )
    out_a = model(
        coords=coords,
        a_vals=2.0 * a_base,
        b_vals=b_base,
        c_vals=c_base,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde_base,
        rhs_norm=rhs_norm,
    )
    out_b = model(
        coords=coords,
        a_vals=a_base,
        b_vals=torch.ones_like(b_base),
        c_vals=c_base,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde_base,
        rhs_norm=rhs_norm,
    )
    out_c = model(
        coords=coords,
        a_vals=a_base,
        b_vals=b_base,
        c_vals=torch.ones_like(c_base),
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde_base,
        rhs_norm=rhs_norm,
    )
    out_f = model(
        coords=coords,
        a_vals=a_base,
        b_vals=b_base,
        c_vals=c_base,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde_alt,
        rhs_norm=rhs_norm,
    )

    assert not torch.allclose(out_base, out_a)
    assert not torch.allclose(out_base, out_b)
    assert not torch.allclose(out_base, out_c)
    assert not torch.allclose(out_base, out_f)


def test_coupling_forward_calls_balance_projection():
    cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    _set_structured_baseline_context(model, n_lines=1, m_points=3)
    called = {"count": 0}
    captured_target = {"value": None}

    def _fake_projection(
        self, flux_int: torch.Tensor, rhs_target_int: torch.Tensor
    ) -> torch.Tensor:
        called["count"] += 1
        captured_target["value"] = rhs_target_int.clone()
        return flux_int

    model._apply_balance_projection = MethodType(_fake_projection, model)

    coords = torch.zeros((2, 1, 3, 2), dtype=torch.float64)
    vals = torch.ones((1, 2, 1, 3), dtype=torch.float64)
    rhs_norm = torch.ones((1, 2, 1), dtype=torch.float64)

    projected_flux = model(
        coords=coords,
        a_vals=vals,
        b_vals=vals,
        c_vals=vals,
        rhs_raw=vals,
        rhs_tilde=vals,
        rhs_norm=rhs_norm,
    )

    assert called["count"] == 1
    assert captured_target["value"] is not None
    expected_target = vals[:, 0, :, 1:-1]
    assert torch.allclose(captured_target["value"], expected_target)
    assert projected_flux.shape == (1, 2, 1, 3)


def test_coupling_model_named_parameters_do_not_include_lambda():
    model = CouplingNet(CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2))

    assert "lambda_logit" not in dict(model.named_parameters())


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


def test_coupling_trainer_skips_best_validation_checkpoint_without_val_dataset(tmp_path):
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
            lambda_consistency=0.0,
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


def test_flux_consistency_loss_zero_for_identical_branch_reconstructions(tmp_path):
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

    loss_flux = trainer._flux_consistency_loss(
        u_phi_x=u_phi_x,
        u_psi_y=u_psi_y,
        a_vals=a_vals,
        x_axis=axis,
        y_axis=axis,
    )

    assert loss_flux.item() >= 0.0
    assert loss_flux.item() == pytest.approx(0.0, abs=1e-12)


def test_step_loss_ignores_flux_consistency_when_disabled(tmp_path):
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
            lambda_consistency=0.0,
            flux_consistency_enabled=False,
            lambda_flux_consistency=1.0,
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

    assert metrics["loss_flux_consistency"] > 0.0
    assert abs(loss.item()) < 1e-12


def test_flux_consistency_loss_backpropagates_to_projected_flux(tmp_path):
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
            lambda_consistency=0.0,
            flux_consistency_enabled=True,
            lambda_flux_consistency=1.0,
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

    assert metrics["loss_flux_consistency"] > 0.0
    assert model.projected_flux.grad is not None
    assert torch.linalg.norm(model.projected_flux.grad).item() > 0.0
