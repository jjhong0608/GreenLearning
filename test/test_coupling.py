from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.coupling_model import CouplingNet, FiveStencilStencilMLPCoupler
from greenonet.coupling_trainer import (
    CouplingTrainer,
    freeze_main_train_coupler_only,
)
from greenonet.config import (
    CouplerConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingHybridDetachConfig,
    CouplingLossTermConfig,
    CouplingLossesConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingStage2Config,
    CouplingTrainingConfig,
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


class _HybridFluxModel(nn.Module):
    def __init__(self, projected_int: torch.Tensor, coupled_int: torch.Tensor) -> None:
        super().__init__()
        if projected_int.shape != coupled_int.shape:
            raise ValueError("projected_int and coupled_int must have matching shapes")
        self.projected_int = nn.Parameter(projected_int.clone())
        self.coupled_delta = nn.Parameter(coupled_int.clone() - projected_int)
        self.coupler = nn.Linear(1, 1, dtype=projected_int.dtype)

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
        return_intermediates: bool = False,
        detach_coupler_input: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del coords, a_vals, b_vals, c_vals, rhs_tilde, rhs_norm, detach_coupler_input
        projected_int = self.projected_int
        coupled_int = projected_int + self.coupled_delta
        bsz, axis, n_lines, n_inner = coupled_int.shape
        output_flux = torch.zeros(
            (bsz, axis, n_lines, n_inner + 2),
            dtype=coupled_int.dtype,
            device=coupled_int.device,
        )
        output_flux[:, :, :, 1:-1] = coupled_int
        if return_intermediates:
            pre_projection_residual = rhs_raw[:, 0, :, 1:-1] - (
                projected_int[:, 0] + projected_int[:, 1].transpose(-1, -2)
            )
            return output_flux, {
                "raw_int": projected_int,
                "pre_projection_residual": pre_projection_residual,
                "projected_int": projected_int,
                "coupled_int": coupled_int,
            }
        return output_flux


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


def test_five_stencil_coupler_shape_preservation():
    torch.set_default_dtype(torch.float64)
    bsz = 3
    n = 7
    m = n + 2
    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    coupler = FiveStencilStencilMLPCoupler(CouplerConfig(enabled=True)).to(
        dtype=torch.float64
    )

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)

    assert out.shape == raw_int.shape


def test_five_stencil_coupler_initial_identity():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 5
    m = n + 2
    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    coupler = FiveStencilStencilMLPCoupler(
        CouplerConfig(
            enabled=True,
            hidden_channels=64,
            depth=2,
            activation="gelu",
            residual_scale_init=0.05,
        )
    ).to(dtype=torch.float64)

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)

    torch.testing.assert_close(out, raw_int)


def test_five_stencil_coupler_uses_projected_diff_and_auxiliary_residual():
    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    phi_projected = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=torch.float64,
    )
    psi_projected = torch.tensor(
        [[[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]]],
        dtype=torch.float64,
    )
    f = phi_projected + psi_projected
    auxiliary_residual = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
        dtype=torch.float64,
    )

    ax = torch.full((1, n, n), 1.0, dtype=torch.float64)
    ay = torch.full((1, n, n), 2.0, dtype=torch.float64)
    bx = torch.full((1, n, n), 3.0, dtype=torch.float64)
    by = torch.full((1, n, n), 4.0, dtype=torch.float64)
    cx = torch.full((1, n, n), 5.0, dtype=torch.float64)
    cy = torch.full((1, n, n), 6.0, dtype=torch.float64)

    raw_int = torch.zeros((1, 2, n, n), dtype=torch.float64)
    raw_int[:, 0] = phi_projected
    raw_int[:, 1] = psi_projected.transpose(-1, -2)

    rhs_raw = torch.zeros((1, 2, n, m), dtype=torch.float64)
    rhs_raw[:, 0, :, 1:-1] = f

    a_vals = torch.zeros((1, 2, n, m), dtype=torch.float64)
    b_vals = torch.zeros((1, 2, n, m), dtype=torch.float64)
    c_vals = torch.zeros((1, 2, n, m), dtype=torch.float64)
    a_vals[:, 0, :, 1:-1] = ax
    a_vals[:, 1, :, 1:-1] = ay.transpose(-1, -2)
    b_vals[:, 0, :, 1:-1] = bx
    b_vals[:, 1, :, 1:-1] = by.transpose(-1, -2)
    c_vals[:, 0, :, 1:-1] = cx
    c_vals[:, 1, :, 1:-1] = cy.transpose(-1, -2)

    coupler = FiveStencilStencilMLPCoupler(CouplerConfig(enabled=True, eps=0.0)).to(
        dtype=torch.float64
    )

    q, scale, phi0, psi0 = coupler._build_canonical_point_features(
        raw_int,
        a_vals,
        b_vals,
        c_vals,
        rhs_raw,
        auxiliary_residual=auxiliary_residual,
    )

    expected_scale = torch.sqrt(
        torch.mean(f * f, dim=(-1, -2), keepdim=True) + coupler.eps
    )
    expected_diff_hat = 0.5 * (phi_projected - psi_projected) / expected_scale
    expected_res_hat = auxiliary_residual / expected_scale

    assert q.shape == (1, 9, n, n)
    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(phi0, phi_projected)
    torch.testing.assert_close(psi0, psi_projected)
    torch.testing.assert_close(q[:, 0], expected_diff_hat)
    torch.testing.assert_close(q[:, 1], expected_res_hat)
    torch.testing.assert_close(q[:, 2], f / expected_scale)
    torch.testing.assert_close(q[:, 3], ax)
    torch.testing.assert_close(q[:, 4], ay)
    torch.testing.assert_close(q[:, 5], bx)
    torch.testing.assert_close(q[:, 6], by)
    torch.testing.assert_close(q[:, 7], cx)
    torch.testing.assert_close(q[:, 8], cy)


def test_five_stencil_coupler_rejects_bad_auxiliary_residual_shape():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 4
    m = n + 2
    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    bad_auxiliary_residual = torch.zeros((bsz, n, n + 1), dtype=torch.float64)
    coupler = FiveStencilStencilMLPCoupler(CouplerConfig(enabled=True)).to(
        dtype=torch.float64
    )

    with pytest.raises(ValueError, match="auxiliary_residual"):
        coupler._build_canonical_point_features(
            raw_int,
            a_vals,
            b_vals,
            c_vals,
            rhs_raw,
            auxiliary_residual=bad_auxiliary_residual,
        )


def test_five_stencil_coupler_preserves_phi_plus_psi_for_nonzero_delta():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 5
    m = n + 2
    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    coupler = FiveStencilStencilMLPCoupler(
        CouplerConfig(enabled=True, residual_scale_init=0.1)
    ).to(dtype=torch.float64)

    final_conv = coupler.local_mlp[-1]
    assert isinstance(final_conv, nn.Conv2d)
    assert final_conv.bias is not None
    with torch.no_grad():
        final_conv.bias.fill_(0.25)

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)
    phi0 = raw_int[:, 0]
    psi0 = raw_int[:, 1].transpose(-1, -2)
    phi1 = out[:, 0]
    psi1 = out[:, 1].transpose(-1, -2)

    torch.testing.assert_close(
        phi1 + psi1,
        phi0 + psi0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert not torch.allclose(out, raw_int)


def test_five_stencil_coupler_gather_uses_only_cardinal_neighbors():
    torch.set_default_dtype(torch.float64)
    coupler = FiveStencilStencilMLPCoupler(
        CouplerConfig(enabled=True, padding="zero")
    ).to(dtype=torch.float64)
    q = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)

    q5 = coupler._gather_5_stencil(q)

    assert q5.shape == (1, 5, 3, 3)
    torch.testing.assert_close(
        q5[0, :, 1, 1],
        torch.tensor([4.0, 7.0, 1.0, 5.0, 3.0], dtype=torch.float64),
    )
    diagonal_values = {0.0, 2.0, 6.0, 8.0}
    assert diagonal_values.isdisjoint(set(q5[0, :, 1, 1].tolist()))


def test_coupling_net_with_disabled_coupler_matches_baseline():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 5
    m = n + 2
    coords = torch.randn(2, n, m, 2)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    rhs_tilde = torch.randn(bsz, 2, n, m)
    rhs_norm = torch.rand(bsz, 2, n) + 0.1

    torch.manual_seed(123)
    cfg_without = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
    )
    model_without = CouplingNet(cfg_without)

    torch.manual_seed(123)
    cfg_disabled = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=False),
    )
    model_disabled = CouplingNet(cfg_disabled)

    out_without = model_without(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )
    out_disabled = model_disabled(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )

    torch.testing.assert_close(out_disabled, out_without)


def test_coupling_net_enabled_coupler_is_initially_identity():
    torch.set_default_dtype(torch.float64)
    bsz = 2
    n = 5
    m = n + 2
    coords = torch.randn(2, n, m, 2)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    rhs_tilde = torch.randn(bsz, 2, n, m)
    rhs_norm = torch.rand(bsz, 2, n) + 0.1

    torch.manual_seed(123)
    cfg_disabled = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
    )
    model_disabled = CouplingNet(cfg_disabled)

    torch.manual_seed(123)
    cfg_enabled = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True, hidden_channels=8, depth=1),
    )
    model_enabled = CouplingNet(cfg_enabled)

    out_disabled = model_disabled(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )
    out_enabled = model_enabled(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )

    torch.testing.assert_close(out_enabled, out_disabled)


def test_freeze_main_train_coupler_only_leaves_only_coupler_trainable():
    torch.set_default_dtype(torch.float64)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True, hidden_channels=4, depth=1),
    )
    model = CouplingNet(cfg)
    assert model.coupler is not None

    freeze_main_train_coupler_only(model)

    coupler_param_ids = {id(param) for param in model.coupler.parameters()}
    assert coupler_param_ids
    for param in model.parameters():
        if id(param) in coupler_param_ids:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_freeze_main_train_coupler_only_rejects_missing_coupler():
    model = CouplingNet(
        CouplingModelConfig(branch_input_dim=5, coupler=CouplerConfig())
    )

    with pytest.raises(RuntimeError, match="model.coupler"):
        freeze_main_train_coupler_only(model)


def test_stage2_optimizer_contains_only_coupler_parameters(tmp_path):
    torch.set_default_dtype(torch.float64)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True, hidden_channels=4, depth=1),
    )
    model = CouplingNet(cfg)
    assert model.coupler is not None
    trainer = CouplingTrainer(
        model=model,
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            stage2=CouplingStage2Config(
                enabled=True,
                checkpoint_path="dummy_stage1.safetensors",
                lr=2.0e-3,
                weight_decay=0.25,
            ),
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        model_cfg=cfg,
    )

    optimizer = trainer._build_optimizer(trainer.config)

    opt_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    coupler_param_ids = {id(param) for param in model.coupler.parameters()}
    assert opt_param_ids == coupler_param_ids
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.25)


def test_coupling_net_passes_projected_input_and_auxiliary_residual_to_coupler():
    class _ZeroBranch(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device
            )

    class _ProjectedInputAssertingCoupler(nn.Module):
        def forward(
            self,
            raw_int: torch.Tensor,
            a_vals: torch.Tensor,
            b_vals: torch.Tensor,
            c_vals: torch.Tensor,
            rhs_raw: torch.Tensor,
            auxiliary_residual: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del a_vals, b_vals, c_vals
            assert auxiliary_residual is not None
            phi = raw_int[:, 0]
            psi = raw_int[:, 1].transpose(-1, -2)
            f = rhs_raw[:, 0, :, 1:-1]
            torch.testing.assert_close(phi + psi, f, rtol=1.0e-12, atol=1.0e-12)
            torch.testing.assert_close(
                auxiliary_residual,
                f,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            return raw_int

    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True),
    )
    model = CouplingNet(cfg)
    model.branch_a = _ZeroBranch(cfg.hidden_dim)
    model.branch_b = _ZeroBranch(cfg.hidden_dim)
    model.branch_c = _ZeroBranch(cfg.hidden_dim)
    model.branch_rhs = _ZeroBranch(cfg.hidden_dim)
    model.trunk = _ZeroBranch(cfg.hidden_dim)
    model.coupler = _ProjectedInputAssertingCoupler()

    coords = torch.zeros((2, n, m, 2), dtype=torch.float64)
    zeros = torch.zeros((1, 2, n, m), dtype=torch.float64)
    rhs_raw = zeros.clone()
    f = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.5, 1.5, 2.5]]],
        dtype=torch.float64,
    )
    rhs_raw[:, 0, :, 1:-1] = f
    rhs_norm = torch.ones((1, 2, n), dtype=torch.float64)

    out = model(
        coords=coords,
        a_vals=zeros,
        b_vals=zeros,
        c_vals=zeros,
        rhs_raw=rhs_raw,
        rhs_tilde=zeros,
        rhs_norm=rhs_norm,
    )

    torch.testing.assert_close(out[:, 0, :, 1:-1], 0.5 * f)
    torch.testing.assert_close(out[:, 1, :, 1:-1], (0.5 * f).transpose(-1, -2))


def test_coupling_net_post_projection_coupler_preserves_balance_with_nonzero_delta():
    class _ZeroBranch(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device
            )

    torch.set_default_dtype(torch.float64)
    n = 3
    m = n + 2
    cfg = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True, hidden_channels=8, depth=1),
    )
    model = CouplingNet(cfg)
    model.branch_a = _ZeroBranch(cfg.hidden_dim)
    model.branch_b = _ZeroBranch(cfg.hidden_dim)
    model.branch_c = _ZeroBranch(cfg.hidden_dim)
    model.branch_rhs = _ZeroBranch(cfg.hidden_dim)
    model.trunk = _ZeroBranch(cfg.hidden_dim)

    assert model.coupler is not None
    final_conv = model.coupler.local_mlp[-1]
    assert isinstance(final_conv, nn.Conv2d)
    assert final_conv.bias is not None
    with torch.no_grad():
        final_conv.bias.fill_(0.25)

    coords = torch.zeros((2, n, m, 2), dtype=torch.float64)
    zeros = torch.zeros((1, 2, n, m), dtype=torch.float64)
    rhs_raw = zeros.clone()
    f = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.5, 1.5, 2.5]]],
        dtype=torch.float64,
    )
    rhs_raw[:, 0, :, 1:-1] = f
    rhs_norm = torch.ones((1, 2, n), dtype=torch.float64)

    out, aux = model(
        coords=coords,
        a_vals=zeros,
        b_vals=zeros,
        c_vals=zeros,
        rhs_raw=rhs_raw,
        rhs_tilde=zeros,
        rhs_norm=rhs_norm,
        return_intermediates=True,
    )

    phi = out[:, 0, :, 1:-1]
    psi = out[:, 1, :, 1:-1].transpose(-1, -2)
    phi_projected = aux["projected_int"][:, 0]
    psi_projected = aux["projected_int"][:, 1].transpose(-1, -2)
    phi_coupled = aux["coupled_int"][:, 0]
    psi_coupled = aux["coupled_int"][:, 1].transpose(-1, -2)
    torch.testing.assert_close(phi_projected + psi_projected, f)
    torch.testing.assert_close(phi_coupled + psi_coupled, f)
    torch.testing.assert_close(out[:, :, :, 1:-1], aux["coupled_int"])
    assert not torch.allclose(aux["projected_int"], aux["coupled_int"])
    torch.testing.assert_close(phi + psi, f, rtol=1.0e-12, atol=1.0e-12)
    assert not torch.allclose(phi, 0.5 * f)


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


def test_coupling_model_forward_can_return_intermediates():
    torch.set_default_dtype(torch.float64)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
    )
    model = CouplingNet(cfg)
    bsz = 2
    n = 3
    m = n + 2
    coords = torch.zeros((2, n, m, 2), dtype=torch.float64)
    a_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    b_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    c_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    rhs_raw = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    output_flux, aux = model(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
        return_intermediates=True,
    )

    assert output_flux.shape == (bsz, 2, n, m)
    assert set(aux) == {
        "raw_int",
        "pre_projection_residual",
        "projected_int",
        "coupled_int",
    }
    assert aux["raw_int"].shape == (bsz, 2, n, n)
    assert aux["pre_projection_residual"].shape == (bsz, n, n)
    assert aux["projected_int"].shape == (bsz, 2, n, n)
    assert aux["coupled_int"].shape == (bsz, 2, n, n)
    torch.testing.assert_close(output_flux[:, :, :, 1:-1], aux["coupled_int"])


def test_coupling_model_detaches_coupler_inputs_but_keeps_aux_attached():
    class _InspectingCoupler(nn.Module):
        def __init__(self, expect_requires_grad: bool) -> None:
            super().__init__()
            self.expect_requires_grad = expect_requires_grad
            self.raw_requires_grad: bool | None = None
            self.residual_requires_grad: bool | None = None

        def forward(
            self,
            raw_int: torch.Tensor,
            a_vals: torch.Tensor,
            b_vals: torch.Tensor,
            c_vals: torch.Tensor,
            rhs_raw: torch.Tensor,
            auxiliary_residual: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del a_vals, b_vals, c_vals, rhs_raw
            assert auxiliary_residual is not None
            self.raw_requires_grad = raw_int.requires_grad
            self.residual_requires_grad = auxiliary_residual.requires_grad
            assert raw_int.requires_grad is self.expect_requires_grad
            assert auxiliary_residual.requires_grad is self.expect_requires_grad
            return raw_int

    torch.set_default_dtype(torch.float64)
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True),
    )
    bsz = 1
    n = 3
    m = n + 2
    coords = torch.randn(2, n, m, 2, dtype=torch.float64)
    a_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    b_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    c_vals = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    rhs_raw = torch.randn(bsz, 2, n, m, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    for detach, expect_requires_grad in ((True, False), (False, True)):
        model = CouplingNet(cfg)
        inspector = _InspectingCoupler(expect_requires_grad=expect_requires_grad)
        model.coupler = inspector
        _output_flux, aux = model(
            coords=coords,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            rhs_tilde=rhs_tilde,
            rhs_norm=rhs_norm,
            return_intermediates=True,
            detach_coupler_input=detach,
        )

        assert inspector.raw_requires_grad is expect_requires_grad
        assert inspector.residual_requires_grad is expect_requires_grad
        assert aux["projected_int"].requires_grad is True
        assert aux["pre_projection_residual"].requires_grad is True


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


def test_pad_interior_flux_like_preserves_interior_values(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    flux_int = torch.arange(18, dtype=torch.float64).reshape(1, 2, 3, 3)
    template_flux = torch.full((1, 2, 3, 5), 99.0, dtype=torch.float64)

    padded = trainer._pad_interior_flux_like(flux_int, template_flux)

    assert padded.shape == template_flux.shape
    torch.testing.assert_close(padded[:, :, :, 1:-1], flux_int)
    torch.testing.assert_close(padded[:, :, :, 0], torch.zeros((1, 2, 3)))
    torch.testing.assert_close(padded[:, :, :, -1], torch.zeros((1, 2, 3)))


def test_pad_interior_flux_like_rejects_incompatible_shape(tmp_path):
    trainer = CouplingTrainer(
        model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(epochs=1, batch_size=1),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )
    flux_int = torch.zeros((1, 2, 3, 4), dtype=torch.float64)
    template_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)

    with pytest.raises(ValueError, match="square flux_int"):
        trainer._pad_interior_flux_like(flux_int, template_flux)


def test_stage2_delta_helpers_recover_delta_and_ratio():
    projected_int = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    phi_p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
    psi_p = torch.tensor([[[0.5, 1.5], [2.5, 3.5]]], dtype=torch.float64)
    delta = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float64)
    projected_int[:, 0] = phi_p
    projected_int[:, 1] = psi_p.transpose(-1, -2)
    coupled_int = projected_int.clone()
    coupled_int[:, 0] = phi_p + delta
    coupled_int[:, 1] = (psi_p - delta).transpose(-1, -2)

    recovered_delta = CouplingTrainer._delta_from_intermediates(
        projected_int=projected_int,
        coupled_int=coupled_int,
    )
    ratio = CouplingTrainer._delta_norm_ratio(
        delta=recovered_delta,
        projected_int=projected_int,
        eps=0.0,
    )

    expected_ratio = torch.linalg.vector_norm(delta.reshape(1, -1), dim=1) / (
        torch.linalg.vector_norm(phi_p.reshape(1, -1), dim=1)
        + torch.linalg.vector_norm(psi_p.reshape(1, -1), dim=1)
    )
    torch.testing.assert_close(recovered_delta, delta)
    torch.testing.assert_close(ratio, expected_ratio.mean())


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


def test_hybrid_detach_loss_uses_projected_and_coupled_energy_weights(tmp_path):
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
    projected_int = torch.tensor(
        [
            [
                [[1.0, 0.5, 0.25], [0.25, 0.75, 1.25], [1.5, 0.5, 0.75]],
                [[0.25, 0.75, 1.0], [1.0, 0.25, 0.5], [0.5, 1.25, 0.25]],
            ]
        ],
        dtype=torch.float64,
    )
    coupled_int = projected_int.clone()
    coupled_int[:, 0] = coupled_int[:, 0] + 0.2
    coupled_int[:, 1] = coupled_int[:, 1] - 0.1
    model = _HybridFluxModel(projected_int=projected_int, coupled_int=coupled_int)
    green_kernel = torch.eye(5, dtype=torch.float64).expand(2, 3, 5, 5).clone()
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=False,
                energy_enabled=False,
                energy_weight=99.0,
                cross_enabled=False,
            ),
            hybrid_detach=CouplingHybridDetachConfig(
                enabled=True,
                projected_energy_weight=2.0,
                coupled_energy_weight=3.0,
                detach_coupler_input=True,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
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

    expected_hybrid = (
        2.0 * metrics["loss_energy_projected"] + 3.0 * metrics["loss_energy_coupled"]
    )
    assert "loss_energy_projected" in metrics
    assert "loss_energy_coupled" in metrics
    assert "loss_hybrid_detach" in metrics
    assert "energy_improvement" in metrics
    assert metrics["loss_hybrid_detach"] > 0.0
    assert metrics["loss_hybrid_detach"] == pytest.approx(expected_hybrid)
    assert metrics["loss_energy_consistency"] == pytest.approx(expected_hybrid)
    assert metrics["energy_improvement"] == pytest.approx(
        metrics["loss_energy_projected"] - metrics["loss_energy_coupled"]
    )
    assert loss.item() == pytest.approx(metrics["loss_hybrid_detach"])
    assert loss.item() != pytest.approx(99.0 * metrics["loss_hybrid_detach"])


def test_stage2_loss_uses_only_coupled_energy_and_reports_diagnostics(tmp_path):
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
    projected_int = torch.tensor(
        [
            [
                [[1.0, 0.5, 0.25], [0.25, 0.75, 1.25], [1.5, 0.5, 0.75]],
                [[0.25, 0.75, 1.0], [1.0, 0.25, 0.5], [0.5, 1.25, 0.25]],
            ]
        ],
        dtype=torch.float64,
    )
    coupled_int = projected_int.clone()
    coupled_int[:, 0] = coupled_int[:, 0] + 0.2
    coupled_int[:, 1] = coupled_int[:, 1] - 0.1
    model = _HybridFluxModel(projected_int=projected_int, coupled_int=coupled_int)
    green_kernel = torch.eye(5, dtype=torch.float64).expand(2, 3, 5, 5).clone()
    trainer = CouplingTrainer(
        model=model,  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_enabled=True,
                l2_weight=99.0,
                energy_enabled=True,
                energy_weight=99.0,
                cross_enabled=True,
                cross_weight=99.0,
            ),
            stage2=CouplingStage2Config(
                enabled=True,
                checkpoint_path="dummy_stage1.safetensors",
                coupled_energy_weight=2.5,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=green_kernel,
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

    expected_loss = 2.5 * metrics["loss_stage2_coupled_energy"]
    assert metrics["loss_stage2_projected_energy"] > 0.0
    assert metrics["loss_stage2_coupled_energy"] > 0.0
    assert metrics["stage2_delta_norm_ratio"] > 0.0
    assert metrics["stage2_energy_improvement"] == pytest.approx(
        metrics["loss_stage2_projected_energy"] - metrics["loss_stage2_coupled_energy"]
    )
    assert metrics["stage2_relative_improvement"] == pytest.approx(
        metrics["stage2_energy_improvement"]
        / (abs(metrics["loss_stage2_projected_energy"]) + 1.0e-12)
    )
    assert loss.item() == pytest.approx(expected_loss)


def test_stage2_best_checkpoint_policy_still_uses_validation_rel_sol(
    tmp_path, monkeypatch
):
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=True, hidden_channels=4, depth=1),
    )
    trainer = CouplingTrainer(
        model=CouplingNet(cfg),
        config=CouplingTrainingConfig(
            epochs=1,
            batch_size=1,
            best_rel_sol_checkpoint=CouplingBestRelSolCheckpointConfig(enabled=True),
            stage2=CouplingStage2Config(
                enabled=True,
                checkpoint_path="dummy_stage1.safetensors",
            ),
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        model_cfg=cfg,
    )
    saved_paths: list[Path] = []

    def _fake_save(path: Path) -> None:
        saved_paths.append(path)

    monkeypatch.setattr(trainer, "_save_checkpoint", _fake_save)

    best_rel_sol = trainer._maybe_save_best_rel_sol_checkpoint(
        {
            "rel_sol": 0.5,
            "loss_stage2_coupled_energy": 100.0,
        },
        best_rel_sol=1.0,
    )
    unchanged = trainer._maybe_save_best_rel_sol_checkpoint(
        {
            "rel_sol": 0.75,
            "loss_stage2_coupled_energy": 0.0,
        },
        best_rel_sol=best_rel_sol,
    )

    assert best_rel_sol == pytest.approx(0.5)
    assert unchanged == pytest.approx(0.5)
    assert saved_paths == [tmp_path / "coupling_model_adam_best_rel_sol.safetensors"]


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
