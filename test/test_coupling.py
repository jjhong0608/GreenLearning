from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.coupling_model import CouplingLineEncoder, CouplingNet
from greenonet.coupling_trainer import CouplingTrainer
from greenonet.config import (
    CouplingBestRelSolCheckpointConfig,
    CouplingLineEncoderConfig,
    CouplingLineEncoderHeadConfig,
    CouplingLossTermConfig,
    CouplingLossesConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
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
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, ap_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
        return self._projected_flux.clone()


class _TrainableFluxModel(nn.Module):
    def __init__(self, projected_flux: torch.Tensor) -> None:
        super().__init__()
        self.projected_flux = nn.Parameter(projected_flux.clone())

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, ap_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
        return self.projected_flux


class _RawSumFromRhsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        del coords, a_vals, ap_vals, b_vals, c_vals, rhs_tilde, rhs_norm
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
    ap_vals = torch.zeros((batch, 2, n_lines, m_points), dtype=torch.float64)
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
        ap_vals,
    )


def _losses_config(
    *,
    l2_enabled: bool = True,
    l2_weight: float = 1.0,
    l2_weight_mode: str = "manual",
    flux_enabled: bool = True,
    flux_weight: float = 1.0,
    flux_weight_mode: str = "manual",
    cross_enabled: bool = True,
    cross_weight: float = 1.0,
    cross_weight_mode: str = "manual",
) -> CouplingLossesConfig:
    return CouplingLossesConfig(
        l2_consistency=CouplingLossTermConfig(
            enabled=l2_enabled,
            weight=l2_weight,
            weight_mode=l2_weight_mode,
        ),
        flux_consistency=CouplingLossTermConfig(
            enabled=flux_enabled,
            weight=flux_weight,
            weight_mode=flux_weight_mode,
        ),
        cross_consistency=CouplingLossTermConfig(
            enabled=cross_enabled,
            weight=cross_weight,
            weight_mode=cross_weight_mode,
        ),
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


def test_coupling_line_encoder_shape_matches_hidden_dim():
    encoder = CouplingLineEncoder(
        config=CouplingLineEncoderConfig(
            conv_channels=[32, 64, 64],
            dilations=[1, 2, 4],
            mlp_head=CouplingLineEncoderHeadConfig(
                depth=2,
                hidden_dim=16,
                activation="rational",
            ),
        ),
        output_dim=8,
        fallback_hidden_dim=8,
        fallback_depth=2,
        fallback_activation="rational",
        fallback_use_bias=True,
        fallback_dropout=0.0,
    ).to(dtype=torch.float64)
    line = torch.randn(6, 9, dtype=torch.float64)
    pos = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64).view(1, 1, 9)
    db = torch.minimum(pos, 1.0 - pos)

    latent = encoder(line, pos, db)

    assert latent.shape == (6, 8)


def test_coupling_model_construction_from_nested_line_encoder_config():
    cfg = CouplingModelConfig(
        branch_input_dim=5,
        hidden_dim=8,
        depth=2,
        line_encoder=CouplingLineEncoderConfig(
            type="cnn1d",
            in_channels=3,
            include_position=True,
            include_boundary_distance=True,
            conv_channels=[32, 64, 64],
            kernel_size=5,
            dilations=[1, 2, 4],
            pooling="meanmax",
            activation="rational",
            mlp_head=CouplingLineEncoderHeadConfig(
                depth=1,
                hidden_dim=16,
                activation="rational",
                use_bias=True,
                dropout=0.0,
            ),
        ),
    )

    model = CouplingNet(cfg)

    assert isinstance(model.branch_a, CouplingLineEncoder)
    assert isinstance(model.branch_ap, CouplingLineEncoder)
    assert model.branch_fuser.in_features == 5 * cfg.hidden_dim


def test_coupling_model_forward():
    cfg = CouplingModelConfig(branch_input_dim=3, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    coords = torch.zeros((2, 1, 3, 2))
    kappa = torch.randn(1, 2, 1, 3)
    ap_vals = torch.randn(1, 2, 1, 3)
    b_vals = torch.randn(1, 2, 1, 3)
    c_vals = torch.randn(1, 2, 1, 3)
    rhs_raw = torch.randn(1, 2, 1, 3)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1, keepdim=False).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)
    projected_flux = model(
        coords=coords,
        a_vals=kappa,
        ap_vals=ap_vals,
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


def test_coupling_model_backprop_through_all_branch_encoders():
    cfg = CouplingModelConfig(
        branch_input_dim=3,
        hidden_dim=8,
        depth=2,
    )
    model = CouplingNet(cfg)

    coords = torch.zeros((2, 1, 3, 2), dtype=torch.float64)
    a_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    ap_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    b_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    c_vals = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_raw = torch.randn(1, 2, 1, 3, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1, keepdim=False).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    projected_flux = model(
        coords=coords,
        a_vals=a_vals,
        ap_vals=ap_vals,
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
    assert _has_grad(model.branch_ap)
    assert _has_grad(model.branch_b)
    assert _has_grad(model.branch_c)
    assert _has_grad(model.branch_rhs)
    assert _has_grad(model.branch_fuser)
    assert _has_grad(model.trunk)


def test_coupling_balance_projection_uses_fixed_half_split():
    class _ZeroBranch(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device
            )

    cfg = CouplingModelConfig(branch_input_dim=5, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    model.branch_a = _ZeroBranch(cfg.hidden_dim)
    model.branch_ap = _ZeroBranch(cfg.hidden_dim)
    model.branch_b = _ZeroBranch(cfg.hidden_dim)
    model.branch_c = _ZeroBranch(cfg.hidden_dim)
    model.branch_rhs = _ZeroBranch(cfg.hidden_dim)
    model.branch_fuser = nn.Linear(5 * cfg.hidden_dim, cfg.hidden_dim, bias=False)
    nn.init.zeros_(model.branch_fuser.weight)
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
        ap_vals=zeros,
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


def test_coupling_model_forward_accepts_zero_b_and_c():
    cfg = CouplingModelConfig(branch_input_dim=5, hidden_dim=8, depth=2)
    model = CouplingNet(cfg)
    coords = torch.zeros((2, 3, 5, 2), dtype=torch.float64)
    coords[0, :, :, 0] = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    coords[1, :, :, 1] = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    a_vals = torch.randn(1, 2, 3, 5, dtype=torch.float64)
    ap_vals = torch.randn(1, 2, 3, 5, dtype=torch.float64)
    b_vals = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    c_vals = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    rhs_raw = torch.randn(1, 2, 3, 5, dtype=torch.float64)
    rhs_norm = torch.linalg.norm(rhs_raw, dim=-1).clamp_min(1e-6)
    rhs_tilde = rhs_raw / rhs_norm.unsqueeze(-1)

    projected_flux = model(
        coords=coords,
        a_vals=a_vals,
        ap_vals=ap_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
    )

    assert projected_flux.shape == (1, 2, 3, 5)


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
    assert "loss_flux_consistency" in metrics
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
        ap_vals,
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
        ap_vals,
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
        ap_vals,
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
                flux_enabled=False,
                flux_weight=1.0,
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
        ap_vals,
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
        ap_vals,
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
                flux_enabled=True,
                flux_weight=1.0,
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
        ap_vals,
    )
    loss.backward()

    assert metrics["loss_flux_consistency"] > 0.0
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
        ap_vals,
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
                flux_enabled=True,
                flux_weight=2.0,
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
        ap_vals,
    )

    expected = (
        0.5 * metrics["loss_l2_consistency"]
        + 2.0 * metrics["loss_flux_consistency"]
        + 3.0 * metrics["loss_cross_consistency"]
    )
    assert loss.item() == pytest.approx(expected)
    assert metrics["weight_l2_effective"] == pytest.approx(0.5)
    assert metrics["weight_flux_effective"] == pytest.approx(2.0)
    assert metrics["weight_cross_effective"] == pytest.approx(3.0)


def test_auto_operator_flux_weight_uses_base_l2_weight(tmp_path):
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
        ap_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    trainer = CouplingTrainer(
        model=_DummyFluxModel(projected_flux),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_weight=2.0,
                flux_weight=99.0,
                flux_weight_mode="auto_operator",
                cross_enabled=False,
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    w_l2, w_flux, w_cross = trainer._compute_effective_loss_weights(
        x_axis=coords[0, 0, :, 0],
        y_axis=coords[1, 0, :, 1],
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
    )

    assert w_l2 == pytest.approx(2.0)
    assert w_flux == pytest.approx(0.25)
    assert w_cross == pytest.approx(0.0)

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
        ap_vals,
    )
    expected = 2.0 * metrics["loss_l2_consistency"] + 0.25 * metrics["loss_flux_consistency"]
    assert loss.item() == pytest.approx(expected)


def test_auto_operator_cross_weight_includes_diffusion_advection_and_reaction(tmp_path):
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
        ap_vals,
    ) = _make_step_loss_inputs(batch=1, n_lines=3, m_points=5)
    a_vals[:] = 1.0
    b_vals[:] = 2.0
    c_vals[:] = 3.0
    projected_flux = torch.zeros((1, 2, 3, 5), dtype=torch.float64)
    projected_flux[:, 0, :, 1:-1] = 1.0
    projected_flux[:, 1, :, 1:-1] = 0.25
    trainer = CouplingTrainer(
        model=_DummyFluxModel(projected_flux),  # type: ignore[arg-type]
        config=CouplingTrainingConfig(
            losses=_losses_config(
                l2_weight=2.0,
                flux_enabled=False,
                cross_weight=77.0,
                cross_weight_mode="auto_operator",
            ),
            epochs=1,
            batch_size=1,
        ),
        work_dir=tmp_path,
        green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
    )

    _w_l2, w_flux, w_cross = trainer._compute_effective_loss_weights(
        x_axis=coords[0, 0, :, 0],
        y_axis=coords[1, 0, :, 1],
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
    )
    expected_gain = 6.0 / (0.25**4) + 4.0 / (2.0 * 0.25**2) + 9.0

    assert w_flux == pytest.approx(0.0)
    assert w_cross == pytest.approx(2.0 / expected_gain)

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
        ap_vals,
    )
    expected = 2.0 * metrics["loss_l2_consistency"] + (2.0 / expected_gain) * metrics[
        "loss_cross_consistency"
    ]
    assert loss.item() == pytest.approx(expected)


def test_trainer_rejects_auto_operator_mode_for_l2_weight(tmp_path):
    with pytest.raises(ValueError, match="l2_consistency.weight_mode"):
        CouplingTrainer(
            model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
            config=CouplingTrainingConfig(
                losses=_losses_config(l2_weight_mode="auto_operator"),
                epochs=1,
                batch_size=1,
            ),
            work_dir=tmp_path,
            green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        )


def test_trainer_requires_positive_l2_base_weight_for_auto_operator_losses(tmp_path):
    with pytest.raises(ValueError, match="l2_consistency.weight must be > 0"):
        CouplingTrainer(
            model=_DummyFluxModel(torch.zeros((1, 2, 3, 5), dtype=torch.float64)),  # type: ignore[arg-type]
            config=CouplingTrainingConfig(
                losses=_losses_config(
                    l2_weight=0.0,
                    flux_weight_mode="auto_operator",
                    cross_enabled=False,
                ),
                epochs=1,
                batch_size=1,
            ),
            work_dir=tmp_path,
            green_kernel=torch.ones((2, 3, 5, 5), dtype=torch.float64),
        )


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
        ap_vals,
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
            ap_vals[0],
        )
    ]

    metrics = trainer.evaluate(dataset)  # type: ignore[arg-type]

    assert "loss" in metrics
    assert "loss_l2_consistency" in metrics
    assert "loss_flux_consistency" in metrics
    assert "loss_cross_consistency" in metrics
    assert "weight_l2_effective" in metrics
    assert "weight_flux_effective" in metrics
    assert "weight_cross_effective" in metrics
