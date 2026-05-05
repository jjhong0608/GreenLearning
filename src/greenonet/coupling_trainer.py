from __future__ import annotations

from math import cos, pi
from pathlib import Path
from typing import List

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad

from greenonet.compile_utils import maybe_compile_model, model_state_dict_for_save
from greenonet.coupling_model import CouplingNet
from greenonet.config import (
    CouplingModelConfig,
    CouplingTrainingConfig,
)
from greenonet.logging_mixin import LoggingMixin
from greenonet.numerics import integrate, line_operator_fd, uniform_spacing
from greenonet.visualizer import LossVisualizer
from greenonet.io import (
    save_model_with_config,
    save_state_dict_safetensors,
)


CouplingBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class CouplingTrainer(LoggingMixin):
    """Trainer for CouplingNet using Green kernel integrals."""

    _METRIC_KEYS: tuple[str, ...] = (
        "loss",
        "loss_l2_consistency",
        "loss_energy_consistency",
        "loss_cross_consistency",
        "rel_flux",
        "rel_sol",
        "source_lift_corr_g_f",
        "source_lift_rel_diff_g_f",
        "source_lift_g_rms",
    )

    def __init__(
        self,
        model: CouplingNet,
        config: CouplingTrainingConfig,
        work_dir: Path | str,
        green_kernel: torch.Tensor,
        model_cfg: CouplingModelConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.model_cfg = model_cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(logger_name="CouplingTrainer", work_dir=self.work_dir)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model = maybe_compile_model(
            self.model,
            self.config.compile,
            logger=self.logger,
            model_name="CouplingNet",
        )
        self.green_kernel = green_kernel.to(self.device)  # (2, n, m, m)
        self.loss_history: List[float] = []
        self.stage_loss_history: dict[str, List[float]] = {}

    def _integrate(
        self, green: torch.Tensor, values: torch.Tensor, x_axis: torch.Tensor
    ) -> torch.Tensor:
        """
        green: (n, m, m)
        values: (B, n, m)
        x_axis: (m,)
        returns: (B, n, m)
        """
        weighted = values.unsqueeze(-2) * green.unsqueeze(0)  # (B, n, m, m)
        return integrate(
            weighted,
            x=x_axis,
            dim=-1,
            rule=self.config.integration_rule,
        )

    @staticmethod
    def _face_average_arithmetic(values: torch.Tensor, dim: int) -> torch.Tensor:
        dim = dim % values.dim()
        if values.shape[dim] < 2:
            raise ValueError(
                "Need at least two nodal samples to compute face averages."
            )
        left = values.narrow(dim, 0, values.shape[dim] - 1)
        right = values.narrow(dim, 1, values.shape[dim] - 1)
        return 0.5 * (left + right)

    def _energy_consistency_loss(
        self,
        u_phi_x: torch.Tensor,
        u_psi_y: torch.Tensor,
        a_vals: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> torch.Tensor:
        if u_phi_x.shape != u_psi_y.transpose(-1, -2).shape:
            raise ValueError(
                "u_phi_x and u_psi_y must define the same common grid after transpose."
            )
        if a_vals.dim() != 4 or a_vals.shape[1] != 2:
            raise ValueError("a_vals must have shape (B, 2, n_lines, m_points).")

        u_psi_common = u_psi_y.transpose(-1, -2)
        r = u_phi_x - u_psi_common
        if r.dim() != 3:
            raise ValueError(
                f"expected residual r to be 3D (B, m, m), got {tuple(r.shape)}"
            )

        hx = uniform_spacing(x_axis)
        hy = uniform_spacing(y_axis)

        dr_dx_face = (r[..., :, 1:] - r[..., :, :-1]) / hx
        dr_dy_face = (r[..., 1:, :] - r[..., :-1, :]) / hy

        dr_dx_face = dr_dx_face[:, 1:-1, :]
        dr_dy_face = dr_dy_face[:, :, 1:-1]

        a_x_nodes = a_vals[:, 0]
        a_y_nodes_common = a_vals[:, 1].transpose(-1, -2)
        a_x_face = self._face_average_arithmetic(a_x_nodes, dim=-1)
        a_y_face = self._face_average_arithmetic(a_y_nodes_common, dim=-2)

        if a_x_face.shape != dr_dx_face.shape:
            raise ValueError(
                f"x-face coefficient shape {tuple(a_x_face.shape)} does not match "
                f"x-face derivative shape {tuple(dr_dx_face.shape)}."
            )
        if a_y_face.shape != dr_dy_face.shape:
            raise ValueError(
                f"y-face coefficient shape {tuple(a_y_face.shape)} does not match "
                f"y-face derivative shape {tuple(dr_dy_face.shape)}."
            )

        # Physical energy density: a_face * |D_face r|^2, not |a_face D_face r|^2.
        density_x = a_x_face * dr_dx_face.pow(2)
        density_y = a_y_face * dr_dy_face.pow(2)

        loss_x_per_batch = density_x.sum(dim=(-1, -2)) * hx * hy
        loss_y_per_batch = density_y.sum(dim=(-1, -2)) * hx * hy
        return (loss_x_per_batch + loss_y_per_batch).mean()

    def _represented_solutions_from_flux(
        self,
        flux: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = self.green_kernel[0]  # (n,m,m)
        gy = self.green_kernel[1]

        phi_x = flux[:, 0].clone()
        psi_y = flux[:, 1].clone()

        phi_x[..., 0] = 0.0
        phi_x[..., -1] = 0.0
        psi_y[..., 0] = 0.0
        psi_y[..., -1] = 0.0
        u_phi_x_raw = self._integrate(gx, phi_x, x_axis)
        u_psi_y_raw = self._integrate(gy, psi_y, y_axis)

        u_phi_x = pad(
            u_phi_x_raw[..., 1:-1],
            pad=(1, 1, 1, 1),
            mode="constant",
            value=0.0,
        )  # homogeneous Dirichlet condition
        u_psi_y = pad(
            u_psi_y_raw[..., 1:-1],
            pad=(1, 1, 1, 1),
            mode="constant",
            value=0.0,
        )  # homogeneous Dirichlet condition
        return phi_x, psi_y, u_phi_x, u_psi_y

    def _cross_consistency_loss(
        self,
        u_phi_x: torch.Tensor,
        u_psi_y: torch.Tensor,
        phi_x: torch.Tensor,
        psi_y: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> torch.Tensor:
        if a_vals.dim() != 4 or a_vals.shape[1] != 2:
            raise ValueError("a_vals must have shape (B, 2, n_lines, m_points).")
        if b_vals.shape != a_vals.shape or c_vals.shape != a_vals.shape:
            raise ValueError("b_vals and c_vals must match a_vals.")

        x_inner = x_axis[1:-1]
        y_inner = y_axis[1:-1]
        if not self._supports_auxiliary_quadrature(
            x_inner
        ) or not self._supports_auxiliary_quadrature(y_inner):
            return torch.zeros((), dtype=u_phi_x.dtype, device=u_phi_x.device)

        u_psi_x_view = u_psi_y.transpose(-1, -2)
        lx_u_psi = line_operator_fd(
            u_lines=u_psi_x_view[:, 1:-1, :],
            a_lines=a_vals[:, 0],
            b_lines=b_vals[:, 0],
            c_lines=c_vals[:, 0],
            axis_coords=x_axis,
        )
        phi_target = phi_x[:, :, 1:-1]
        if lx_u_psi.shape != phi_target.shape:
            raise ValueError("Lx(u_psi^(y)) and phi must have matching shapes.")
        cross_x_res = lx_u_psi - phi_target
        cross_x_sq_int = integrate(
            cross_x_res.pow(2), x=x_inner, dim=-1, rule=self.config.integration_rule
        )
        loss_cross_x = integrate(
            cross_x_sq_int, x=y_inner, dim=-1, rule=self.config.integration_rule
        ).mean()

        u_phi_y_view = u_phi_x.transpose(-1, -2)
        ly_u_phi = line_operator_fd(
            u_lines=u_phi_y_view[:, 1:-1, :],
            a_lines=a_vals[:, 1],
            b_lines=b_vals[:, 1],
            c_lines=c_vals[:, 1],
            axis_coords=y_axis,
        )
        psi_target = psi_y[:, :, 1:-1]
        if ly_u_phi.shape != psi_target.shape:
            raise ValueError("Ly(u_phi^(x)) and psi must have matching shapes.")
        cross_y_res = ly_u_phi - psi_target
        cross_y_sq_int = integrate(
            cross_y_res.pow(2), x=y_inner, dim=-1, rule=self.config.integration_rule
        )
        loss_cross_y = integrate(
            cross_y_sq_int, x=x_inner, dim=-1, rule=self.config.integration_rule
        ).mean()
        return loss_cross_x + loss_cross_y

    @staticmethod
    def _effective_weight(enabled: bool, weight: float) -> float:
        return weight if enabled else 0.0

    @classmethod
    def _metric_accumulator(cls, fill_value: float) -> dict[str, float]:
        return {key: fill_value for key in cls._METRIC_KEYS}

    def _source_lift_metrics(self, rhs_raw: torch.Tensor) -> dict[str, float]:
        nan = float("nan")
        default = {
            "source_lift_corr_g_f": nan,
            "source_lift_rel_diff_g_f": nan,
            "source_lift_g_rms": nan,
        }
        module = getattr(self.model, "_orig_mod", self.model)
        diagnostics_fn = getattr(module, "source_lift_diagnostics", None)
        if not callable(diagnostics_fn):
            return default
        with torch.no_grad():
            diagnostics = diagnostics_fn(rhs_raw)
        if not diagnostics:
            return default
        values = dict(default)
        values.update(
            {
                key: float(value.detach().item())
                for key, value in diagnostics.items()
                if key in default
            }
        )
        return values

    def _source_lift_enabled(self) -> bool:
        module = getattr(self.model, "_orig_mod", self.model)
        return getattr(module, "source_stencil_lift", None) is not None

    def _supports_auxiliary_quadrature(self, axis_coords: torch.Tensor) -> bool:
        samples = int(axis_coords.numel())
        if self.config.integration_rule == "simpson":
            return samples >= 3 and samples % 2 == 1
        return samples >= 2

    def _relative_l2_integral(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_axis: torch.Tensor,
        eps: float = 1e-12,
    ) -> float:
        num = integrate(
            (pred - target).pow(2),
            x=x_axis,
            dim=-1,
            rule=self.config.integration_rule,
        )
        num = integrate(num, x=x_axis, dim=-1, rule=self.config.integration_rule).mean()
        den = integrate(
            target.pow(2), x=x_axis, dim=-1, rule=self.config.integration_rule
        )
        den = (
            integrate(den, x=x_axis, dim=-1, rule=self.config.integration_rule)
            .mean()
            .clamp_min(eps)
        )
        return float(torch.sqrt(num / den).item())

    def _step_loss(
        self,
        coords: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
        sol: torch.Tensor,
        flux_target: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # coords: (2, n, m, 2), rhs/sol/flux_target: (B,2,n,m)
        x_axis = coords[0, 0, :, 0].to(self.device)
        y_axis = coords[1, 0, :, 1].to(self.device)

        rhs_raw = rhs_raw.to(self.device)
        rhs_tilde = rhs_tilde.to(self.device)
        rhs_norm = rhs_norm.to(self.device)
        sol = sol.to(self.device)
        flux_target = flux_target.to(self.device)
        a_vals = a_vals.to(self.device)
        b_vals = b_vals.to(self.device)
        c_vals = c_vals.to(self.device)
        coords = coords.to(self.device)

        pred_flux = self.model(
            coords=coords,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            rhs_tilde=rhs_tilde,
            rhs_norm=rhs_norm,
        )  # (B,2,n,m)

        phi_x, psi_y, u_phi_x, u_psi_y = self._represented_solutions_from_flux(
            flux=pred_flux,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        pred_flux_target = torch.stack((phi_x, psi_y), dim=1)
        pred_flux_target = pad(
            pred_flux_target[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0
        )
        flux_target = pad(
            flux_target[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0
        )
        sol = pad(sol[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)

        cons_res = u_phi_x - u_psi_y.transpose(-1, -2)
        loss_cons = integrate(
            cons_res.pow(2), x=x_axis, dim=-1, rule=self.config.integration_rule
        )
        loss_cons = integrate(
            loss_cons, x=x_axis, dim=-1, rule=self.config.integration_rule
        ).mean()
        loss_energy_final = self._energy_consistency_loss(
            u_phi_x=u_phi_x,
            u_psi_y=u_psi_y,
            a_vals=a_vals,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        loss_cross_consistency = self._cross_consistency_loss(
            u_phi_x=u_phi_x,
            u_psi_y=u_psi_y,
            phi_x=phi_x,
            psi_y=psi_y,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        l2_cfg = self.config.losses.l2_consistency
        energy_cfg = self.config.losses.energy_consistency
        cross_cfg = self.config.losses.cross_consistency
        w_l2 = self._effective_weight(l2_cfg.enabled, l2_cfg.weight)
        w_energy = self._effective_weight(energy_cfg.enabled, energy_cfg.weight)
        w_cross = self._effective_weight(cross_cfg.enabled, cross_cfg.weight)

        loss_energy_consistency = loss_energy_final

        loss = torch.zeros((), dtype=loss_cons.dtype, device=loss_cons.device)
        if w_l2 != 0.0:
            loss = loss + w_l2 * loss_cons
        if w_energy != 0.0:
            loss = loss + w_energy * loss_energy_consistency
        if w_cross != 0.0:
            loss = loss + w_cross * loss_cross_consistency

        source_lift_metrics = self._source_lift_metrics(rhs_raw)
        metrics = {
            "loss": float(loss.detach().item()),
            "loss_l2_consistency": float(loss_cons.detach().item()),
            "loss_energy_consistency": float(loss_energy_consistency.detach().item()),
            "loss_cross_consistency": float(loss_cross_consistency.detach().item()),
            "rel_flux": self._relative_l2_integral(
                pred_flux_target.detach(), flux_target, x_axis
            ),
            "rel_sol": self._relative_l2_integral(
                torch.stack((u_phi_x, u_psi_y), dim=1).detach(),
                torch.stack((sol[:, 0], sol[:, 1]), dim=1),
                x_axis,
            ),
            **source_lift_metrics,
        }
        return loss, metrics

    def _make_loader(
        self, dataset: Dataset[CouplingBatch], shuffle: bool = True
    ) -> DataLoader[CouplingBatch]:
        from greenonet.coupling_data import coupling_collate_fn

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=coupling_collate_fn,
            pin_memory=True,
        )

    def _save_checkpoint(self, path: Path) -> None:
        if self.model_cfg is not None:
            save_model_with_config(self.model, self.model_cfg, path, self.logger)
        else:
            save_state_dict_safetensors(
                model_state_dict_for_save(self.model), path, self.logger
            )

    def _validate_periodic_checkpoint_config(self) -> None:
        checkpoint_cfg = self.config.periodic_checkpoint
        if checkpoint_cfg.enabled and checkpoint_cfg.every_epochs <= 0:
            raise ValueError(
                "coupling_training.periodic_checkpoint.every_epochs must be > 0 when periodic checkpointing is enabled."
            )

    def _periodic_checkpoint_path(self, epoch: int) -> Path:
        return self.work_dir / f"coupling_model_epoch_{epoch:04d}.safetensors"

    def _adam_best_rel_sol_checkpoint_path(self) -> Path:
        return self.work_dir / "coupling_model_adam_best_rel_sol.safetensors"

    def _maybe_save_periodic_checkpoint(self, epoch: int) -> None:
        checkpoint_cfg = self.config.periodic_checkpoint
        if not checkpoint_cfg.enabled:
            return
        if epoch % checkpoint_cfg.every_epochs != 0:
            return
        self._save_checkpoint(self._periodic_checkpoint_path(epoch))

    def _maybe_save_best_rel_sol_checkpoint(
        self,
        val_metrics: dict[str, float],
        best_rel_sol: float,
    ) -> float:
        if not self.config.best_rel_sol_checkpoint.enabled:
            return best_rel_sol
        val_rel_sol = val_metrics.get("rel_sol")
        if not isinstance(val_rel_sol, float) or val_rel_sol != val_rel_sol:
            return best_rel_sol
        if val_rel_sol >= best_rel_sol:
            return best_rel_sol
        self._save_checkpoint(self._adam_best_rel_sol_checkpoint_path())
        return val_rel_sol

    def _optimization_config(self) -> CouplingTrainingConfig:
        return self.config

    def _build_optimizer(self, optimization_cfg: CouplingTrainingConfig) -> optim.Adam:
        return optim.Adam(self.model.parameters(), lr=optimization_cfg.learning_rate)

    def _build_scheduler(
        self,
        optimization_cfg: CouplingTrainingConfig,
        optimizer: optim.Optimizer,
        total_epochs: int,
    ) -> optim.lr_scheduler.LambdaLR | None:
        if not optimization_cfg.use_lr_schedule:
            return None
        warmup_epochs = max(optimization_cfg.warmup_epochs, 0)
        if warmup_epochs >= total_epochs:
            warmup_epochs = total_epochs - 1

        def lr_lambda(epoch_idx: int) -> float:
            base_lr = optimization_cfg.learning_rate
            min_lr = optimization_cfg.min_lr
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            if total_epochs <= warmup_epochs + 1:
                return float(min_lr / base_lr)
            progress = float(epoch_idx - warmup_epochs) / float(
                total_epochs - warmup_epochs - 1
            )
            cosine = 0.5 * (1.0 + cos(pi * progress))
            return float((min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _evaluate_loader(
        self,
        loader: DataLoader[CouplingBatch],
    ) -> dict[str, float]:
        with torch.no_grad():
            accum = self._metric_accumulator(0.0)
            batch_count = 0
            for (
                coords,
                rhs_raw,
                rhs_tilde,
                rhs_norm,
                sol,
                flux,
                a_vals,
                b_vals,
                c_vals,
                ap,
            ) in loader:
                del ap
                _, metrics = self._step_loss(
                    coords,
                    rhs_raw,
                    rhs_tilde,
                    rhs_norm,
                    sol,
                    flux,
                    a_vals,
                    b_vals,
                    c_vals,
                )
                for key in accum:
                    accum[key] += metrics.get(key, 0.0)
                batch_count += 1
        return {key: value / max(batch_count, 1) for key, value in accum.items()}

    def _run_training_phase(
        self,
        train_dataset: Dataset[CouplingBatch],
        val_dataset: Dataset[CouplingBatch] | None,
        epochs: int,
    ) -> None:
        self._validate_periodic_checkpoint_config()
        self.model.train()
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = (
            self._make_loader(val_dataset, shuffle=False)
            if val_dataset is not None
            else None
        )
        optimization_cfg = self._optimization_config()
        optimizer = self._build_optimizer(optimization_cfg)
        scheduler = self._build_scheduler(optimization_cfg, optimizer, max(epochs, 1))
        phase_history: List[float] = []
        best_rel_sol = float("inf")

        for epoch in range(1, epochs + 1):
            epoch_losses: List[float] = []
            accum = self._metric_accumulator(0.0)
            batch_count = 0
            for (
                coords,
                rhs_raw,
                rhs_tilde,
                rhs_norm,
                sol,
                flux,
                a_vals,
                b_vals,
                c_vals,
                ap,
            ) in train_loader:
                del ap
                optimizer.zero_grad()
                loss, metrics = self._step_loss(
                    coords,
                    rhs_raw,
                    rhs_tilde,
                    rhs_norm,
                    sol,
                    flux,
                    a_vals,
                    b_vals,
                    c_vals,
                )
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.detach().item())
                for key in accum:
                    accum[key] += metrics.get(key, 0.0)
                batch_count += 1

            mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            train_metrics = {
                key: value / max(batch_count, 1) for key, value in accum.items()
            }
            self.loss_history.append(mean_loss)
            phase_history.append(mean_loss)

            val_metrics = self._metric_accumulator(float("nan"))
            if val_loader is not None:
                val_metrics = self._evaluate_loader(val_loader)
                best_rel_sol = self._maybe_save_best_rel_sol_checkpoint(
                    val_metrics, best_rel_sol
                )

            if epoch % self.config.log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                l2_cfg = self.config.losses.l2_consistency
                energy_cfg = self.config.losses.energy_consistency
                cross_cfg = self.config.losses.cross_consistency
                log_message = (
                    "epoch %s | train loss=%.4e l2_cons=%.4e energy_cons=%.4e "
                    "cross_cons=%.4e rel_flux=%.4e rel_sol=%.4e | "
                    "w_l2=%.4e on_l2=%s w_energy=%.4e on_energy=%s "
                    "w_cross=%.4e on_cross=%s | lr=%.4e | val loss=%.4e "
                    "l2_cons=%.4e energy_cons=%.4e cross_cons=%.4e "
                    "rel_flux=%.4e rel_sol=%.4e"
                )
                log_args: tuple[object, ...] = (
                    epoch,
                    train_metrics["loss"],
                    train_metrics["loss_l2_consistency"],
                    train_metrics["loss_energy_consistency"],
                    train_metrics["loss_cross_consistency"],
                    train_metrics["rel_flux"],
                    train_metrics["rel_sol"],
                    l2_cfg.weight,
                    l2_cfg.enabled,
                    energy_cfg.weight,
                    energy_cfg.enabled,
                    cross_cfg.weight,
                    cross_cfg.enabled,
                    current_lr,
                    val_metrics["loss"],
                    val_metrics["loss_l2_consistency"],
                    val_metrics["loss_energy_consistency"],
                    val_metrics["loss_cross_consistency"],
                    val_metrics["rel_flux"],
                    val_metrics["rel_sol"],
                )
                if self._source_lift_enabled():
                    log_message = (
                        log_message + " | source_lift train_corr=%.4e "
                        "train_rel_diff=%.4e train_g_rms=%.4e val_corr=%.4e "
                        "val_rel_diff=%.4e val_g_rms=%.4e"
                    )
                    log_args = log_args + (
                        train_metrics["source_lift_corr_g_f"],
                        train_metrics["source_lift_rel_diff_g_f"],
                        train_metrics["source_lift_g_rms"],
                        val_metrics["source_lift_corr_g_f"],
                        val_metrics["source_lift_rel_diff_g_f"],
                        val_metrics["source_lift_g_rms"],
                    )
                self.logger.info(
                    log_message,
                    *log_args,
                )
            self._maybe_save_periodic_checkpoint(epoch)
            if scheduler is not None:
                scheduler.step()

        adam_model_path = self.work_dir / "coupling_model_adam.safetensors"
        self._save_checkpoint(adam_model_path)

        self.stage_loss_history["single"] = phase_history
        phase_curve_path = self.work_dir / "coupling_loss_curve.html"
        if phase_history:
            LossVisualizer.save_loss_curve(
                losses=phase_history,
                output_path=phase_curve_path,
                logger=self.logger,
            )

        final_path = self.work_dir / "coupling_model.safetensors"
        self._save_checkpoint(final_path)

    def train(
        self,
        train_dataset: Dataset[CouplingBatch],
        val_dataset: Dataset[CouplingBatch] | None = None,
    ) -> None:
        self.loss_history = []
        self.stage_loss_history = {}
        epochs = self.config.epochs
        self._run_training_phase(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
        )

    def evaluate(
        self,
        dataset: Dataset[CouplingBatch],
    ) -> dict[str, float]:
        self.model.eval()
        loader = self._make_loader(dataset, shuffle=False)
        accum = self._metric_accumulator(0.0)
        batch_count = 0
        with torch.no_grad():
            for (
                coords,
                rhs_raw,
                rhs_tilde,
                rhs_norm,
                sol,
                flux,
                a_vals,
                b_vals,
                c_vals,
                ap,
            ) in loader:
                del ap
                _loss, metrics = self._step_loss(
                    coords,
                    rhs_raw,
                    rhs_tilde,
                    rhs_norm,
                    sol,
                    flux,
                    a_vals,
                    b_vals,
                    c_vals,
                )
                for key in accum:
                    accum[key] += metrics.get(key, 0.0)
                batch_count += 1
        return {key: value / max(batch_count, 1) for key, value in accum.items()}
