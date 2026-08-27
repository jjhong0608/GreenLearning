from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import (
    build_boundary_energy_context,
    canonical_complex_energy_loss,
)
from greenonet.complex_projection_response_audit import (
    ComplexProjectionResponseAudit,
    ProjectionTransitionEdges,
)
from greenonet.complex_symmetric_tangent_audit import (
    ClosedLoopTangentBatchDiagnostics,
    TangentBatchEvaluation,
    TangentMethod,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    SymmetricTangentGreenResponseContext,
    SymmetricTangentGreenResponseContextCache,
    matrix_free_krylov_subspace_step,
)
from greenonet.complex_tangent_subspace_audit import (
    ComplexTangentSubspaceAudit,
    PreparedTangentBatch,
    TangentSubspaceAuditRequest,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    SymmetricTangentGreenResponseProjectionConfig,
    TangentContextCheckpointConfig,
)
from greenonet.coupling_artifacts import (
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class LowRankSpectralAuditRequest:
    """Inputs for a frozen diagonal versus spectral-low-rank K audit."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    tangent_context: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    batch_size: int = 10
    selected_samples: tuple[int, ...] | None = None
    transition_log_threshold: float = math.log(2.0)
    subspace_relative_eps: float = 1.0e-12
    metric_eps: float = 1.0e-30
    operator_equivalence_tol: float = 1.0e-10
    monotonicity_relative_tol: float = 1.0e-10
    max_subspace_dimension: int = 4
    ranks: tuple[int, ...] = (4, 8, 16, 32)
    oversampling: int = 16
    power_iterations: int = 3
    probe_count: int = 32
    probe_batch_size: int = 16
    seed: int = 1729
    eigenvalue_relative_floor: float = 1.0e-10
    complement_scale: Literal["unit", "next_ritz"] = "next_ritz"
    benchmark_warmup: int = 1
    benchmark_repeats: int = 3
    save_generated_data: bool = True

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("ranks must not be empty.")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must not contain duplicates.")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
            for rank in self.ranks
        ):
            raise ValueError("ranks must contain positive integers.")
        if tuple(sorted(self.ranks)) != self.ranks:
            raise ValueError("ranks must be strictly increasing.")
        for name, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("max_subspace_dimension", self.max_subspace_dimension, 1),
            ("oversampling", self.oversampling, 0),
            ("power_iterations", self.power_iterations, 0),
            ("probe_count", self.probe_count, 1),
            ("probe_batch_size", self.probe_batch_size, 1),
            ("benchmark_warmup", self.benchmark_warmup, 0),
            ("benchmark_repeats", self.benchmark_repeats, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer.")
        for name, numeric_value, allow_zero in (
            ("transition_log_threshold", self.transition_log_threshold, True),
            ("subspace_relative_eps", self.subspace_relative_eps, False),
            ("metric_eps", self.metric_eps, False),
            ("operator_equivalence_tol", self.operator_equivalence_tol, False),
            ("monotonicity_relative_tol", self.monotonicity_relative_tol, False),
            ("eigenvalue_relative_floor", self.eigenvalue_relative_floor, False),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(float(numeric_value))
                or float(numeric_value) < 0.0
                or (not allow_zero and float(numeric_value) == 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        if self.complement_scale not in {"unit", "next_ritz"}:
            raise ValueError("complement_scale must be 'unit' or 'next_ritz'.")
        if self.selected_samples is not None:
            if len(set(self.selected_samples)) != len(self.selected_samples):
                raise ValueError("selected_samples must not contain duplicates.")
            if any(sample_id < 0 for sample_id in self.selected_samples):
                raise ValueError("selected_samples must be non-negative.")


class MatrixFreeScaledTangentOperator:
    """Apply T=D^-1/2 S^T M S D^-1/2 without materializing T."""

    def __init__(self, context: SymmetricTangentGreenResponseContext) -> None:
        self.context = context
        self.inverse_sqrt_denominator = context.denominator.rsqrt()
        self.application_count = 0
        self.vector_application_count = 0
        self.global_matrix_materialized = False
        self.global_matrix_solve = False

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        self.context.validate_for(values)
        if values.dim() != 2:
            raise ValueError("Scaled tangent operator input must have shape (N, P).")
        whitened_source = values * self.inverse_sqrt_denominator.unsqueeze(0)
        response_pair = self.context.response_operator.forward_pair(
            torch.stack((whitened_source, whitened_source), dim=1)
        )
        response = response_pair[:, 0] + response_pair[:, 1]
        output = self.context.tangent_gradient(response)
        output = output * self.inverse_sqrt_denominator.unsqueeze(0)
        if not torch.all(torch.isfinite(output)):
            raise RuntimeError("Scaled tangent operator produced non-finite values.")
        self.application_count += 1
        self.vector_application_count += values.shape[0]
        return output


@dataclass(frozen=True)
class LowRankSpectralContext:
    """Nested spectral basis and cheap D-plus-low-rank inverse actions."""

    operator: MatrixFreeScaledTangentOperator
    ranks: tuple[int, ...]
    eigenvalues: torch.Tensor
    basis: torch.Tensor
    ritz_residual_norm: torch.Tensor
    eigenvalue_relative_floor: float
    t_frobenius_squared_estimate: float
    t_minus_i_frobenius_squared_estimate: float
    basis_setup_seconds: float
    probe_seconds: float
    setup_operator_application_count: int
    setup_vector_application_count: int

    @classmethod
    def from_eigenpairs(
        cls,
        *,
        operator: MatrixFreeScaledTangentOperator,
        ranks: tuple[int, ...],
        eigenvalues: torch.Tensor,
        basis: torch.Tensor,
        eigenvalue_relative_floor: float,
    ) -> LowRankSpectralContext:
        if basis.dim() != 2 or eigenvalues.dim() != 1:
            raise ValueError("basis and eigenvalues must have shapes (P, R) and (R,).")
        if basis.shape[1] != eigenvalues.shape[0]:
            raise ValueError("basis column count must match eigenvalue count.")
        if max(ranks) > eigenvalues.shape[0]:
            raise ValueError("The largest rank exceeds the supplied eigenpair count.")
        action = operator.apply(basis.transpose(0, 1)).transpose(0, 1)
        residual = torch.linalg.vector_norm(
            action - basis * eigenvalues.unsqueeze(0),
            dim=0,
        )
        return cls(
            operator=operator,
            ranks=ranks,
            eigenvalues=eigenvalues,
            basis=basis,
            ritz_residual_norm=residual,
            eigenvalue_relative_floor=eigenvalue_relative_floor,
            t_frobenius_squared_estimate=float(eigenvalues.square().sum().item()),
            t_minus_i_frobenius_squared_estimate=float(
                (eigenvalues - 1.0).square().sum().item()
            ),
            basis_setup_seconds=0.0,
            probe_seconds=0.0,
            setup_operator_application_count=1,
            setup_vector_application_count=basis.shape[1],
        )

    @property
    def maximum_rank(self) -> int:
        return self.basis.shape[1]

    @property
    def eigenvalue_floor(self) -> float:
        maximum = float(self.eigenvalues.abs().max().item())
        return max(
            self.eigenvalue_relative_floor * maximum,
            torch.finfo(self.eigenvalues.dtype).tiny,
        )

    def apply_inverse(
        self,
        values: torch.Tensor,
        *,
        rank: int,
        complement_scale: Literal["unit", "next_ritz"] = "unit",
    ) -> torch.Tensor:
        self.operator.context.validate_for(values)
        if values.dim() != 2:
            raise ValueError("Low-rank inverse input must have shape (N, P).")
        if rank not in self.ranks:
            raise ValueError(f"rank must be one of {self.ranks}.")
        inverse_sqrt = self.operator.inverse_sqrt_denominator.unsqueeze(0)
        whitened = values * inverse_sqrt
        basis = self.basis[:, :rank]
        eigenvalues = self.eigenvalues[:rank].clamp_min(self.eigenvalue_floor)
        coefficients = whitened @ basis
        tail_reference = self.tail_reference_eigenvalue(
            rank=rank,
            complement_scale=complement_scale,
        )
        correction_scale = tail_reference / eigenvalues - 1.0
        corrected = whitened + (coefficients * correction_scale.unsqueeze(0)) @ basis.T
        output = corrected * inverse_sqrt
        if not torch.all(torch.isfinite(output)):
            raise RuntimeError("Low-rank inverse produced non-finite values.")
        return output

    def tail_reference_eigenvalue(
        self,
        *,
        rank: int,
        complement_scale: Literal["unit", "next_ritz"],
    ) -> torch.Tensor:
        if complement_scale == "unit":
            return self.eigenvalues.new_tensor(1.0)
        if complement_scale != "next_ritz":
            raise ValueError("complement_scale must be 'unit' or 'next_ritz'.")
        next_index = min(rank, self.eigenvalues.shape[0] - 1)
        return self.eigenvalues[next_index].clamp_min(self.eigenvalue_floor)

    def gradient_coverage(self, gradients: torch.Tensor, *, rank: int) -> torch.Tensor:
        self.operator.context.validate_for(gradients)
        whitened = gradients * self.operator.inverse_sqrt_denominator.unsqueeze(0)
        numerator = (whitened @ self.basis[:, :rank]).square().sum(dim=1)
        denominator = (
            whitened.square().sum(dim=1).clamp_min(torch.finfo(whitened.dtype).tiny)
        )
        return numerator / denominator

    def spectral_rows(self) -> list[dict[str, float | int | str]]:
        t_total = max(
            self.t_frobenius_squared_estimate,
            torch.finfo(self.eigenvalues.dtype).tiny,
        )
        deviation_total = max(
            self.t_minus_i_frobenius_squared_estimate,
            torch.finfo(self.eigenvalues.dtype).tiny,
        )
        cumulative_t = torch.cumsum(self.eigenvalues.square(), dim=0)
        cumulative_deviation = torch.cumsum((self.eigenvalues - 1.0).square(), dim=0)
        return [
            {
                "mode": index + 1,
                "eigenvalue": float(self.eigenvalues[index].item()),
                "ritz_residual_norm": float(self.ritz_residual_norm[index].item()),
                "ritz_relative_residual_norm": float(
                    (
                        self.ritz_residual_norm[index]
                        / self.eigenvalues[index]
                        .abs()
                        .clamp_min(torch.finfo(self.eigenvalues.dtype).tiny)
                    ).item()
                ),
                "cumulative_t_frobenius_fraction": float(
                    cumulative_t[index].item() / t_total
                ),
                "cumulative_t_minus_i_fraction": float(
                    cumulative_deviation[index].item() / deviation_total
                ),
            }
            for index in range(self.maximum_rank)
        ]


class RandomizedSpectralContextBuilder:
    """Build a nested randomized Rayleigh-Ritz basis using only T-vector actions."""

    def __init__(
        self,
        *,
        operator: MatrixFreeScaledTangentOperator,
        ranks: tuple[int, ...],
        oversampling: int,
        power_iterations: int,
        probe_count: int,
        seed: int,
        eigenvalue_relative_floor: float,
        probe_batch_size: int | None = None,
    ) -> None:
        self.operator = operator
        self.ranks = ranks
        self.oversampling = oversampling
        self.power_iterations = power_iterations
        self.probe_count = probe_count
        self.seed = seed
        self.eigenvalue_relative_floor = eigenvalue_relative_floor
        self.probe_batch_size = (
            probe_count if probe_batch_size is None else probe_batch_size
        )

    @torch.no_grad()
    def build(self) -> LowRankSpectralContext:
        point_count = self.operator.context.denominator.numel()
        maximum_rank = max(self.ranks)
        if maximum_rank > point_count:
            raise ValueError(
                f"maximum rank {maximum_rank} exceeds point count {point_count}."
            )
        sketch_size = min(point_count, maximum_rank + self.oversampling)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        dtype = self.operator.context.denominator.dtype
        device = self.operator.context.denominator.device
        omega = torch.randn(
            (sketch_size, point_count),
            dtype=dtype,
            generator=generator,
        ).to(device=device)
        start_calls = self.operator.application_count
        start_vectors = self.operator.vector_application_count
        basis_started = time.perf_counter()
        sample = self.operator.apply(omega)
        q, _ = torch.linalg.qr(sample.T, mode="reduced")
        for _ in range(self.power_iterations):
            sample = self.operator.apply(q.T)
            q, _ = torch.linalg.qr(sample.T, mode="reduced")
        tq = self.operator.apply(q.T).T
        compressed = q.T @ tq
        compressed = 0.5 * (compressed + compressed.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(compressed)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order][:maximum_rank]
        basis = (q @ eigenvectors[:, order])[:, :maximum_rank]
        tbasis = self.operator.apply(basis.T).T
        residual = torch.linalg.vector_norm(
            tbasis - basis * eigenvalues.unsqueeze(0),
            dim=0,
        )
        basis_seconds = time.perf_counter() - basis_started

        probe_started = time.perf_counter()
        t_norm_sum = 0.0
        t_minus_i_norm_sum = 0.0
        remaining = self.probe_count
        while remaining:
            count = min(remaining, self.probe_batch_size)
            raw = torch.randint(
                0,
                2,
                (count, point_count),
                generator=generator,
                dtype=torch.int64,
            )
            probes = (2.0 * raw.to(dtype=dtype) - 1.0).to(device=device)
            action = self.operator.apply(probes)
            t_norm_sum += float(action.square().sum().item())
            t_minus_i_norm_sum += float((action - probes).square().sum().item())
            remaining -= count
        probe_seconds = time.perf_counter() - probe_started
        return LowRankSpectralContext(
            operator=self.operator,
            ranks=self.ranks,
            eigenvalues=eigenvalues,
            basis=basis,
            ritz_residual_norm=residual,
            eigenvalue_relative_floor=self.eigenvalue_relative_floor,
            t_frobenius_squared_estimate=t_norm_sum / self.probe_count,
            t_minus_i_frobenius_squared_estimate=(
                t_minus_i_norm_sum / self.probe_count
            ),
            basis_setup_seconds=basis_seconds,
            probe_seconds=probe_seconds,
            setup_operator_application_count=(
                self.operator.application_count - start_calls
            ),
            setup_vector_application_count=(
                self.operator.vector_application_count - start_vectors
            ),
        )


def _matrix_free_k1_step(
    *,
    context: SymmetricTangentGreenResponseContext,
    mismatch: torch.Tensor,
    gradient: torch.Tensor,
    relative_eps: float,
    inverse_preconditioner: Callable[[torch.Tensor], torch.Tensor],
) -> KrylovSubspaceStepResult:
    direction = inverse_preconditioner(gradient)
    if direction.shape != gradient.shape:
        raise ValueError("inverse_preconditioner must preserve shape.")
    directional_response = context.response_operator.forward_pair(
        torch.stack((direction, direction), dim=1)
    )
    response = directional_response[:, 0] + directional_response[:, 1]
    mass = context.point_mass
    mismatch_energy = mass * mismatch.square().sum(dim=1)
    response_energy = mass * response.square().sum(dim=1)
    eps = (
        relative_eps * torch.maximum(mismatch_energy, response_energy)
        + torch.finfo(mismatch.dtype).tiny
    )
    numerator = (gradient * direction).sum(dim=1).clamp_min(0.0)
    denominator = response_energy + eps
    active = response_energy > eps
    coefficient = torch.where(
        active,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    direction = torch.where(active.unsqueeze(1), direction, torch.zeros_like(direction))
    directional_response = torch.where(
        active.view(-1, 1, 1),
        directional_response,
        torch.zeros_like(directional_response),
    )
    response = torch.where(active.unsqueeze(1), response, torch.zeros_like(response))
    delta = -coefficient.unsqueeze(1) * direction
    mismatch_next = mismatch - coefficient.unsqueeze(1) * response
    cost = mass * mismatch_next.square().sum(dim=1)
    residual = context.tangent_gradient(mismatch_next)
    gram = response_energy[:, None, None]
    zero_orthogonality = torch.zeros(
        (1, mismatch.shape[0]), dtype=mismatch.dtype, device=mismatch.device
    )
    return KrylovSubspaceStepResult(
        directions=direction.unsqueeze(0),
        directional_responses=directional_response.unsqueeze(0),
        response_directions=response.unsqueeze(0),
        coefficients=coefficient.unsqueeze(0),
        direction_active=active.unsqueeze(0),
        deltas=delta.unsqueeze(0),
        mismatches=mismatch_next.unsqueeze(0),
        costs=cost.unsqueeze(0),
        residual_gradient_post=residual,
        response_gram=gram,
        response_orthogonality_max=zero_orthogonality,
        line_search_numerator_0=numerator,
        line_search_denominator_0=denominator,
    )


def matrix_free_preconditioned_krylov_subspace_step(
    *,
    context: SymmetricTangentGreenResponseContext,
    mismatch: torch.Tensor,
    gradient: torch.Tensor,
    max_dimension: int,
    relative_eps: float,
    monotonicity_relative_tol: float,
    inverse_preconditioner: Callable[[torch.Tensor], torch.Tensor],
) -> KrylovSubspaceStepResult:
    if max_dimension == 1:
        return _matrix_free_k1_step(
            context=context,
            mismatch=mismatch,
            gradient=gradient,
            relative_eps=relative_eps,
            inverse_preconditioner=inverse_preconditioner,
        )
    return matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=max_dimension,
        relative_eps=relative_eps,
        monotonicity_relative_tol=monotonicity_relative_tol,
        inverse_preconditioner=inverse_preconditioner,
    )


class ComplexTangentLowRankAudit(ComplexTangentSubspaceAudit):
    """Compare diagonal and nested global-low-rank preconditioners at K=1..4."""

    _TAIL_METRICS = (
        "response_mismatch_cost",
        "canonical_energy",
        "loss_energy_optimized",
        "rel_sol",
        "rel_sol_equal_mean",
        "rel_u_phi",
        "rel_u_psi",
        "rel_flux",
        "tangent_correction_rel_symmetric_pair",
    )

    def __init__(
        self,
        request: LowRankSpectralAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.low_rank_request = request
        super().__init__(
            TangentSubspaceAuditRequest(
                config=request.config,
                coupling_checkpoint=request.coupling_checkpoint,
                green_checkpoint=request.green_checkpoint,
                outdir=request.outdir,
                geometry=request.geometry,
                test_path=request.test_path,
                coefficients=request.coefficients,
                tangent_context=request.tangent_context,
                device=request.device,
                theme=request.theme,
                batch_size=request.batch_size,
                selected_samples=request.selected_samples,
                transition_log_threshold=request.transition_log_threshold,
                subspace_relative_eps=request.subspace_relative_eps,
                metric_eps=request.metric_eps,
                operator_equivalence_tol=request.operator_equivalence_tol,
                monotonicity_relative_tol=request.monotonicity_relative_tol,
                max_subspace_dimension=max(2, request.max_subspace_dimension),
                save_generated_data=request.save_generated_data,
            ),
            logger=logger,
        )
        self.methods = self._build_methods()
        self.spectral_context: LowRankSpectralContext

    def _build_methods(self) -> tuple[TangentMethod, ...]:
        methods = [
            TangentMethod("physical_symmetric", "Physical symmetric", "symmetric")
        ]
        for prefix, label in self._preconditioner_labels():
            methods.extend(
                TangentMethod(
                    f"{prefix}_k{dimension}",
                    f"{label}, K={dimension}",
                    "uncapped_subspace",
                )
                for dimension in range(
                    1, self.low_rank_request.max_subspace_dimension + 1
                )
            )
        return tuple(methods)

    def _initialize_context(self, batch: ComplexCouplingBatch) -> None:
        if self.low_rank_request.tangent_context is None:
            super()._initialize_context(batch)
            return
        if hasattr(self, "response_operator"):
            return
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        checkpoint = TangentContextCheckpointConfig(
            enabled=True,
            path=self.low_rank_request.tangent_context,
            load_policy="required",
            save_after_build=False,
        )
        self._tangent_context_cache = SymmetricTangentGreenResponseContextCache(
            replace(tangent, subspace_dimension=1),
            checkpoint=checkpoint,
            checkpoint_path=self.low_rank_request.tangent_context,
        )
        self.tangent_context = self._tangent_context_cache.get_or_build(
            green_model=self._green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        self.response_operator = self.tangent_context.response_operator
        self._context_build_count = self._tangent_context_cache.build_count
        self._verify_operator_equivalence(batch)

    def _preconditioner_labels(self) -> tuple[tuple[str, str], ...]:
        return (
            ("diag", "Diagonal"),
            *tuple(
                (f"rank{rank}", f"Rank {rank}") for rank in self.low_rank_request.ranks
            ),
        )

    def _inverse_preconditioners(
        self,
    ) -> tuple[tuple[str, str, int, Callable[[torch.Tensor], torch.Tensor]], ...]:
        inverse_denominator = self.tangent_context.denominator.reciprocal().unsqueeze(0)
        output: list[tuple[str, str, int, Callable[[torch.Tensor], torch.Tensor]]] = [
            (
                "diag",
                "Diagonal",
                0,
                lambda values: values * inverse_denominator,
            )
        ]
        for rank in self.low_rank_request.ranks:
            output.append(
                (
                    f"rank{rank}",
                    f"Rank {rank}",
                    rank,
                    self._low_rank_inverse(rank),
                )
            )
        return tuple(output)

    def _low_rank_inverse(
        self,
        rank: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def apply(values: torch.Tensor) -> torch.Tensor:
            return self.spectral_context.apply_inverse(
                values,
                rank=rank,
                complement_scale=self.low_rank_request.complement_scale,
            )

        return apply

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        request = self.low_rank_request
        request.outdir.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(request.config)
        if self._configs.dataset.geometry_mode != "complex":
            raise ValueError("Low-rank tangent audit requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        if projection.mode != "symmetric_tangent_green_response":
            raise ValueError(
                "Low-rank tangent audit requires symmetric_tangent_green_response."
            )
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        if tangent.eta_strategy != "closed_loop_exact_line_search":
            raise ValueError("Low-rank tangent audit requires closed-loop line search.")
        geometry_path = request.geometry or self._configs.dataset.geometry_path
        test_path = request.test_path or self._configs.dataset.test_path
        coefficient_path = (
            request.coefficients or self._configs.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")
        for checkpoint in (request.coupling_checkpoint, request.green_checkpoint):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

        self._device = torch.device(
            request.device or self._configs.coupling_training.device
        )
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=self._configs.dataset.dtype,
        )
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            load_coefficient_functions(coefficient_path),
            branch_input_dim=self._configs.coupling_model.branch_input_dim,
            dtype=self._configs.dataset.dtype,
            coefficient_terms=self._configs.coupling_model.coefficient_terms,
            integration_rule=self._configs.coupling_training.integration_rule,
        )
        if len(dataset) == 0:
            raise ValueError("The test dataset is empty.")
        self._load_models()
        self._cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            self._configs.coupling_model.cross_axis_reconstruction
        )
        self.boundary_context = build_boundary_energy_context(self.geometry)
        edges = ComplexProjectionResponseAudit.build_transition_edges(
            self.geometry,
            threshold=request.transition_log_threshold,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        batches = list(loader)
        first_batch = batches[0].to(self._device)
        self._initialize_context(first_batch)
        scaled_operator = MatrixFreeScaledTangentOperator(self.tangent_context)
        self.spectral_context = RandomizedSpectralContextBuilder(
            operator=scaled_operator,
            ranks=request.ranks,
            oversampling=request.oversampling,
            power_iterations=request.power_iterations,
            probe_count=request.probe_count,
            probe_batch_size=request.probe_batch_size,
            seed=request.seed,
            eigenvalue_relative_floor=request.eigenvalue_relative_floor,
        ).build()
        if self.logger is not None:
            self.logger.info(
                "Spectral context ready: rank=%d setup=%.3fs probes=%.3fs",
                self.spectral_context.maximum_rank,
                self.spectral_context.basis_setup_seconds,
                self.spectral_context.probe_seconds,
            )

        rows: list[dict[str, float | int | str]] = []
        coverage_rows: list[dict[str, float | int | str]] = []
        offsets: dict[int, int] = {}
        first_prepared: PreparedTangentBatch | None = None
        offset = 0
        for batch_index, raw_batch in enumerate(batches, start=1):
            batch = raw_batch.to(self._device)
            prepared = self._prepare_batch(batch)
            if first_prepared is None:
                first_prepared = prepared
            coverage_rows.extend(self._coverage_rows(batch, prepared.gradient))
            for prefix, label, rank, inverse in self._inverse_preconditioners():
                evaluation, krylov = self._evaluate_preconditioner(
                    batch=batch,
                    prepared=prepared,
                    inverse_preconditioner=inverse,
                )
                local_rows = self._metric_rows(
                    batch,
                    evaluation,
                    krylov,
                    edges,
                    context=self.tangent_context,
                )
                rows.extend(
                    self._rename_rows(
                        local_rows,
                        prefix=prefix,
                        label=label,
                        rank=rank,
                    )
                )
            for sample_id in batch.sample_indices.tolist():
                offsets[int(sample_id)] = offset
                offset += 1
            if self.logger is not None:
                self.logger.info(
                    "Quality sweep batch %d/%d complete",
                    batch_index,
                    len(batches),
                )
        if first_prepared is None:
            raise RuntimeError("No batch was prepared for the low-rank audit.")

        aggregate = self._aggregate(rows)
        paired_rows, paired = self._paired(rows)
        spectral_rows = self.spectral_context.spectral_rows()
        selected = self._select_low_rank_samples(rows, offsets)
        quality_metrics = request.outdir / "metrics"
        self._write_csv(
            quality_metrics / "per_sample_low_rank_k1_k4.csv",
            rows,
        )
        self._write_csv(
            quality_metrics / "aggregate_low_rank_k1_k4.csv",
            list(aggregate.values()),
        )
        self._write_csv(
            quality_metrics / "paired_low_rank_k1_k4.csv",
            paired_rows,
        )
        self._write_csv(
            quality_metrics / "per_sample_gradient_coverage.csv",
            coverage_rows,
        )
        self._write_csv(quality_metrics / "spectral_decay.csv", spectral_rows)
        if self.logger is not None:
            self.logger.info("Quality metrics saved; starting runtime benchmark")
        runtime_rows = self._benchmark(self._first_sample(first_prepared))
        self._write_outputs(
            rows=rows,
            aggregate=aggregate,
            paired_rows=paired_rows,
            coverage_rows=coverage_rows,
            runtime_rows=runtime_rows,
            spectral_rows=spectral_rows,
        )
        spectral_archive = self._write_spectral_context()
        figure_paths = self._write_figures(
            aggregate=aggregate,
            coverage_rows=coverage_rows,
            runtime_rows=runtime_rows,
            spectral_rows=spectral_rows,
        )
        if request.save_generated_data:
            self._write_selected_archive(dataset, offsets, selected)
        summary = self._build_low_rank_summary(
            dataset_size=len(dataset),
            geometry_path=geometry_path,
            test_path=test_path,
            coefficient_path=coefficient_path,
            aggregate=aggregate,
            paired=paired,
            coverage_rows=coverage_rows,
            runtime_rows=runtime_rows,
            spectral_archive=spectral_archive,
            selected=selected,
            figure_paths=figure_paths,
            total_seconds=time.perf_counter() - started,
            edges=edges,
        )
        (request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Low-rank K=1..%d audit complete: samples=%d, max_rank=%d",
                request.max_subspace_dimension,
                len(dataset),
                max(request.ranks),
            )
        return summary

    @staticmethod
    def _first_sample(prepared: PreparedTangentBatch) -> PreparedTangentBatch:
        return PreparedTangentBatch(
            raw_physical=prepared.raw_physical[:1],
            symmetric_physical=prepared.symmetric_physical[:1],
            mismatch=prepared.mismatch[:1],
            gradient=prepared.gradient[:1],
        )

    def _local_methods(self) -> tuple[TangentMethod, ...]:
        return (
            TangentMethod("symmetric", "Physical symmetric", "symmetric"),
            TangentMethod("k1_uncapped", "K=1", "uncapped_subspace"),
            *tuple(
                TangentMethod(
                    f"k{dimension}_unconstrained",
                    f"K={dimension}",
                    "uncapped_subspace",
                )
                for dimension in range(
                    2, self.low_rank_request.max_subspace_dimension + 1
                )
            ),
        )

    @torch.no_grad()
    def _evaluate_preconditioner(
        self,
        *,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
        inverse_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[TangentBatchEvaluation, KrylovSubspaceStepResult]:
        request = self.low_rank_request
        krylov = matrix_free_preconditioned_krylov_subspace_step(
            context=self.tangent_context,
            mismatch=prepared.mismatch,
            gradient=prepared.gradient,
            max_dimension=request.max_subspace_dimension,
            relative_eps=request.subspace_relative_eps,
            monotonicity_relative_tol=request.monotonicity_relative_tol,
            inverse_preconditioner=inverse_preconditioner,
        )
        tangent_delta = torch.cat(
            (torch.zeros_like(krylov.deltas[:1]), krylov.deltas),
            dim=0,
        )
        symmetric = prepared.symmetric_physical
        candidate_physical = torch.stack(
            (
                symmetric.unsqueeze(0)[:, :, 0] + tangent_delta,
                symmetric.unsqueeze(0)[:, :, 1] - tangent_delta,
            ),
            dim=2,
        )
        method_count, batch_count, _axis, point_count = candidate_physical.shape
        flat_physical = candidate_physical.reshape(
            method_count * batch_count, 2, point_count
        )
        flat_solution = self.response_operator.forward_pair(flat_physical)
        candidate_solution = flat_solution.reshape(
            method_count, batch_count, 2, point_count
        )
        energy = canonical_complex_energy_loss(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            a_valid=batch.a_valid.repeat(method_count, 1),
            geometry=batch.geometry,
            boundary_context=self.boundary_context,
        )
        cross_axis = self._cross_axis_reconstructor.reconstruct(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            projected_physical=flat_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        diagnostics = ClosedLoopTangentBatchDiagnostics(
            method_id="k1_uncapped",
            eta_cap=None,
            eta_star=krylov.coefficients[0],
            eta_applied=krylov.coefficients[0],
            eta_capped=torch.zeros_like(krylov.coefficients[0], dtype=torch.bool),
            line_search_numerator=krylov.line_search_numerator_0,
            line_search_denominator=krylov.line_search_denominator_0,
        )
        evaluation = TangentBatchEvaluation(
            methods=self._local_methods(),
            raw_physical=prepared.raw_physical,
            symmetric_physical=symmetric,
            configured_physical=candidate_physical[1],
            tangent_gradient=prepared.gradient,
            tangent_preconditioner_base=self.tangent_context.preconditioner_base,
            tangent_delta=tangent_delta,
            candidate_physical=candidate_physical,
            candidate_solution=candidate_solution,
            candidate_equal_prediction=cross_axis.u_equal_mean_valid.reshape(
                method_count, batch_count, point_count
            ),
            candidate_prediction=cross_axis.u_pred_valid.reshape(
                method_count, batch_count, point_count
            ),
            canonical_energy=energy.total_per_sample.reshape(method_count, batch_count),
            canonical_bulk_energy=energy.bulk_per_sample.reshape(
                method_count, batch_count
            ),
            canonical_boundary_energy=energy.boundary_per_sample.reshape(
                method_count, batch_count
            ),
            closed_loop=(diagnostics,),
        )
        return evaluation, krylov

    def _rename_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
        *,
        prefix: str,
        label: str,
        rank: int,
    ) -> list[dict[str, float | int | str]]:
        output: list[dict[str, float | int | str]] = []
        for row in rows:
            original = str(row["method_id"])
            if original == "symmetric":
                if prefix != "diag":
                    continue
                method_id = "physical_symmetric"
                dimension = 0
                preconditioner = "none"
            else:
                dimension = (
                    1
                    if original == "k1_uncapped"
                    else int(original[1 : original.index("_")])
                )
                method_id = f"{prefix}_k{dimension}"
                preconditioner = "diagonal" if rank == 0 else f"low_rank_{rank}"
            method = next(item for item in self.methods if item.method_id == method_id)
            renamed = dict(row)
            renamed.update(
                {
                    "method_id": method_id,
                    "method_label": method.label,
                    "method_kind": method.kind,
                    "preconditioner": preconditioner,
                    "spectral_rank": rank,
                    "subspace_dimension": dimension,
                    "preconditioner_label": "None"
                    if rank == 0 and dimension == 0
                    else label,
                }
            )
            output.append(renamed)
        return output

    def _coverage_rows(
        self,
        batch: ComplexCouplingBatch,
        gradients: torch.Tensor,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for rank in self.low_rank_request.ranks:
            coverage = self.spectral_context.gradient_coverage(gradients, rank=rank)
            for offset, sample_id in enumerate(batch.sample_indices.tolist()):
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "rank": rank,
                        "gradient_coverage": float(coverage[offset].item()),
                    }
                )
        return rows

    def _aggregate(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, dict[str, float | int | str]]:
        aggregate = self.aggregate_rows(rows, self.methods)
        for method in self.methods:
            payload = aggregate[method.method_id]
            payload["method_id"] = method.method_id
            selected = [row for row in rows if row["method_id"] == method.method_id]
            if selected:
                payload["preconditioner"] = str(selected[0]["preconditioner"])
                payload["spectral_rank"] = int(selected[0]["spectral_rank"])
                payload["subspace_dimension"] = int(selected[0]["subspace_dimension"])
            for metric in self._TAIL_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in selected if metric in row],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                if values.size:
                    payload[f"{metric}_std"] = float(values.std())
                    payload[f"{metric}_p90"] = float(np.quantile(values, 0.90))
                    payload[f"{metric}_p95"] = float(np.quantile(values, 0.95))
                    payload[f"{metric}_max"] = float(values.max())
        return aggregate

    def _paired(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
        by_method = {
            method.method_id: {
                int(row["sample_id"]): row
                for row in rows
                if row["method_id"] == method.method_id
            }
            for method in self.methods
        }
        comparisons: list[tuple[str, str, str]] = []
        prefixes = ["diag", *[f"rank{rank}" for rank in self.low_rank_request.ranks]]
        for prefix in prefixes:
            for dimension in range(2, self.low_rank_request.max_subspace_dimension + 1):
                comparisons.append(
                    (
                        f"{prefix}_k{dimension}_vs_k{dimension - 1}",
                        f"{prefix}_k{dimension - 1}",
                        f"{prefix}_k{dimension}",
                    )
                )
        for rank in self.low_rank_request.ranks:
            for dimension in range(1, self.low_rank_request.max_subspace_dimension + 1):
                comparisons.append(
                    (
                        f"rank{rank}_vs_diag_k{dimension}",
                        f"diag_k{dimension}",
                        f"rank{rank}_k{dimension}",
                    )
                )
        flat: list[dict[str, float | int | str]] = []
        nested: dict[str, Any] = {}
        for comparison_id, baseline_id, candidate_id in comparisons:
            common = sorted(set(by_method[baseline_id]) & set(by_method[candidate_id]))
            payload: dict[str, Any] = {}
            for metric in self._TAIL_METRICS:
                pairs = [
                    (
                        float(by_method[baseline_id][sample_id][metric]),
                        float(by_method[candidate_id][sample_id][metric]),
                    )
                    for sample_id in common
                    if metric in by_method[baseline_id][sample_id]
                    and metric in by_method[candidate_id][sample_id]
                ]
                if not pairs:
                    continue
                baseline = np.asarray([pair[0] for pair in pairs])
                candidate = np.asarray([pair[1] for pair in pairs])
                delta = candidate - baseline
                metric_payload = {
                    "sample_count": len(pairs),
                    "baseline_mean": float(baseline.mean()),
                    "candidate_mean": float(candidate.mean()),
                    "mean_delta": float(delta.mean()),
                    "relative_mean_change": self._relative_change(
                        baseline=float(baseline.mean()),
                        candidate=float(candidate.mean()),
                    ),
                    "improved_sample_count": int(np.count_nonzero(delta < 0.0)),
                    "worsened_sample_count": int(np.count_nonzero(delta > 0.0)),
                    "unchanged_sample_count": int(np.count_nonzero(delta == 0.0)),
                    "max_worsening": float(max(0.0, float(delta.max()))),
                }
                payload[metric] = metric_payload
                flat.append(
                    {
                        "comparison_id": comparison_id,
                        "baseline_method": baseline_id,
                        "candidate_method": candidate_id,
                        "metric": metric,
                        **metric_payload,
                    }
                )
            nested[comparison_id] = payload
        return flat, nested

    def _select_low_rank_samples(
        self,
        rows: Sequence[dict[str, float | int | str]],
        offsets: dict[int, int],
    ) -> tuple[int, ...]:
        if self.low_rank_request.selected_samples is not None:
            missing = sorted(set(self.low_rank_request.selected_samples) - set(offsets))
            if missing:
                raise ValueError(f"Selected sample IDs are unavailable: {missing}.")
            return self.low_rank_request.selected_samples
        target_id = f"rank{max(self.low_rank_request.ranks)}_k{self.low_rank_request.max_subspace_dimension}"
        target = [
            row for row in rows if row["method_id"] == target_id and "rel_sol" in row
        ]
        if not target:
            return (min(offsets),)
        ordered = sorted(target, key=lambda row: float(row["rel_sol"]))
        return tuple(
            dict.fromkeys(
                (
                    int(ordered[len(ordered) // 2]["sample_id"]),
                    int(ordered[-1]["sample_id"]),
                )
            )
        )

    def _benchmark(
        self, prepared: PreparedTangentBatch
    ) -> list[dict[str, float | int | str]]:
        request = self.low_rank_request
        rows: list[dict[str, float | int | str]] = []
        for prefix, label, rank, inverse in self._inverse_preconditioners():
            for dimension in range(1, request.max_subspace_dimension + 1):
                forward = self._benchmark_cell(
                    prepared=prepared,
                    inverse=inverse,
                    dimension=dimension,
                    backward=False,
                )
                forward_backward = self._benchmark_cell(
                    prepared=prepared,
                    inverse=inverse,
                    dimension=dimension,
                    backward=True,
                )
                rows.append(
                    {
                        "preconditioner_prefix": prefix,
                        "preconditioner_label": label,
                        "spectral_rank": rank,
                        "subspace_dimension": dimension,
                        "batch_size": prepared.mismatch.shape[0],
                        "forward_mean_ms": float(np.mean(forward)),
                        "forward_p95_ms": float(np.quantile(forward, 0.95)),
                        "forward_max_ms": float(np.max(forward)),
                        "forward_backward_mean_ms": float(np.mean(forward_backward)),
                        "forward_backward_p95_ms": float(
                            np.quantile(forward_backward, 0.95)
                        ),
                        "forward_backward_max_ms": float(np.max(forward_backward)),
                    }
                )
            if self.logger is not None:
                self.logger.info("Runtime benchmark complete: %s", label)
        diagonal_k4 = next(
            row
            for row in rows
            if row["preconditioner_prefix"] == "diag"
            and row["subspace_dimension"] == request.max_subspace_dimension
        )
        baseline_ms = float(diagonal_k4["forward_backward_mean_ms"])
        for row in rows:
            saving_ms = baseline_ms - float(row["forward_backward_mean_ms"])
            row["milliseconds_saved_vs_diagonal_kmax"] = saving_ms
            uses_spectral_setup = row["preconditioner_prefix"] != "diag"
            row["spectral_setup_applicable"] = uses_spectral_setup
            if not uses_spectral_setup:
                row["break_even_optimizer_steps"] = 0.0
            else:
                row["break_even_optimizer_steps"] = (
                    math.inf
                    if saving_ms <= 0.0
                    else 1000.0 * self.spectral_context.basis_setup_seconds / saving_ms
                )
        return rows

    def _benchmark_cell(
        self,
        *,
        prepared: PreparedTangentBatch,
        inverse: Callable[[torch.Tensor], torch.Tensor],
        dimension: int,
        backward: bool,
    ) -> list[float]:
        request = self.low_rank_request
        timings: list[float] = []
        total = request.benchmark_warmup + request.benchmark_repeats
        for iteration in range(total):
            mismatch = prepared.mismatch.detach().clone().requires_grad_(backward)
            gradient = self.tangent_context.tangent_gradient(mismatch)
            self._synchronize()
            started = time.perf_counter()
            result = matrix_free_preconditioned_krylov_subspace_step(
                context=self.tangent_context,
                mismatch=mismatch,
                gradient=gradient,
                max_dimension=dimension,
                relative_eps=request.subspace_relative_eps,
                monotonicity_relative_tol=request.monotonicity_relative_tol,
                inverse_preconditioner=inverse,
            )
            if backward:
                torch.autograd.backward(result.costs[-1].sum())
            self._synchronize()
            elapsed = 1000.0 * (time.perf_counter() - started)
            if iteration >= request.benchmark_warmup:
                timings.append(elapsed)
        return timings

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _write_outputs(
        self,
        *,
        rows: Sequence[dict[str, float | int | str]],
        aggregate: dict[str, dict[str, float | int | str]],
        paired_rows: Sequence[dict[str, float | int | str]],
        coverage_rows: Sequence[dict[str, float | int | str]],
        runtime_rows: Sequence[dict[str, float | int | str]],
        spectral_rows: Sequence[dict[str, float | int | str]],
    ) -> None:
        metrics = self.low_rank_request.outdir / "metrics"
        self._write_csv(metrics / "per_sample_low_rank_k1_k4.csv", rows)
        self._write_csv(
            metrics / "aggregate_low_rank_k1_k4.csv", list(aggregate.values())
        )
        self._write_csv(metrics / "paired_low_rank_k1_k4.csv", paired_rows)
        self._write_csv(metrics / "per_sample_gradient_coverage.csv", coverage_rows)
        self._write_csv(metrics / "runtime_benchmark.csv", runtime_rows)
        self._write_csv(metrics / "spectral_decay.csv", spectral_rows)

    def _write_spectral_context(self) -> dict[str, int | str]:
        path = self.low_rank_request.outdir / "data" / "spectral_context.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        basis = self.spectral_context.basis.detach().cpu().numpy()
        eigenvalues = self.spectral_context.eigenvalues.detach().cpu().numpy()
        residual = self.spectral_context.ritz_residual_norm.detach().cpu().numpy()
        np.savez_compressed(
            path,
            basis=basis,
            eigenvalues=eigenvalues,
            ritz_residual_norm=residual,
            ranks=np.asarray(self.low_rank_request.ranks, dtype=np.int64),
        )
        return {
            "path": str(path),
            "raw_tensor_bytes": int(
                basis.nbytes + eigenvalues.nbytes + residual.nbytes
            ),
            "archive_bytes": path.stat().st_size,
        }

    def _write_selected_archive(
        self,
        dataset: ComplexCouplingDataset,
        offsets: dict[int, int],
        selected: tuple[int, ...],
    ) -> None:
        batch = complex_coupling_collate_fn(
            [dataset[offsets[sample_id]] for sample_id in selected]
        ).to(self._device)
        prepared = self._prepare_batch(batch)
        physical: list[torch.Tensor] = []
        solution: list[torch.Tensor] = []
        method_ids = ["physical_symmetric"]
        symmetric_solution = self.response_operator.forward_pair(
            prepared.symmetric_physical
        )
        physical.append(prepared.symmetric_physical)
        solution.append(symmetric_solution)
        for prefix, _label, _rank, inverse in self._inverse_preconditioners():
            evaluation, _ = self._evaluate_preconditioner(
                batch=batch,
                prepared=prepared,
                inverse_preconditioner=inverse,
            )
            for dimension in range(1, self.low_rank_request.max_subspace_dimension + 1):
                method_ids.append(f"{prefix}_k{dimension}")
                physical.append(evaluation.candidate_physical[dimension])
                solution.append(evaluation.candidate_solution[dimension])
        path = self.low_rank_request.outdir / "data" / "selected_low_rank_k1_k4.npz"
        np.savez_compressed(
            path,
            sample_ids=np.asarray(selected, dtype=np.int64),
            method_ids=np.asarray(method_ids),
            rhs=batch.rhs_valid.detach().cpu().numpy(),
            physical=torch.stack(physical).detach().cpu().numpy(),
            solution=torch.stack(solution).detach().cpu().numpy(),
        )

    def _write_figures(
        self,
        *,
        aggregate: dict[str, dict[str, float | int | str]],
        coverage_rows: Sequence[dict[str, float | int | str]],
        runtime_rows: Sequence[dict[str, float | int | str]],
        spectral_rows: Sequence[dict[str, float | int | str]],
    ) -> list[Path]:
        figures = self.low_rank_request.outdir / "figures"
        spectral_path = figures / "spectral_decay_and_gradient_coverage"
        spectral = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Ritz spectrum",
                "Captured operator energy",
                "Gradient coverage",
            ),
        )
        modes = [int(row["mode"]) for row in spectral_rows]
        spectral.add_trace(
            go.Scatter(
                x=modes,
                y=[float(row["eigenvalue"]) for row in spectral_rows],
                mode="lines+markers",
                name="eigenvalue",
            ),
            row=1,
            col=1,
        )
        spectral.add_trace(
            go.Scatter(
                x=modes,
                y=[
                    float(row["cumulative_t_frobenius_fraction"])
                    for row in spectral_rows
                ],
                mode="lines+markers",
                name="T Frobenius",
            ),
            row=1,
            col=2,
        )
        spectral.add_trace(
            go.Scatter(
                x=modes,
                y=[
                    float(row["cumulative_t_minus_i_fraction"]) for row in spectral_rows
                ],
                mode="lines+markers",
                name="T-I Frobenius",
            ),
            row=1,
            col=2,
        )
        for rank in self.low_rank_request.ranks:
            values = [
                float(row["gradient_coverage"])
                for row in coverage_rows
                if int(row["rank"]) == rank
            ]
            spectral.add_trace(
                go.Box(y=values, name=f"r={rank}", boxmean=True), row=1, col=3
            )
        spectral.update_yaxes(type="log", row=1, col=1)
        spectral.update_layout(
            template=self.low_rank_request.theme,
            width=1500,
            height=480,
            title="Matrix-Free Spectral Decay and Training-Gradient Coverage",
        )
        save_plotly_figure(spectral, spectral_path, self.logger)

        quality_path = figures / "quality_rank_by_k"
        quality = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("rel_sol (%)", "response mismatch", "optimized energy"),
        )
        labels = [label for _prefix, label in self._preconditioner_labels()]
        prefixes = [prefix for prefix, _label in self._preconditioner_labels()]
        metric_specs = (
            ("rel_sol_mean", 100.0),
            ("response_mismatch_cost_mean", 1.0),
            ("loss_energy_optimized_mean", 1.0),
        )
        for column, (metric, scale) in enumerate(metric_specs, start=1):
            z = [
                [
                    scale * float(aggregate[f"{prefix}_k{k}"][metric])
                    for k in range(1, self.low_rank_request.max_subspace_dimension + 1)
                ]
                for prefix in prefixes
            ]
            quality.add_trace(
                go.Heatmap(
                    z=z,
                    x=list(range(1, self.low_rank_request.max_subspace_dimension + 1)),
                    y=labels,
                    colorscale="Viridis",
                    colorbar={"title": metric} if column == 3 else None,
                    showscale=column == 3,
                ),
                row=1,
                col=column,
            )
        quality.update_layout(
            template=self.low_rank_request.theme,
            width=1500,
            height=520,
            title="Frozen Checkpoint Quality: Preconditioner Rank by Tangent Dimension",
        )
        save_plotly_figure(quality, quality_path, self.logger)

        runtime_path = figures / "runtime_rank_by_k"
        runtime = go.Figure()
        for prefix, label in self._preconditioner_labels():
            selected_rows = [
                row for row in runtime_rows if row["preconditioner_prefix"] == prefix
            ]
            runtime.add_trace(
                go.Scatter(
                    x=[int(row["subspace_dimension"]) for row in selected_rows],
                    y=[float(row["forward_backward_mean_ms"]) for row in selected_rows],
                    mode="lines+markers",
                    name=label,
                )
            )
        runtime.update_layout(
            template=self.low_rank_request.theme,
            width=900,
            height=520,
            title="Tangent Core Forward+Backward Runtime",
            xaxis_title="K",
            yaxis_title="milliseconds / batch",
        )
        save_plotly_figure(runtime, runtime_path, self.logger)
        return [spectral_path, quality_path, runtime_path]

    def _build_low_rank_summary(
        self,
        *,
        dataset_size: int,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path,
        aggregate: dict[str, dict[str, float | int | str]],
        paired: dict[str, Any],
        coverage_rows: Sequence[dict[str, float | int | str]],
        runtime_rows: Sequence[dict[str, float | int | str]],
        spectral_archive: dict[str, int | str],
        selected: tuple[int, ...],
        figure_paths: Sequence[Path],
        total_seconds: float,
        edges: ProjectionTransitionEdges,
    ) -> dict[str, Any]:
        request = self.low_rank_request
        canonical = ComplexCanonicalEnergyConfig.from_raw(
            self._configs.coupling_training.canonical_energy
        )
        coverage = {
            str(rank): {
                "mean": float(
                    np.mean(
                        [
                            float(row["gradient_coverage"])
                            for row in coverage_rows
                            if int(row["rank"]) == rank
                        ]
                    )
                ),
                "p10": float(
                    np.quantile(
                        [
                            float(row["gradient_coverage"])
                            for row in coverage_rows
                            if int(row["rank"]) == rank
                        ],
                        0.10,
                    )
                ),
                "p50": float(
                    np.quantile(
                        [
                            float(row["gradient_coverage"])
                            for row in coverage_rows
                            if int(row["rank"]) == rank
                        ],
                        0.50,
                    )
                ),
                "p90": float(
                    np.quantile(
                        [
                            float(row["gradient_coverage"])
                            for row in coverage_rows
                            if int(row["rank"]) == rank
                        ],
                        0.90,
                    )
                ),
            }
            for rank in request.ranks
        }
        quality_methods = [
            method
            for method in self.methods
            if method.method_id != "physical_symmetric"
        ]
        findings: dict[str, Any] = {}
        for metric in (
            "rel_sol_mean",
            "rel_u_phi_mean",
            "rel_u_psi_mean",
            "response_mismatch_cost_mean",
            "loss_energy_optimized_mean",
        ):
            available = [
                method
                for method in quality_methods
                if metric in aggregate[method.method_id]
            ]
            if available:
                best = min(
                    available,
                    key=lambda method: float(aggregate[method.method_id][metric]),
                )
                findings[f"lowest_{metric}_method"] = best.method_id
                findings[f"lowest_{metric}_value"] = float(
                    aggregate[best.method_id][metric]
                )
        return {
            "diagnostic": "matrix_free_low_rank_spectral_k1_k4_posthoc_audit",
            "status": "frozen_checkpoint_posthoc",
            "production_training_changed": False,
            "config": str(request.config),
            "tangent_subspace_dimension_provenance": self._configs.raw.get(
                "tangent_subspace_dimension_provenance"
            ),
            "coupling_checkpoint": str(request.coupling_checkpoint),
            "green_checkpoint": str(request.green_checkpoint),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": str(coefficient_path),
            "sample_count": dataset_size,
            "ranks": list(request.ranks),
            "complement_scale": request.complement_scale,
            "maximum_subspace_dimension": request.max_subspace_dimension,
            "canonical_boundary_weight": canonical.boundary_weight,
            "formula": {
                "scaled_operator": "T=D^-1/2*(H_x+H_y)^T*M*(H_x+H_y)*D^-1/2",
                "inverse": ("P_r^-1=D^-1/2*[I+U_r*(mu_r*Lambda_r^-1-I)*U_r^T]*D^-1/2"),
                "tail_reference": (
                    "mu_r=1"
                    if request.complement_scale == "unit"
                    else "mu_r=next stored Ritz value (rank-32 uses lambda_32)"
                ),
                "krylov": "z_k=P_r^-1*g_k with two-pass response-space MGS, K=1..4",
            },
            "matrix_policy": {
                "global_matrix_materialized": False,
                "global_matrix_solve": False,
                "small_dense_eigendecomposition_dimension": min(
                    self.geometry.num_points, max(request.ranks) + request.oversampling
                ),
                "response_context_build_count": self._context_build_count,
                "operator_production_equivalence_max_abs": self._operator_equivalence_max_abs,
            },
            "spectral_context": {
                "stored_max_rank": self.spectral_context.maximum_rank,
                "basis_setup_seconds": self.spectral_context.basis_setup_seconds,
                "probe_seconds": self.spectral_context.probe_seconds,
                "operator_application_count": self.spectral_context.setup_operator_application_count,
                "vector_application_count": self.spectral_context.setup_vector_application_count,
                "eigenvalue_floor": self.spectral_context.eigenvalue_floor,
                "top_eigenvalue": float(self.spectral_context.eigenvalues[0].item()),
                "smallest_stored_eigenvalue": float(
                    self.spectral_context.eigenvalues[-1].item()
                ),
                "maximum_ritz_residual_norm": float(
                    self.spectral_context.ritz_residual_norm.max().item()
                ),
                "maximum_ritz_relative_residual_norm": float(
                    (
                        self.spectral_context.ritz_residual_norm
                        / self.spectral_context.eigenvalues.abs().clamp_min(
                            torch.finfo(self.spectral_context.eigenvalues.dtype).tiny
                        )
                    )
                    .max()
                    .item()
                ),
                "t_frobenius_squared_estimate": self.spectral_context.t_frobenius_squared_estimate,
                "t_minus_i_frobenius_squared_estimate": self.spectral_context.t_minus_i_frobenius_squared_estimate,
                "archive": spectral_archive,
            },
            "gradient_coverage": coverage,
            "reference_policy": {
                "sol_and_flux_used_for_correction": False,
                "sol_and_flux_used_for_evaluation_only": True,
            },
            "transition_definition": {
                "log_threshold": request.transition_log_threshold,
                "phi_transition_edge_count": int(edges.phi_transition.shape[0]),
                "psi_transition_edge_count": int(edges.psi_transition.shape[0]),
            },
            "aggregate_metrics": aggregate,
            "paired_comparisons": paired,
            "runtime_benchmark": list(runtime_rows),
            "findings": findings,
            "selected_samples": list(selected),
            "metric_files": {
                "per_sample": "metrics/per_sample_low_rank_k1_k4.csv",
                "aggregate": "metrics/aggregate_low_rank_k1_k4.csv",
                "paired": "metrics/paired_low_rank_k1_k4.csv",
                "spectral_decay": "metrics/spectral_decay.csv",
                "gradient_coverage": "metrics/per_sample_gradient_coverage.csv",
                "runtime": "metrics/runtime_benchmark.csv",
            },
            "figure_json": [
                str(path.relative_to(request.outdir)) + ".json" for path in figure_paths
            ],
            "total_seconds": total_seconds,
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate_metrics"]
        maximum_dimension = self.low_rank_request.max_subspace_dimension
        diagonal_id = f"diag_k{maximum_dimension}"
        best_solution_id = str(summary["findings"]["lowest_rel_sol_mean_method"])
        diagonal = aggregate[diagonal_id]
        best_solution = aggregate[best_solution_id]
        runtime_by_method = {
            f"{row['preconditioner_prefix']}_k{row['subspace_dimension']}": row
            for row in summary["runtime_benchmark"]
        }

        def relative_change(candidate: dict[str, Any], metric: str) -> float:
            baseline_value = float(diagonal[metric])
            return 100.0 * (float(candidate[metric]) / baseline_value - 1.0)

        diagonal_runtime = float(
            runtime_by_method[diagonal_id]["forward_backward_mean_ms"]
        )
        best_runtime = float(
            runtime_by_method[best_solution_id]["forward_backward_mean_ms"]
        )
        smaller_low_rank_dominators: list[str] = []
        for rank in self.low_rank_request.ranks:
            for dimension in range(1, maximum_dimension):
                method_id = f"rank{rank}_k{dimension}"
                if (
                    float(aggregate[method_id]["rel_sol_mean"])
                    < float(diagonal["rel_sol_mean"])
                    and float(runtime_by_method[method_id]["forward_backward_mean_ms"])
                    < diagonal_runtime
                ):
                    smaller_low_rank_dominators.append(method_id)
        lines = [
            "# Matrix-Free Low-Rank Spectral Tangent Audit",
            "",
            "The CouplingNet and GreenNet checkpoints are frozen. The correction uses",
            "only source, geometry, coefficients, and frozen Green response operators;",
            "reference solution and directional targets are evaluation-only.",
            "",
            "## Quality",
            "",
            "| preconditioner | K | mismatch | optimized energy | rel_sol | rel_u_phi | rel_u_psi | rel_flux |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in self.methods:
            if method.method_id == "physical_symmetric":
                continue
            payload = aggregate[method.method_id]
            lines.append(
                "| "
                f"{payload['preconditioner']} | {int(payload['subspace_dimension'])} | "
                f"{float(payload['response_mismatch_cost_mean']):.6e} | "
                f"{float(payload['loss_energy_optimized_mean']):.6e} | "
                f"{100.0 * float(payload.get('rel_sol_mean', math.nan)):.4f}% | "
                f"{100.0 * float(payload.get('rel_u_phi_mean', math.nan)):.4f}% | "
                f"{100.0 * float(payload.get('rel_u_psi_mean', math.nan)):.4f}% | "
                f"{100.0 * float(payload.get('rel_flux_mean', math.nan)):.4f}% |"
            )
        lines.extend(
            [
                "",
                "## Spectral Setup",
                "",
                f"- Maximum stored rank: {summary['spectral_context']['stored_max_rank']}",
                f"- Basis setup: {summary['spectral_context']['basis_setup_seconds']:.6f} s",
                f"- Probe diagnostics: {summary['spectral_context']['probe_seconds']:.6f} s",
                f"- Sidecar bytes: {summary['spectral_context']['archive']['archive_bytes']}",
                f"- Maximum Ritz residual: {summary['spectral_context']['maximum_ritz_residual_norm']:.6e}",
                f"- Maximum relative Ritz residual: {summary['spectral_context']['maximum_ritz_relative_residual_norm']:.6e}",
                "",
                "## Gradient Coverage",
                "",
                "| rank | mean | p10 | p50 | p90 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for rank in self.low_rank_request.ranks:
            payload = summary["gradient_coverage"][str(rank)]
            lines.append(
                f"| {rank} | {payload['mean']:.6f} | {payload['p10']:.6f} | "
                f"{payload['p50']:.6f} | {payload['p90']:.6f} |"
            )
        lines.extend(
            [
                "",
                "## Runtime and Decision",
                "",
                f"- The lowest mean rel_sol is `{best_solution_id}` at "
                f"{100.0 * float(best_solution['rel_sol_mean']):.4f}%, a "
                f"{relative_change(best_solution, 'rel_sol_mean'):.3f}% change from "
                f"`{diagonal_id}`.",
                f"- Its rel_sol p95 is "
                f"{100.0 * float(best_solution['rel_sol_p95']):.4f}% versus "
                f"{100.0 * float(diagonal['rel_sol_p95']):.4f}% for `{diagonal_id}`.",
                f"- Response mismatch changes by "
                f"{relative_change(best_solution, 'response_mismatch_cost_mean'):.3f}% "
                f"and optimized energy by "
                f"{relative_change(best_solution, 'loss_energy_optimized_mean'):.3f}%.",
                f"- Representative tangent-core forward+backward runtime is "
                f"{best_runtime:.3f} ms for `{best_solution_id}` and "
                f"{diagonal_runtime:.3f} ms for `{diagonal_id}` "
                f"({best_runtime / diagonal_runtime:.3f}x). This excludes the model, "
                "data loading, validation, and checkpoint I/O.",
                f"- Low-rank cells with K < {maximum_dimension} that beat "
                f"`{diagonal_id}` in both mean rel_sol and tangent-core runtime: "
                f"{len(smaller_low_rank_dominators)}.",
                "- The reusable spectral setup is therefore an accuracy extension, "
                "not an efficiency replacement for increasing K in this frozen audit.",
                "",
                "## Interpretation",
                "",
                "- Increasing K and increasing global spectral rank are separate axes:",
                "  K expands the sample-specific response Krylov subspace, while rank",
                "  improves each preconditioned direction with reusable operator-global modes.",
                "- The spectral tail uses the next Ritz value as its scale. Treating the",
                "  unresolved complement as the identity over-damps it and is not the",
                "  recommended approximation for this operator.",
                "- A low response objective is necessary for the tangent correction but is",
                "  not by itself evidence of lower reference solution or directional error.",
                "- No full P-by-P matrix and no global linear solve are used.",
            ]
        )
        (self.low_rank_request.outdir / "diagnosis_report.md").write_text(
            "\n".join(lines) + "\n"
        )


def run_tangent_low_rank_audit(
    request: LowRankSpectralAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexTangentLowRankAudit(request, logger=logger).run()
