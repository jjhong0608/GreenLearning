"""Reference-source diagnostics and matched-accuracy frozen initialization audit.

Production response/MGS/reconstruction and the existing frozen loader are reused;
neither checkpoint weights nor their persisted contexts are modified.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingBatch
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
)
from greenonet.complex_frozen_tangent_csv import (
    FrozenRunSpec,
    FrozenTangentBenchmark,
    FrozenTangentCsvAudit,
    FrozenTangentCsvRequest,
    _FrozenRunSession,
    _json,
    _sha256,
    _write_csv,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
)
from greenonet.complex_losses import canonical_complex_energy_loss
from greenonet.complex_projection import (
    apply_complex_balance_projection,
    reconstruct_complex_projection,
)
from greenonet.complex_tangent_projection import KrylovSubspaceStepResult
from greenonet.complex_tangent_subspace_audit import PreparedTangentBatch
from greenonet.source_initialization_metrics import (
    match_accuracy,
    physical_equal_split,
    relative_norm,
    summarize,
    symmetric_source_balance,
)
from greenonet.source_initialization_report import (
    SourceInitializationReport,
    verify_coverage,
)
from greenonet.source_initialization_context import verified_runtime_cache


@dataclass(frozen=True)
class SourceInitializationRequest:
    manifest: Path
    outdir: Path
    stage: str = "all"
    device: str = "cuda:1"
    batch_size: int = 5
    max_k: int = 64
    warmup_repeats: int = 3
    timing_repeats: int = 5
    num_threads: int = 4

    def __post_init__(self) -> None:
        if self.stage not in {"preflight", "reference", "initialization", "all"}:
            raise ValueError("Unknown stage.")
        if self.device != "cuda:1":
            raise ValueError("This paper audit is restricted to GPU:1.")
        for name in (
            "batch_size",
            "max_k",
            "timing_repeats",
            "num_threads",
            "warmup_repeats",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 9 <= self.max_k <= 64:
            raise ValueError("Require 9 <= max_k <= 64; no automatic extension.")


@dataclass
class SourceRun:
    example: str
    native_k: int
    audit: FrozenTangentCsvAudit
    spec: FrozenRunSpec
    generation: dict[str, Any]


def operator_fingerprint(spec: FrozenRunSpec) -> str:
    model, training = spec.configs.coupling_model, spec.configs.coupling_training
    identity: dict[str, Any] = {
        "green": spec.input_hashes[str(spec.green)],
        "geometry": spec.input_hashes[str(spec.geometry)],
        "coefficients": spec.input_hashes[str(spec.coefficients)],
        "test": [
            (p.name, spec.input_hashes[str(p)]) for p in sorted(spec.test.glob("*.npz"))
        ],
        "branch_input_dim": model.branch_input_dim,
        "coefficient_terms": asdict(model.coefficient_terms),
        "projection": asdict(
            BalanceProjectionConfig.from_raw(model.balance_projection)
        ),
        "reconstruction": asdict(
            ComplexCrossAxisReconstructionConfig.from_raw(
                model.cross_axis_reconstruction
            )
        ),
        "integration": training.integration_rule,
        "canonical": asdict(
            ComplexCanonicalEnergyConfig.from_raw(training.canonical_energy)
        ),
    }
    # K and neural architecture do not change the frozen equal-split operator.
    tangent = identity["projection"]["symmetric_tangent_green_response"]
    for key in ("subspace_dimension", "max_subspace_dimension", "geometry_k_selection"):
        tangent.pop(key, None)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


class SourceRunSession(_FrozenRunSession):
    """Reuse frozen loading and production tangent calls, not a second solver."""

    def __init__(self, run: SourceRun) -> None:
        super().__init__(run.audit, run.spec)
        self.run = run
        self.batches = [batch.to(self.device) for batch in self.loader]
        if not self.batches:
            raise ValueError("Empty evaluation dataset.")
        for batch in self.batches:
            if not bool(batch.has_solution.all() and batch.has_flux.all()):
                raise ValueError("Every sample must supply sol, phi, psi.")
        self._initialize_context(self.batches[0])
        for k in range(1, self.request.max_k + 1):
            self.projection_by_k[k] = replace(
                self.projection,
                symmetric_tangent_green_response=replace(
                    self.tangent,
                    subspace_dimension=k,
                    max_subspace_dimension=max(k, self.tangent.max_subspace_dimension),
                ),
            )
            self.context_by_k[k] = replace(self.context, subspace_dimension=k)

    def _initialize_context(self, batch: ComplexCouplingBatch) -> None:
        path = self.spec.run_dir / "tangent_response_context.safetensors"
        if path.exists():
            cache, info = verified_runtime_cache(path, self.green, batch, self.tangent)
            if cache is not None:
                self.audit.logger.warning(
                    "%s explicit runtime context rebuild: %s", self.spec.run_id, info
                )
                self.audit.shared_cache = cache
                self.audit.validated_sidecars.add(
                    (
                        _sha256(path),
                        _sha256(path.with_suffix(".json"))
                        if path.with_suffix(".json").exists()
                        else None,
                    )
                )
            self.verification["runtime_context_validation"] = info
        super()._initialize_context(batch)

    def equal_prepared(self, batch: ComplexCouplingBatch) -> PreparedTangentBatch:
        pair = physical_equal_split(batch.rhs_valid)
        response = self.context.response_operator.forward_pair(pair)
        mismatch = response[:, 0] - response[:, 1]
        return PreparedTangentBatch(
            pair, pair, mismatch, self.context.tangent_gradient(mismatch)
        )

    @torch.no_grad()
    def smoke(self) -> dict[str, Any]:
        batch = self.batches[0]
        prepared = self._prepare(batch)
        result = self._krylov(prepared, self.run.native_k)
        pair, solution, cross = self._candidate(
            batch, prepared, result, self.run.native_k
        )
        assert self.evaluator is not None
        native = self.evaluator.predict_batch(batch)
        torch.testing.assert_close(
            pair, native.projection.projected_physical, rtol=1e-8, atol=1e-12
        )
        torch.testing.assert_close(
            cross.u_pred_valid,
            native.cross_axis_reconstruction.u_pred_valid,
            rtol=1e-8,
            atol=1e-12,
        )
        balanced, residual = symmetric_source_balance(batch.flux_valid, batch.rhs_valid)
        self.fields(batch, balanced)
        for k in (0, 1, self.run.native_k):
            direct_pair = self.equal_corrected(batch, k)
            _, direct_cross = self.fields(batch, direct_pair)
            prediction = self.equal_prediction(batch, k)
            torch.testing.assert_close(
                direct_cross.u_pred_valid,
                prediction.u_pred_valid,
                rtol=1e-8,
                atol=1e-12,
            )
        return dict(
            run_id=self.spec.run_id,
            sample_count=len(batch.file_stems),
            native_fields_verified=True,
            equal_prediction_verified=True,
            reference_balance_max=float(residual.abs().max()),
        )

    def fields(
        self, batch: ComplexCouplingBatch, pair: torch.Tensor
    ) -> tuple[torch.Tensor, ComplexCrossAxisReconstructionResult]:
        solution = self.context.response_operator.forward_pair(pair)
        cross = self.cross_axis.reconstruct(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            projected_physical=pair,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        if (
            not torch.isfinite(solution).all()
            or not torch.isfinite(cross.u_pred_valid).all()
        ):
            raise RuntimeError("Non-finite reconstructed fields.")
        return solution, cross

    def metrics(
        self,
        batch: ComplexCouplingBatch,
        pair: torch.Tensor,
        condition: str,
        k: int,
        initial: torch.Tensor,
        result: KrylovSubspaceStepResult | None = None,
    ) -> list[dict[str, Any]]:
        solution, cross = self.fields(batch, pair)
        energy = canonical_complex_energy_loss(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            a_valid=batch.a_valid,
            geometry=batch.geometry,
            boundary_context=self.boundary,
        )
        residual = pair.sum(1) - batch.rhs_valid
        mass_root = float(self.context.point_mass) ** 0.5
        if condition != "reference_raw":
            torch.testing.assert_close(
                pair.sum(1), batch.rhs_valid, rtol=1e-12, atol=1e-12
            )
        rows = []
        for i, sample_id in enumerate(batch.sample_indices.tolist()):
            balance_relative, balance_status = relative_norm(
                residual[i], batch.rhs_valid[i]
            )
            row: dict[str, Any] = dict(
                example=self.run.example,
                fingerprint=self.spec.fingerprint,
                run_id=self.spec.run_id if condition == "learned" else "shared",
                seed=self.spec.seed if condition == "learned" else None,
                condition=condition,
                evaluation_k=k,
                sample_id=sample_id,
                file_stem=batch.file_stems[i],
                balance_l2=mass_root * float(residual[i].norm()),
                balance_relative=balance_relative,
                balance_relative_status=balance_status,
                balance_max_abs=float(residual[i].abs().max()),
                correction_pair_l2=mass_root * float((pair[i] - initial[i]).norm()),
                response_cost=float(
                    self.context.point_mass
                    * (solution[i, 0] - solution[i, 1]).square().sum()
                ),
                loss_energy_optimized=float(
                    energy.bulk_per_sample[i]
                    + self.canonical.boundary_weight * energy.boundary_per_sample[i]
                ),
                loss_energy_bulk=float(energy.bulk_per_sample[i]),
                loss_energy_boundary=float(energy.boundary_per_sample[i]),
                effective_dimension=int(result.direction_active[:k, i].sum())
                if result is not None
                else 0,
                direction_active=int(result.direction_active[k - 1, i])
                if result is not None
                else 0,
            )
            for key, value in (
                ("rel_sol", cross.u_pred_valid[i]),
                ("rel_sol_equal_mean", cross.u_equal_mean_valid[i]),
                ("rel_u_phi", solution[i, 0]),
                ("rel_u_psi", solution[i, 1]),
            ):
                row[key], row[f"{key}_status"] = relative_norm(
                    value - batch.sol_valid[i], batch.sol_valid[i]
                )
            if any(not np.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise RuntimeError("Non-finite audit metric.")
            rows.append(row)
        return rows

    @torch.no_grad()
    def reference_rows(self) -> list[dict[str, Any]]:
        rows = []
        for batch in self.batches:
            balanced, _ = symmetric_source_balance(batch.flux_valid, batch.rhs_valid)
            for condition, pair in (
                ("reference_raw", batch.flux_valid),
                ("reference_balanced", balanced),
            ):
                rows.extend(self.metrics(batch, pair, condition, 0, batch.flux_valid))
        return rows

    @torch.no_grad()
    def learned_rows(self) -> list[dict[str, Any]]:
        rows = []
        seen = set()
        for batch in self.batches:
            prepared = self._prepare(batch)
            result = self._krylov(prepared, self.run.native_k)
            physical, solution, cross = self._candidate(
                batch, prepared, result, self.run.native_k
            )
            assert self.evaluator is not None
            production = self.evaluator.predict_batch(batch)
            for actual, expected in (
                (physical, production.projection.projected_physical),
                (solution[:, 0], production.reconstruction.u_phi_valid),
                (solution[:, 1], production.reconstruction.u_psi_valid),
                (cross.u_pred_valid, production.cross_axis_reconstruction.u_pred_valid),
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
            batch_rows = self.metrics(
                batch,
                physical,
                "learned",
                self.run.native_k,
                prepared.symmetric_physical,
                result,
            )
            for row in batch_rows:
                key = row["sample_id"], row["file_stem"]
                if key in seen or key not in self.spec.artifacts:
                    raise ValueError(f"Unexpected native sample {key}.")
                seen.add(key)
                stored = self.spec.artifacts[key]
                for metric in (
                    "rel_sol",
                    "rel_sol_equal_mean",
                    "loss_energy_optimized",
                    "loss_energy_bulk",
                    "loss_energy_boundary",
                    "response_cost",
                ):
                    if metric == "rel_sol_equal_mean" and metric not in stored:
                        self.verification["artifact_equal_mean_available"] = False
                        continue
                    original = (
                        f"tangent_response_cost_k{self.run.native_k}"
                        if metric == "response_cost"
                        else metric
                    )
                    np.testing.assert_allclose(
                        row[metric],
                        float(stored[original]),
                        rtol=1e-8,
                        atol=1e-12,
                        err_msg=f"{self.spec.run_id} {key} {metric}",
                    )
            rows.extend(batch_rows)
        if seen != set(self.spec.artifacts):
            raise ValueError("Incomplete native artifact coverage.")
        self.verification.update(native_verified=True, baseline_samples=len(seen))
        return rows

    @torch.no_grad()
    def equal_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, batch in enumerate(self.batches):
            prepared = self.equal_prepared(batch)
            result = self._krylov(prepared, self.request.max_k)
            initial_cost = self.context.point_mass * prepared.mismatch.square().sum(1)
            previous = torch.cat((initial_cost[None], result.costs[:-1]))
            tolerance = (
                1e-10 * torch.maximum(previous, initial_cost[None])
                + torch.finfo(torch.float64).tiny
            )
            if torch.any(result.costs > previous + tolerance):
                raise RuntimeError(
                    "Response cost increased beyond production tolerance."
                )
            for k in range(self.request.max_k + 1):
                pair = (
                    prepared.symmetric_physical
                    if k == 0
                    else self._candidate(batch, prepared, result, k)[0]
                )
                rows.extend(
                    self.metrics(
                        batch,
                        pair,
                        "equal_split",
                        k,
                        prepared.symmetric_physical,
                        result if k else None,
                    )
                )
            self.audit.logger.info(
                "%s equal split batch %d/%d",
                self.run.example,
                index + 1,
                len(self.batches),
            )
        return rows

    @torch.no_grad()
    def validate_independent(self, ks: set[int], rows: list[dict[str, Any]]) -> None:
        expected = {(row["evaluation_k"], row["sample_id"]): row for row in rows}
        for k in sorted(ks):
            for batch in self.batches:
                prepared = self.equal_prepared(batch)
                pair = self.equal_corrected(batch, k)
                _, direct_cross = self.fields(batch, pair)
                prediction = self.equal_prediction(batch, k)
                torch.testing.assert_close(
                    direct_cross.u_pred_valid,
                    prediction.u_pred_valid,
                    rtol=1e-8,
                    atol=1e-12,
                )
                for row in self.metrics(
                    batch, pair, "equal_split", k, prepared.symmetric_physical
                ):
                    base = expected[k, row["sample_id"]]
                    for metric in (
                        "rel_sol",
                        "rel_sol_equal_mean",
                        "rel_u_phi",
                        "rel_u_psi",
                        "response_cost",
                        "loss_energy_optimized",
                    ):
                        np.testing.assert_allclose(
                            row[metric],
                            base[metric],
                            rtol=1e-8,
                            atol=1e-12,
                            err_msg=f"Independent prefix K={k} {metric}",
                        )
            self.audit.logger.info("%s independent K=%d verified", self.run.example, k)
        self.verification["independent_ks"] = sorted(ks)

    def equal_corrected(self, batch: ComplexCouplingBatch, k: int) -> torch.Tensor:
        pair = physical_equal_split(batch.rhs_valid)
        if k:
            prepared = self.equal_prepared(batch)
            delta = (
                replace(self.context, subspace_dimension=k)
                .tangent_step(
                    mismatch=prepared.mismatch,
                    gradient=prepared.gradient,
                )
                .delta
            )
            pair = torch.stack((pair[:, 0] + delta, pair[:, 1] - delta), dim=1)
        return pair

    def equal_prediction(
        self, batch: ComplexCouplingBatch, k: int
    ) -> ComplexCrossAxisReconstructionResult:
        if k == 0:
            return self.fields(batch, physical_equal_split(batch.rhs_valid))[1]
        # Use the same production projection/reconstruction as learned timing.
        raw = torch.stack(
            (
                batch.rhs_valid
                * 0.5
                * batch.geometry.x_lengths_for_valid_points().square(),
                batch.rhs_valid
                * 0.5
                * batch.geometry.y_lengths_for_valid_points().square(),
            ),
            dim=1,
        )
        projection = apply_complex_balance_projection(
            raw_response=raw,
            rhs_phys=batch.rhs_valid,
            geometry=batch.geometry,
            config=self.projection_by_k[k],
            symmetric_tangent_context=self.context_by_k[k],
        )
        reconstruction = reconstruct_complex_projection(
            projection=projection,
            green_model=self.green,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        return self.cross_axis.reconstruct(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            projected_physical=projection.projected_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )

    def close(self) -> None:
        super().close()
        self.batches.clear()


class SourceInitializationAudit:
    def __init__(self, request: SourceInitializationRequest) -> None:
        self.request = request
        self.outdir = request.outdir.resolve()
        self.runs: list[SourceRun] = []
        self.hashes: dict[str, str] = {}
        self.rows: list[dict[str, Any]] = []
        self.timings: list[dict[str, Any]] = []
        self.matches: list[dict[str, Any]] = []
        self.verification: dict[str, Any] = {"status": "running", "runs": []}

    def _generation_contract(
        self, spec: FrozenRunSpec, entry: dict[str, Any]
    ) -> dict[str, Any]:
        path = Path(entry["generation_summary"]).resolve()
        summary = json.loads(path.read_text())
        self.hashes[str(path)] = _sha256(path)
        recorded_geometry = Path(summary["config"]["geometry"]).resolve()
        geometry_path = Path(
            entry.get("generation_geometry_override", recorded_geometry)
        ).resolve()
        coefficient_path = Path(summary["config"]["coefficients"]).resolve()
        for old in (geometry_path, coefficient_path):
            self.hashes[str(old)] = _sha256(old)
        geometry = load_complex_geometry(spec.geometry, dtype=torch.float64)
        generated_geometry = load_complex_geometry(geometry_path, dtype=torch.float64)
        if geometry_path != recorded_geometry:
            if recorded_geometry.exists():
                raise ValueError(
                    "A geometry override must not conceal an available historical file."
                )
            metadata = summary["geometry_metadata"]
            with np.load(geometry_path, allow_pickle=False) as archive:
                for key in (
                    "domain_type",
                    "radius",
                    "center",
                    "step_size",
                    "boundary_tol",
                ):
                    np.testing.assert_equal(archive[key], metadata[key])
                grid_x, grid_y = np.meshgrid(archive["grid_x"], archive["grid_y"])
                center = np.asarray(metadata["center"])
                inside = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 < (
                    metadata["radius"] - metadata["boundary_tol"]
                ) ** 2
                yi, xi = np.nonzero(inside)
                np.testing.assert_array_equal(yi, archive["valid_grid_y_index"])
                np.testing.assert_array_equal(xi, archive["valid_grid_x_index"])
                np.testing.assert_array_equal(
                    np.column_stack((grid_x[inside], grid_y[inside])),
                    archive["coords_valid"],
                )
        for key in (
            "coords_valid",
            "valid_grid_x_index",
            "valid_grid_y_index",
            "x_segment_id",
            "y_segment_id",
            "x_segment_length",
            "y_segment_length",
        ):
            torch.testing.assert_close(
                getattr(geometry, key), getattr(generated_geometry, key), rtol=0, atol=0
            )
        current, original = (
            load_coefficient_functions(spec.coefficients),
            load_coefficient_functions(coefficient_path),
        )
        xy = geometry.coords_valid
        for key in ("a_fun", "apx_fun", "apy_fun", "bx_fun", "by_fun", "c_fun"):
            torch.testing.assert_close(
                getattr(current, key)(xy[:, 0], xy[:, 1]),
                getattr(original, key)(xy[:, 0], xy[:, 1]),
                rtol=1e-12,
                atol=1e-12,
            )
        files = sorted(spec.test.glob("*.npz"))
        records = [s for s in summary["samples"] if s["split"] == "test"]
        if len(records) != len(files) or {Path(s["path"]).name for s in records} != {
            p.name for p in files
        }:
            raise ValueError(f"{spec.test}: generation/test identities differ.")
        for file in files:
            with np.load(file, allow_pickle=False) as archive:
                if not {"rhs", "sol", "phi", "psi"} <= set(archive.files):
                    raise ValueError(f"{file}: missing reference arrays.")
                shapes = {archive[key].shape for key in ("rhs", "sol", "phi", "psi")}
                if len(shapes) != 1:
                    raise ValueError(f"{file}: unequal reference shapes.")
                for key in ("rhs", "sol", "phi", "psi"):
                    values = archive[key][
                        geometry.valid_grid_y_index.numpy(),
                        geometry.valid_grid_x_index.numpy(),
                    ]
                    if not np.isfinite(values).all():
                        raise ValueError(f"{file}: non-finite {key} at axial points.")
        return dict(
            summary_path=str(path),
            config=summary["config"],
            sample_count=len(files),
            recorded_generation_geometry=str(recorded_geometry),
            resolved_generation_geometry=str(geometry_path),
            historical_geometry_bytes_verified=geometry_path == recorded_geometry,
            current_geometry=str(spec.geometry),
            current_coefficients=str(spec.coefficients),
            geometry_coordinates_identical=True,
            coefficients_at_axial_points_agree=True,
            historical_source_code_identity="not_proven",
            formula="phi=-dx(a dx u)+bx dx u+c*u/2; psi=-dy(a dy u)+by dy u+c*u/2",
            generated_test_paths=[s["path"] for s in records],
            current_test_path=str(spec.test),
        )

    def preflight(self) -> None:
        manifest = json.loads(self.request.manifest.read_text())
        entries = manifest["runs"]
        expected = {
            (example, seed)
            for example in ("unit_square", "disk", "annulus", "pentagram")
            for seed in range(4)
        }
        if (
            len(entries) != 16
            or {(e["example"], e["seed"]) for e in entries} != expected
        ):
            raise ValueError(
                "Manifest must contain exactly four seeds for each example."
            )
        self.hashes[str(self.request.manifest.resolve())] = _sha256(
            self.request.manifest
        )
        for entry in entries:
            k = {"unit_square": 2, "disk": 2, "annulus": 4, "pentagram": 9}[
                entry["example"]
            ]
            if entry["native_k"] != k:
                raise ValueError("Manifest native K violates the paper protocol.")
            request = FrozenTangentCsvRequest(
                run_dirs=(Path(entry["run_dir"]),),
                outdir=self.outdir,
                device=self.request.device,
                baseline_k=k,
                max_k=self.request.max_k,
                batch_size=self.request.batch_size,
                num_threads=self.request.num_threads,
            )
            audit = FrozenTangentCsvAudit(request)
            spec = audit._preflight()[0]
            if spec.seed != entry["seed"]:
                raise ValueError("Manifest/checkpoint seed mismatch.")
            spec.run_id = f"{entry['example']}_seed{spec.seed}"
            spec.fingerprint = operator_fingerprint(spec)
            self.hashes.update(spec.input_hashes)
            generation = self._generation_contract(spec, entry)
            self.runs.append(SourceRun(entry["example"], k, audit, spec, generation))
        _json(
            self.outdir / "preflight.json",
            [
                dict(
                    example=r.example,
                    native_k=r.native_k,
                    run_id=r.spec.run_id,
                    fingerprint=r.spec.fingerprint,
                    generation=r.generation,
                    paths=dict(
                        green=r.spec.green,
                        geometry=r.spec.geometry,
                        coefficients=r.spec.coefficients,
                        test=r.spec.test,
                    ),
                    normalization=asdict(
                        BalanceProjectionConfig.from_raw(
                            r.spec.configs.coupling_model.balance_projection
                        )
                    ),
                )
                for r in self.runs
            ],
        )
        _json(self.outdir / "input_hashes.json", self.hashes)

    def _gpu_idle(self) -> None:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        ).strip()
        others = [
            row
            for row in rows.splitlines()
            if row.split(",")[0].strip() == uuid
            and int(row.split(",")[1]) != os.getpid()
        ]
        if others:
            raise RuntimeError(f"GPU:1 has other compute processes: {others}")

    @torch.no_grad()
    def _benchmark(
        self, session: SourceRunSession, equal_ks: set[int], models: dict[str, Any]
    ) -> None:
        clock = FrozenTangentBenchmark(session.device)
        conditions = [("learned", session.run.native_k, name) for name in models] + [
            ("equal_split", k, "shared") for k in sorted(equal_ks)
        ]
        for repeat in range(-self.request.warmup_repeats, self.request.timing_repeats):
            for condition, k, run_id in conditions[:: (-1 if repeat % 2 else 1)]:
                self._gpu_idle()
                if condition == "learned":
                    session.model = models[run_id]

                def forward() -> None:
                    for batch in session.batches:
                        if condition == "learned":
                            session._prediction_forward(batch, k)
                        else:
                            session.equal_prediction(batch, k)

                start_reserved = torch.cuda.memory_reserved(session.device) / 2**20
                measured = clock.measure(forward)
                self._gpu_idle()
                if repeat >= 0:
                    self.timings.append(
                        dict(
                            example=session.run.example,
                            fingerprint=session.spec.fingerprint,
                            run_id=run_id,
                            condition=condition,
                            evaluation_k=k,
                            repeat=repeat,
                            sample_count=len(session.dataset),
                            seconds_per_sample=measured["seconds"]
                            / len(session.dataset),
                            **measured,
                            start_reserved_mib=start_reserved,
                            peak_reserved_mib=torch.cuda.max_memory_reserved(
                                session.device
                            )
                            / 2**20,
                        )
                    )
                    _write_csv(self.outdir / "timing.csv", self.timings)
                session.audit.logger.info(
                    "%s %s K%d timing repeat %d complete",
                    session.spec.run_id,
                    condition,
                    k,
                    repeat,
                )

    def _save_rows(self) -> None:
        _write_csv(self.outdir / "per_sample.csv", self.rows)
        _write_csv(self.outdir / "summary.csv", summarize(self.rows))
        _write_csv(self.outdir / "matches.csv", self.matches)
        _json(self.outdir / "verification.json", self.verification)

    def run(self) -> None:
        if self.outdir.exists():
            raise FileExistsError(f"Refusing to overwrite {self.outdir}")
        self.outdir.mkdir(parents=True)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.request.num_threads)
            self.verification["stage"] = self.request.stage
            _json(self.outdir / "request.json", asdict(self.request))
            _json(
                self.outdir / "implementation_hashes.json",
                {
                    str(path): _sha256(path)
                    for path in (
                        Path(__file__),
                        Path(__file__).with_name("source_initialization_metrics.py"),
                        Path(__file__).with_name("source_initialization_report.py"),
                        Path(__file__).with_name("source_initialization_context.py"),
                        Path(__file__).with_name("complex_frozen_tangent_csv.py"),
                        Path(__file__).with_name("complex_tangent_projection.py"),
                        Path(__file__).with_name("complex_projection.py"),
                        Path(__file__).with_name(
                            "complex_cross_axis_reconstruction.py"
                        ),
                        Path(__file__).parent / "fenicsx_samples" / "solver.py",
                        Path("cli/audit_source_initialization.py"),
                        Path("PLAN.md"),
                    )
                },
            )
            self.preflight()
            host = self.runs[0].audit
            formatter = logging.Formatter("%(funcName)s - %(message)s")
            for handler in list(host.logger.handlers):
                host.logger.removeHandler(handler)
                handler.close()
            for handler in (
                RichHandler(
                    rich_tracebacks=True, show_path=True, omit_repeated_times=False
                ),
                logging.FileHandler(self.outdir / "run.log"),
            ):
                handler.setFormatter(formatter)
                host.logger.addHandler(handler)
            host.logger.setLevel(logging.DEBUG)
            host.logger.propagate = False
            for run in self.runs:
                run.audit.logger = host.logger
            _json(self.outdir / "environment.json", host._environment())
            if self.request.stage != "preflight":
                self._gpu_idle()
                groups: dict[str, list[SourceRun]] = {}
                for run in self.runs:
                    groups.setdefault(run.spec.fingerprint, []).append(run)
                for group in groups.values():
                    session = SourceRunSession(group[0])
                    try:
                        self.verification.setdefault("smoke", []).append(
                            session.smoke()
                        )
                        self._save_rows()
                    finally:
                        session.close()
                        del session
                        group[0].audit.shared_cache = None
                        gc.collect()
                        torch.cuda.empty_cache()
                if self.request.stage in {"all", "reference"}:
                    for group in groups.values():
                        session = SourceRunSession(group[0])
                        try:
                            self.rows.extend(session.reference_rows())
                            self.verification.setdefault("reference_groups", []).append(
                                group[0].spec.fingerprint
                            )
                            self._save_rows()
                        finally:
                            session.close()
                            del session
                            group[0].audit.shared_cache = None
                            gc.collect()
                            torch.cuda.empty_cache()
                for group in groups.values():
                    if self.request.stage == "reference":
                        break
                    models: dict[str, Any] = {}
                    targets = []
                    for run in group:
                        host.logger.info("Starting %s", run.spec.run_id)
                        session = SourceRunSession(run)
                        try:
                            learned = session.learned_rows()
                            self.rows.extend(learned)
                            targets.extend(summarize(learned))
                            models[run.spec.run_id] = session.model
                            self.verification["runs"].append(
                                session.verification | {"status": "complete"}
                            )
                            self._save_rows()
                        finally:
                            session.close()
                            del session
                            run.audit.shared_cache = None
                            gc.collect()
                            torch.cuda.empty_cache()
                    session = SourceRunSession(group[0])
                    try:
                        shared_rows = session.equal_rows()
                        self.rows.extend(shared_rows)
                        group_matches = [
                            match_accuracy(target, summarize(shared_rows))
                            for target in targets
                        ]
                        self.matches.extend(group_matches)
                        self._save_rows()
                        ks = {group[0].native_k, self.request.max_k}
                        ks.update(
                            m[key]
                            for m in group_matches
                            for key in ("mean_k", "mean_p95_k")
                            if m[key] is not None
                        )
                        session.validate_independent(ks, shared_rows)
                        self._benchmark(session, ks, models)
                        self.verification.setdefault("prefix_validations", []).append(
                            session.verification
                        )
                    finally:
                        models.clear()
                        session.close()
                        del session
                        group[0].audit.shared_cache = None
                        gc.collect()
                        torch.cuda.empty_cache()
            self.verification["coverage"] = verify_coverage(
                self.rows,
                self.timings,
                self.matches,
                [
                    dict(
                        example=r.example,
                        run_id=r.spec.run_id,
                        fingerprint=r.spec.fingerprint,
                        native_k=r.native_k,
                        sample_count=r.generation["sample_count"],
                    )
                    for r in self.runs
                ],
                self.request.stage,
                self.request.max_k,
                self.request.timing_repeats,
            )
            if self.request.stage != "preflight":
                SourceInitializationReport(self.outdir).write(
                    self.rows, self.timings, self.matches
                )
            self.verification["status"] = "complete"
        except Exception as exc:
            self.verification.update(
                status="failed", error_type=type(exc).__name__, error=str(exc)
            )
            raise
        finally:
            changed = [
                path
                for path, digest in self.hashes.items()
                if _sha256(Path(path)) != digest
            ]
            self.verification["changed_inputs"] = changed
            if changed:
                self.verification["status"] = "failed"
            self._save_rows()
            torch.set_num_threads(old_threads)
            if changed:
                raise RuntimeError(f"Input hashes changed: {changed}")
