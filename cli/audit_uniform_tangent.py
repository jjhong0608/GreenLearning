"""CSV-only frozen uniform versus separable preconditioner screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import build_boundary_energy_context
from greenonet.complex_projection_response_audit import ComplexProjectionResponseAudit
from greenonet.complex_tangent_projection import SymmetricTangentGreenResponseContext
from greenonet.complex_tangent_subspace_audit import (
    ComplexTangentSubspaceAudit,
    TangentSubspaceAuditRequest,
)
from greenonet.coupling_artifacts import load_coupling_artifact_configs


def uniform_context(
    context: SymmetricTangentGreenResponseContext,
) -> SymmetricTangentGreenResponseContext:
    """Override only the active denominator; never serialize as a production cache."""
    denominator = context.separable_denominator
    if not torch.isfinite(denominator).all() or not (denominator > 0).all():
        raise ValueError("Uniform screening requires a finite positive denominator.")
    return replace(context, denominator=denominator.mean().expand_as(denominator))


class UniformTangentAudit(ComplexTangentSubspaceAudit):
    """Reuse production operators, subspace corrections, and metric definitions."""

    include_identity = False

    @torch.no_grad()
    def run(self) -> dict:
        out = self.request.outdir
        if (out / "per_sample.csv").exists():
            raise FileExistsError(f"Refusing to overwrite audit: {out}")
        out.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(self.request.config)
        config = self._configs
        geometry_path = self.request.geometry or config.dataset.geometry_path
        test_path = self.request.test_path or config.dataset.test_path
        coefficient_path = (
            self.request.coefficients or config.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")
        self._device = torch.device(self.request.device or "cpu")
        self.geometry = load_complex_geometry(geometry_path, dtype=config.dataset.dtype)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            load_coefficient_functions(coefficient_path),
            branch_input_dim=config.coupling_model.branch_input_dim,
            dtype=config.dataset.dtype,
            coefficient_terms=config.coupling_model.coefficient_terms,
            integration_rule=config.coupling_training.integration_rule,
        )
        if not len(dataset):
            raise ValueError("Empty test dataset.")
        self._load_models()
        self._cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            config.coupling_model.cross_axis_reconstruction
        )
        self.boundary_context = build_boundary_energy_context(self.geometry)
        edges = ComplexProjectionResponseAudit.build_transition_edges(
            self.geometry, threshold=self.request.transition_log_threshold
        )
        rows = []
        balance_max = 0.0
        seen = 0
        for batch in DataLoader(
            dataset,
            batch_size=self.request.batch_size,
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        ):
            batch = batch.to(self._device)
            self._initialize_context(batch)
            prepared = self._prepare_batch(batch)
            separable = self.tangent_context.with_preconditioner_variant("separable")
            contexts = [
                ("separable", separable),
                ("uniform", uniform_context(separable)),
            ]
            if self.include_identity:
                contexts.append(
                    (
                        "identity",
                        replace(
                            separable,
                            denominator=torch.ones_like(separable.denominator),
                        ),
                    )
                )
            for name, context in contexts:
                evaluation, krylov = self._evaluate_prepared_batch(
                    batch, prepared, context=context
                )
                for tensor in (
                    evaluation.candidate_physical,
                    evaluation.candidate_solution,
                    krylov.deltas,
                ):
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(f"Non-finite {name} result.")
                balance = evaluation.candidate_physical.sum(dim=2) - batch.rhs_valid
                maximum = float(balance.abs().max())
                balance_max = max(balance_max, maximum)
                tolerance = (
                    256
                    * torch.finfo(balance.dtype).eps
                    * max(1.0, float(evaluation.candidate_physical.abs().max()))
                )
                if maximum > tolerance:
                    raise RuntimeError(f"Balance failed: {maximum} > {tolerance}")
                method_rows = self._metric_rows(
                    batch, evaluation, krylov, edges, context=context
                )
                for row in method_rows:
                    row["preconditioner"] = name
                    row["active_denominator_min"] = float(context.denominator.min())
                    row["active_denominator_max"] = float(context.denominator.max())
                rows.extend(method_rows)
            seen += batch.rhs_valid.shape[0]
            if self.logger:
                self.logger.info("Evaluated %d/%d test samples", seen, len(dataset))
        self._write_csv(out / "per_sample.csv", rows)
        summaries = []
        for name in (
            ("separable", "uniform", "identity")
            if self.include_identity
            else ("separable", "uniform")
        ):
            for method in self.methods:
                group = [
                    row
                    for row in rows
                    if row["preconditioner"] == name
                    and row["method_id"] == method.method_id
                ]
                for metric in group[0]:
                    values = [row[metric] for row in group]
                    if not all(isinstance(value, (int, float)) for value in values):
                        continue
                    array = np.asarray(values, dtype=float)
                    finite = array[np.isfinite(array)]
                    if not len(finite):
                        continue
                    summaries.append(
                        {
                            "preconditioner": name,
                            "method": method.method_id,
                            "metric": metric,
                            "count": len(finite),
                            "mean": float(finite.mean()),
                            "p95": float(np.quantile(finite, 0.95)),
                            "max": float(finite.max()),
                        }
                    )
        self._write_csv(out / "summary.csv", summaries)
        inputs = [
            self.request.config,
            self.request.coupling_checkpoint,
            self.request.green_checkpoint,
            Path(geometry_path),
            Path(coefficient_path),
            *sorted(Path(test_path).glob("*.npz")),
            Path(__file__),
            Path("src/greenonet/complex_tangent_projection.py"),
        ]
        summary = {
            "samples": seen,
            "device": str(self._device),
            "threads": torch.get_num_threads(),
            "primary_comparison_k": 2,
            "max_k": self.request.max_subspace_dimension,
            "subspace_relative_eps": self.request.subspace_relative_eps,
            "uniform_formula": "D_uniform = mean(D_separable) * I",
            "uniform_is_audit_only_override": True,
            "identity_included": self.include_identity,
            "identity_formula": "D=I; active denominator exactly one; no rescaling",
            "reference_used_for_correction": False,
            "training_performed": False,
            "maximum_balance_error": balance_max,
            "operator_equivalence_max_abs": self._operator_equivalence_max_abs,
            "context_build_count": self._context_build_count,
            "input_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in inputs
            },
        }
        (out / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "coupling-checkpoint", "green-checkpoint", "outdir"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--subspace-relative-eps", type=float, default=1e-12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uniform_tangent_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        RichHandler(show_path=False),
        logging.FileHandler(args.outdir / "audit.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    audit = UniformTangentAudit(
        TangentSubspaceAuditRequest(
            config=args.config,
            coupling_checkpoint=args.coupling_checkpoint,
            green_checkpoint=args.green_checkpoint,
            outdir=args.outdir,
            device=args.device,
            batch_size=args.batch_size,
            max_subspace_dimension=4,
            subspace_relative_eps=args.subspace_relative_eps,
            save_generated_data=False,
        ),
        logger=logger,
    )
    audit.include_identity = args.include_identity
    result = audit.run()
    logger.info(
        "Completed %d samples; provenance: %s",
        result["samples"],
        args.outdir / "provenance.json",
    )


if __name__ == "__main__":
    main()
