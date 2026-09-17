"""Frozen best-energy audit for the twelve Example 1 trunk experiments."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import TangentContextCheckpointConfig
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)

ROOT = Path(__file__).resolve().parents[2]
BEST = "complex_coupling_model_best_energy.safetensors"
KINDS = {
    "off": "unit_square_trunk_off_seed",
    "on": "unit_square_trunk_on_seed",
    "wide_off": "unit_square_primary_w428_trunk_off_seed",
}
PARAMETERS = {"off": 560182, "on": 955994, "wide_off": 958018}
Row = dict[str, Any]


def read_csv(path: Path) -> list[Row]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def local_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    for prefix in (str(ROOT), "/Users/jjhong0608/Documents/ComplexGeometryGreenNet"):
        if path.is_relative_to(prefix):
            return ROOT / path.relative_to(prefix)
    raise ValueError(f"Unrecognized project path: {value}")


def relative_error(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape or not all(
        np.isfinite(x).all() for x in (pred, target)
    ):
        raise ValueError("Expected matching finite arrays.")
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), 1e-12))


def validate_samples(rows: list[Row]) -> None:
    if [int(r["sample_id"]) for r in rows] != list(range(100)) or [
        r["file_stem"] for r in rows
    ] != [f"sample_{i:06d}" for i in range(100)]:
        raise ValueError("Sample identity/order/coverage mismatch.")


def representative_samples(rows: list[Row]) -> tuple[Row, Row]:
    median = float(np.median([float(r["rel_sol"]) for r in rows]))
    distances = [abs(float(r["rel_sol"]) - median) for r in rows]
    tolerance = 8 * np.finfo(float).eps * max(abs(median), np.finfo(float).tiny)
    candidates = [
        r
        for r, d in zip(rows, distances, strict=True)
        if d <= min(distances) + tolerance
    ]
    return min(candidates, key=lambda r: int(r["sample_id"])), min(
        rows, key=lambda r: (-float(r["rel_sol"]), int(r["sample_id"]))
    )


def check_baseline(actual: Row, baseline: Row, identity: str) -> None:
    for key, value in actual.items():
        expected = float(baseline[key])
        if not np.isfinite(value) or not np.isfinite(expected):
            raise ValueError(f"Non-finite metric: {identity}/{key}")
        np.testing.assert_allclose(
            value,
            expected,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"{identity}/{key}",
        )


@dataclass(frozen=True)
class TrunkAuditRequest:
    run_root: Path
    outdir: Path
    device: str = "cpu"
    batch_size: int = 10
    num_threads: int = 4

    def __post_init__(self) -> None:
        if self.device != "cpu" or self.batch_size < 1 or self.num_threads < 1:
            raise ValueError(
                "This audit requires CPU and positive batch/thread counts."
            )


class UnitSquareTrunkAudit:
    def __init__(self, request: TrunkAuditRequest) -> None:
        self.request = request
        self.out = request.outdir.resolve()
        self.rows: list[Row] = []
        self.manifest: dict[str, str] = {}
        self.runs: list[Row] = []
        self.fields: dict[str, np.ndarray] = {}
        self.logger = logging.getLogger(__name__)

    def _record(self, path: Path) -> None:
        self.manifest[str(path.resolve())] = digest(path)

    def _preflight(self) -> None:
        if self.out.exists() and any(self.out.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty audit output.")
        if self.out.is_relative_to(self.request.run_root.resolve()):
            raise ValueError("Audit output must be outside the original run tree.")
        common: Row | None = None
        for kind, prefix in KINDS.items():
            for seed in range(4):
                run = self.request.run_root / f"{prefix}{seed}"
                raw = json.loads((run / "config_used.json").read_text())
                if raw["coupling_training"]["seed"] != seed:
                    raise ValueError("Training seed mismatch.")
                dataset = raw["dataset"]
                if common is not None and dataset != common:
                    raise ValueError("Dataset configuration differs across runs.")
                common = dataset
                projection = raw["coupling_model"]["balance_projection"]
                if projection["mode"] != "symmetric_tangent_green_response" or (
                    projection["symmetric_tangent_green_response"]["subspace_dimension"]
                    != 2
                ):
                    raise ValueError("Expected fixed tangent K=2.")
                for name in (
                    "config_used.json",
                    BEST,
                    "complex_training_metrics.csv",
                    "artifacts_best_energy/summary.json",
                    "artifacts_best_energy/metrics/per_sample_metrics.csv",
                    "artifacts_best_energy/data/selected_raw_arrays.npz",
                ):
                    self._record(run / name)
                sidecar = run / "tangent_response_context.safetensors"
                if sidecar.exists():
                    self._record(sidecar)
                baseline = read_csv(
                    run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
                )
                validate_samples(baseline)
                summary = json.loads(
                    (run / "artifacts_best_energy/summary.json").read_text()
                )
                if Path(summary["coupling_checkpoint"]).name != BEST:
                    raise ValueError("Artifact is not a best-energy evaluation.")
                for key in ("geometry_path", "coefficient_functions_path"):
                    self._record(local_path(dataset[key]))
                self._record(local_path(raw["pipeline"]["green_pretrained_path"]))
                tests = sorted(local_path(dataset["test_path"]).glob("*.npz"))
                if [p.stem for p in tests] != [r["file_stem"] for r in baseline]:
                    raise ValueError("Sample dataset mismatch.")
                for path in tests:
                    self._record(path)
                self._record(
                    local_path(dataset["test_path"]).parent / "generation_summary.json"
                )
        for folder in ("example_03_annulus", "example_04_pentagram"):
            for path in (ROOT / "docs/paper/numerical_examples" / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    self._record(path)
        self._record(ROOT / "PLAN.md")

    def _run_one(self, kind: str, seed: int) -> None:
        run = self.request.run_root / f"{KINDS[kind]}{seed}"
        configs = load_coupling_artifact_configs(run / "config_used.json")
        raw = configs.raw
        dataset_raw = raw["dataset"]
        device = torch.device("cpu")
        geometry = load_complex_geometry(
            local_path(dataset_raw["geometry_path"]), dtype=torch.float64
        )
        dataset = ComplexCouplingDataset(
            local_path(dataset_raw["test_path"]),
            geometry,
            load_coefficient_functions(
                local_path(dataset_raw["coefficient_functions_path"])
            ),
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=torch.float64,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        exporter = ComplexCouplingArtifactExporter(
            CouplingArtifactRequest(
                config=run / "config_used.json",
                coupling_checkpoint=run / BEST,
                green_checkpoint=local_path(raw["pipeline"]["green_pretrained_path"]),
                outdir=self.out,
                device="cpu",
            ),
            logger=self.logger,
        )
        model = exporter._load_complex_model(configs, device)
        green = exporter._load_green_model(configs, device)
        if sum(p.numel() for p in model.parameters()) != PARAMETERS[kind]:
            raise ValueError("Parameter count mismatch.")
        for network in (model, green):
            network.eval()
            for p in network.parameters():
                if p.dtype != torch.float64:
                    raise ValueError("Expected float64 model.")
                p.requires_grad_(False)
        training = replace(
            configs.coupling_training,
            tangent_context_checkpoint=TangentContextCheckpointConfig(
                enabled=True, load_policy="if_available", save_after_build=False
            ),
        )
        evaluator = ComplexCouplingEvaluator(
            model=model,
            green_model=green,
            config=training,
            device=device,
            work_dir=self.out / f"{kind}_seed{seed}",
            tangent_context_path=run / "tangent_response_context.safetensors",
        )
        baseline = read_csv(
            run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
        )
        base_by_stem = {r["file_stem"]: r for r in baseline}
        selected = read_csv(
            self.request.run_root
            / f"{KINDS['off']}0/artifacts_best_energy/metrics/per_sample_metrics.csv"
        )
        candidates = {r["file_stem"] for r in representative_samples(selected)}
        run_rows: list[Row] = []
        max_error = 0.0
        loader = DataLoader(
            dataset,
            batch_size=self.request.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=complex_coupling_collate_fn,
        )
        with torch.no_grad():
            for batch_index, cpu_batch in enumerate(loader):
                batch = cpu_batch.to(device)
                if not bool(batch.has_solution.all() and batch.has_flux.all()):
                    raise ValueError("Missing test reference.")
                result = evaluator.predict_batch(batch)
                cross = result.cross_axis_reconstruction
                arrays = {
                    "u_phi": result.reconstruction.u_phi_valid.cpu().numpy(),
                    "u_psi": result.reconstruction.u_psi_valid.cpu().numpy(),
                    "equal": cross.u_equal_mean_valid.cpu().numpy(),
                    "weak": cross.u_pred_valid.cpu().numpy(),
                }
                for i, stem in enumerate(batch.file_stems):
                    target = batch.sol_valid[i].cpu().numpy()
                    errors = {
                        k: relative_error(v[i], target) for k, v in arrays.items()
                    }
                    physical = result.projection.projected_physical[i].cpu().numpy()
                    reference = batch.flux_valid[i].cpu().numpy()
                    rel_flux = float(
                        np.mean(
                            [
                                relative_error(p, t)
                                for p, t in zip(physical, reference, strict=True)
                            ]
                        )
                    )
                    actual = dict(
                        rel_sol=errors["weak"],
                        rel_sol_equal_mean=errors["equal"],
                        rel_flux=rel_flux,
                    )
                    check_baseline(
                        actual, base_by_stem[stem], f"{kind}/seed{seed}/{stem}"
                    )
                    max_error = max(
                        max_error,
                        *(
                            abs(v - float(base_by_stem[stem][k]))
                            for k, v in actual.items()
                        ),
                    )
                    run_rows.append(
                        dict(
                            kind=kind,
                            seed=seed,
                            sample_id=int(batch.sample_indices[i]),
                            file_stem=stem,
                            rel_u_phi=errors["u_phi"],
                            rel_u_psi=errors["u_psi"],
                            **actual,
                        )
                    )
                    if seed == 0 and stem in candidates:
                        prefix = f"{kind}_{stem}_"
                        self.fields[prefix + "sol"] = target
                        self.fields[prefix + "coords_valid"] = (
                            geometry.coords_valid.cpu().numpy()
                        )
                        for k, v in arrays.items():
                            self.fields[prefix + k] = v[i]
                self.logger.info(
                    "%s seed%d batch%d: %d/100 baseline verified",
                    kind,
                    seed,
                    batch_index,
                    len(run_rows),
                )
        validate_samples(run_rows)
        self.rows.extend(run_rows)
        self.runs.append(
            dict(
                kind=kind,
                seed=seed,
                samples=len(run_rows),
                parameters=PARAMETERS[kind],
                training_device=str(configs.coupling_training.device),
                evaluation_device="cpu",
                max_baseline_abs_error=max_error,
                context=evaluator.symmetric_tangent_green_response_context_telemetry,
            )
        )
        write_csv(self.out / "per_sample.csv", self.rows)
        (self.out / "progress.json").write_text(
            json.dumps(self.runs, indent=2, default=str)
        )

    def run(self) -> None:
        self._preflight()
        self.out.mkdir(parents=True, exist_ok=True)
        handlers: list[logging.Handler] = [
            RichHandler(show_path=True, omit_repeated_times=False),
            logging.FileHandler(self.out / "evaluation.log"),
        ]
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)
        torch.set_num_threads(self.request.num_threads)
        (self.out / "input_manifest.json").write_text(
            json.dumps(self.manifest, indent=2)
        )
        sources = (
            Path(__file__),
            ROOT / "cli/audit_unit_square_trunk.py",
            ROOT / "src/greenonet/complex_coupling_model.py",
            ROOT / "src/greenonet/complex_coupling_evaluator.py",
        )
        (self.out / "execution_context.json").write_text(
            json.dumps({str(p.relative_to(ROOT)): digest(p) for p in sources}, indent=2)
        )
        try:
            for kind in KINDS:
                for seed in range(4):
                    self._run_one(kind, seed)
                    gc.collect()
            for path, expected in self.manifest.items():
                if digest(Path(path)) != expected:
                    raise ValueError(f"Frozen input changed: {path}")
            np.savez_compressed(
                self.out / "selected_fields.npz", allow_pickle=False, **self.fields
            )
            verification = dict(
                status="complete",
                samples=len(self.rows),
                runs=self.runs,
                original_inputs_unchanged=True,
                device="cpu",
                dtype="float64",
                batch_size=self.request.batch_size,
                num_threads=self.request.num_threads,
                python=platform.python_version(),
                torch=torch.__version__,
                platform=platform.platform(),
            )
            (self.out / "verification.json").write_text(
                json.dumps(verification, indent=2, default=str)
            )
        finally:
            for handler in handlers:
                self.logger.removeHandler(handler)
                handler.close()
