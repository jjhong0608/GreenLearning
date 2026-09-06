"""Sequential, read-only frozen-checkpoint tangent sweeps and CSV evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import (
    build_boundary_energy_context,
    canonical_complex_energy_loss,
    relative_l2_valid,
)
from greenonet.complex_projection import (
    apply_complex_balance_projection,
    reconstruct_complex_projection,
)
from greenonet.complex_tangent_geometry_selection import (
    AxialSegmentTopologyAnalyzer,
    geometry_k_reach_metric,
    pointwise_reach_fraction,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    SymmetricTangentGreenResponseContextCache,
    matrix_free_krylov_subspace_step,
)
from greenonet.complex_tangent_subspace_audit import (
    PreparedTangentBatch,
    prepare_tangent_audit_batch,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    SymmetricTangentGreenResponseProjectionConfig,
    TangentContextCheckpointConfig,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.frozen_tangent_csv_metrics import (
    CsvRow,
    add_paired_metrics,
    aggregate_samples,
    aggregate_seeds,
)

AUDIT_ID = "sequential_frozen_tangent_csv_v1"
BEST_CHECKPOINT = "complex_coupling_model_best_energy.safetensors"
TIMING_COLUMNS = (
    "run_id",
    "seed",
    "evaluation_k",
    "scope",
    "repeat",
    "sample_count",
    "seconds",
    "seconds_per_sample",
    "start_allocated_mib",
    "peak_allocated_mib",
    "memory_status",
)


@dataclass(frozen=True)
class FrozenTangentCsvRequest:
    run_dirs: tuple[Path, ...]
    outdir: Path
    device: str
    baseline_k: int = 10
    max_k: int = 64
    batch_size: int = 10
    num_threads: int = 4
    benchmark: bool = False
    warmup_repeats: int = 3
    timing_repeats: int = 5
    green_checkpoint: Path | None = None
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.run_dirs:
            raise ValueError("At least one run directory is required.")
        for name in (
            "baseline_k",
            "max_k",
            "batch_size",
            "num_threads",
            "warmup_repeats",
            "timing_repeats",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < (0 if name == "warmup_repeats" else 1):
                raise ValueError(f"Invalid {name}: {value}")
        if not 2 <= self.baseline_k <= self.max_k:
            raise ValueError("Require 2 <= baseline_k <= max_k.")
        if self.device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", self.device):
            raise ValueError("Use cpu or an explicit CUDA device, e.g. cuda:1.")
        if type(self.benchmark) is not bool or type(self.overwrite) is not bool:
            raise TypeError("benchmark and overwrite must be booleans.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[CsvRow], fields: Sequence[str] = ()) -> None:
    keys = list(fields) or sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class FrozenRunSpec:
    run_dir: Path
    run_id: str
    seed: int
    configs: CouplingArtifactConfigs
    green: Path
    geometry: Path
    test: Path
    coefficients: Path
    artifacts: dict[tuple[int, str], dict[str, str]]
    input_hashes: dict[str, str]
    fingerprint: str


class FrozenTangentBenchmark:
    """Measure independent calls, not the time to extract a cached K-prefix."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, call: Callable[[], object]) -> dict[str, Any]:
        self.synchronize()
        start_memory: float | None = None
        if self.device.type == "cuda":
            start_memory = torch.cuda.memory_allocated(self.device) / 2**20
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = call()
        self.synchronize()
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated(self.device) / 2**20
            if self.device.type == "cuda"
            else None
        )
        del result
        return dict(
            seconds=elapsed,
            start_allocated_mib=start_memory,
            peak_allocated_mib=peak,
            memory_status="cuda_allocated" if peak is not None else "cpu_unmeasured",
        )


class FrozenTangentCsvAudit:
    def __init__(self, request: FrozenTangentCsvRequest) -> None:
        self.request = request
        self.outdir = request.outdir.resolve()
        self.device = torch.device(request.device)
        self.rows: list[CsvRow] = []
        self.timings: list[CsvRow] = []
        self.verification: dict[str, Any] = {"status": "running", "runs": []}
        self.metadata: dict[str, Any] = {"audit_id": AUDIT_ID, "status": "running"}
        self.shared_cache: SymmetricTangentGreenResponseContextCache | None = None
        self.validated_sidecars: set[tuple[str, str | None]] = set()
        self.logger = logging.getLogger(AUDIT_ID)

    def _initialize_output(self) -> None:
        inputs = [path.resolve() for path in self.request.run_dirs]
        if any(self.outdir == path or self.outdir in path.parents for path in inputs):
            raise ValueError(
                "Output must not replace or contain an input run directory."
            )
        if self.outdir.exists() and any(self.outdir.iterdir()):
            marker = self.outdir / "metadata.json"
            if (
                not self.request.overwrite
                or not marker.is_file()
                or json.loads(marker.read_text()).get("audit_id") != AUDIT_ID
            ):
                raise FileExistsError(
                    "Nonempty output requires --overwrite and this audit's metadata."
                )
            if marker.is_symlink():
                raise ValueError("Output metadata must not be a symlink.")
            if any(path.is_symlink() for path in self.outdir.rglob("*")):
                raise ValueError(
                    "Refuse to overwrite an audit tree containing symlinks."
                )
        self.outdir.mkdir(parents=True, exist_ok=True)
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        for handler in (
            RichHandler(
                rich_tracebacks=True, show_path=True, omit_repeated_times=False
            ),
            logging.FileHandler(self.outdir / "run.log", mode="w"),
        ):
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()
        for name in (
            "posthoc_per_sample.csv",
            "posthoc_per_seed.csv",
            "posthoc_summary.csv",
        ):
            _write_csv(self.outdir / name, [], ("run_id", "evaluation_k"))
        _write_csv(self.outdir / "posthoc_timing.csv", [], TIMING_COLUMNS)
        _json(self.outdir / "metadata.json", self.metadata)
        _json(self.outdir / "verification.json", self.verification)

    def _preflight(self) -> list[FrozenRunSpec]:
        runs: list[FrozenRunSpec] = []
        for index, path in enumerate(self.request.run_dirs):
            path = path.resolve()
            config_path = path / "config_used.json"
            configs = load_coupling_artifact_configs(config_path)
            model, dataset, training = (
                configs.coupling_model,
                configs.dataset,
                configs.coupling_training,
            )
            projection = BalanceProjectionConfig.from_raw(model.balance_projection)
            tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
                projection.symmetric_tangent_green_response
            )
            if (
                dataset.geometry_mode != "complex"
                or not projection.enabled
                or projection.mode != "symmetric_tangent_green_response"
                or tangent.subspace_dimension != self.request.baseline_k
            ):
                raise ValueError(
                    f"{path}: requires complex tangent training K={self.request.baseline_k}."
                )
            if any(
                dtype != torch.float64
                for dtype in (dataset.dtype, model.dtype, configs.green_model.dtype)
            ):
                raise ValueError(
                    f"{path}: float64 is required without implicit conversion."
                )
            seed = training.seed
            if seed is None:
                raise ValueError(f"{path}: an explicit training seed is required.")
            green = self.request.green_checkpoint or configs.raw.get(
                "pipeline", {}
            ).get("green_pretrained_path")
            geometry = self.request.geometry or dataset.geometry_path
            test = self.request.test_path or dataset.test_path
            coefficients = (
                self.request.coefficients or dataset.coefficient_functions_path
            )
            if any(value is None for value in (green, geometry, test, coefficients)):
                raise ValueError(
                    f"{path}: Green checkpoint, geometry, test and coefficients are required."
                )
            assert (
                green is not None
                and geometry is not None
                and test is not None
                and coefficients is not None
            )
            green, geometry, test, coefficients = (
                Path(value).resolve() for value in (green, geometry, test, coefficients)
            )
            artifact_dir = path / "artifacts_best_energy"
            artifact_csv = artifact_dir / "metrics" / "per_sample_metrics.csv"
            summary_path = artifact_dir / "summary.json"
            files = [
                config_path,
                path / BEST_CHECKPOINT,
                green,
                geometry,
                coefficients,
                artifact_csv,
                summary_path,
            ]
            if not test.is_dir():
                raise FileNotFoundError(test)
            sample_files = sorted(test.glob("*.npz"))
            if not sample_files:
                raise ValueError(f"Empty test dataset: {test}")
            files += sample_files
            sidecar = path / "tangent_response_context.safetensors"
            files += [
                file
                for file in (sidecar, sidecar.with_suffix(".json"))
                if file.exists()
            ]
            hashes = {str(file): _sha256(file) for file in files}
            summary = json.loads(summary_path.read_text())
            if Path(summary.get("coupling_checkpoint", "")).name != BEST_CHECKPOINT:
                raise ValueError(f"{summary_path}: not a best-energy artifact.")
            artifacts: dict[tuple[int, str], dict[str, str]] = {}
            for row in _read_csv(artifact_csv):
                required = {
                    "sample_id",
                    "file_stem",
                    "rel_sol",
                    "rel_flux",
                    "loss_energy_optimized",
                    "loss_energy_bulk",
                    "loss_energy_boundary",
                    f"tangent_response_cost_k{self.request.baseline_k}",
                }
                if missing := required - row.keys():
                    raise ValueError(
                        f"{artifact_csv}: missing baseline columns {sorted(missing)}"
                    )
                key = (int(row["sample_id"]), row["file_stem"])
                if key in artifacts:
                    raise ValueError(f"Duplicate sample in {artifact_csv}: {key}")
                artifacts[key] = row
            identity = dict(
                model=asdict(model),
                integration_rule=training.integration_rule,
                canonical_energy=asdict(
                    ComplexCanonicalEnergyConfig.from_raw(training.canonical_energy)
                ),
                green=hashes[str(green)],
                geometry=hashes[str(geometry)],
                coefficients=hashes[str(coefficients)],
                samples=[(file.name, hashes[str(file)]) for file in sample_files],
            )
            fingerprint = hashlib.sha256(
                json.dumps(identity, sort_keys=True, default=str).encode()
            ).hexdigest()
            runs.append(
                FrozenRunSpec(
                    path,
                    f"run{index:02d}_seed{seed}",
                    seed,
                    configs,
                    green,
                    geometry,
                    test,
                    coefficients,
                    artifacts,
                    hashes,
                    fingerprint,
                )
            )
        if len({run.run_dir for run in runs}) != len(runs) or len(
            {run.seed for run in runs}
        ) != len(runs):
            raise ValueError("Duplicate run directories or training seeds.")
        if len({run.fingerprint for run in runs}) != 1:
            raise ValueError(
                "Runs differ in model/operator/energy settings or Green/geometry/coefficient/test contents."
            )
        return runs

    def _environment(self) -> dict[str, Any]:
        hardware: dict[str, Any] = {
            "device": str(self.device),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "hostname": platform.node(),
        }
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            hardware.update(
                gpu_name=properties.name,
                gpu_uuid=str(getattr(properties, "uuid", "unknown")),
                gpu_total_memory=properties.total_memory,
            )
        return hardware | dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            num_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            compile=False,
            dtype="float64",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            environment={
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        )

    def run(self) -> dict[str, Any]:
        self._initialize_output()
        old_threads = torch.get_num_threads()
        try:
            if self.device.type == "cuda" and (
                not torch.cuda.is_available()
                or self.device.index is None
                or self.device.index >= torch.cuda.device_count()
            ):
                raise ValueError(
                    f"Requested device unavailable: {self.device}; no fallback."
                )
            torch.set_num_threads(self.request.num_threads)
            runs = self._preflight()
            self.metadata.update(
                request=asdict(self.request),
                environment=self._environment(),
                execution_mode="eager",
                reference_used_for_correction=False,
                response_cost_formula="hx*hy*sum((u_phi-u_psi)^2); no half factor",
                implementation_sha256={
                    name: _sha256(Path(__file__).with_name(name))
                    for name in (
                        "complex_frozen_tangent_csv.py",
                        "frozen_tangent_csv_metrics.py",
                        "complex_tangent_subspace_audit.py",
                        "complex_tangent_projection.py",
                        "complex_projection.py",
                        "complex_cross_axis_reconstruction.py",
                    )
                },
                tolerances=dict(
                    baseline_metric_rtol=1e-8,
                    baseline_metric_atol=1e-14,
                    baseline_tensor_rtol=1e-10,
                    baseline_tensor_atol=1e-12,
                    response_nonincrease_relative=1e-10,
                    balance_machine_eps_multiplier=64,
                ),
                runs=[
                    dict(
                        run_id=run.run_id,
                        seed=run.seed,
                        run_dir=str(run.run_dir),
                        input_hashes=run.input_hashes,
                        fingerprint=run.fingerprint,
                        original_config=json.loads(
                            (run.run_dir / "config_used.json").read_text()
                        ),
                        resolved_paths=dict(
                            green=str(run.green),
                            geometry=str(run.geometry),
                            test=str(run.test),
                            coefficients=str(run.coefficients),
                        ),
                    )
                    for run in runs
                ],
            )
            _json(self.outdir / "metadata.json", self.metadata)
            geometry = load_complex_geometry(runs[0].geometry, dtype=torch.float64)
            started = time.perf_counter()
            topology = AxialSegmentTopologyAnalyzer.from_geometry(geometry).analyze()
            self.metadata["topology_setup_seconds"] = time.perf_counter() - started
            self.reach: dict[int, CsvRow] = {}
            for k in range(self.request.baseline_k, self.request.max_k + 1):
                metric = geometry_k_reach_metric(topology, k)
                self.reach[k] = dict(
                    global_reach=metric.global_reach_fraction,
                    lower_5pct_reach=metric.pointwise_tail_reach_fraction,
                    minimum_reach=float(pointwise_reach_fraction(topology, k).min()),
                    full_reach_k=topology.a_graph_diameter + 1,
                )
            for spec in runs:
                self._run_one(spec)
                _write_csv(self.outdir / "posthoc_per_sample.csv", self.rows)
                _write_csv(
                    self.outdir / "posthoc_timing.csv", self.timings, TIMING_COLUMNS
                )
                _json(self.outdir / "verification.json", self.verification)
            for spec in runs:
                for path, expected in spec.input_hashes.items():
                    if _sha256(Path(path)) != expected:
                        raise RuntimeError(f"Input changed during audit: {path}")
            per_seed = aggregate_samples(self.rows)
            self._attach_timing_summaries(per_seed)
            _write_csv(self.outdir / "posthoc_per_seed.csv", per_seed)
            _write_csv(self.outdir / "posthoc_summary.csv", aggregate_seeds(per_seed))
            self.verification.update(status="complete", inputs_unchanged=True)
            self.metadata["status"] = "complete"
            self._write_readme()
            self.logger.info(
                "Complete: %d runs, %d scalar rows, no figures or field archives",
                len(runs),
                len(self.rows),
            )
        except Exception as error:
            self.metadata.update(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
            self.verification.update(status="failed", error=str(error))
            _write_csv(self.outdir / "posthoc_per_sample.csv", self.rows)
            _write_csv(self.outdir / "posthoc_timing.csv", self.timings, TIMING_COLUMNS)
            self.logger.exception(
                "Audit failed; partial rows are not a completed aggregate"
            )
            raise
        finally:
            torch.set_num_threads(old_threads)
            _json(self.outdir / "metadata.json", self.metadata)
            _json(self.outdir / "verification.json", self.verification)
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()
        return self.metadata

    def _run_one(self, spec: FrozenRunSpec) -> None:
        self.logger.info(
            "Starting %s: %s; Eager float64 K=%d..%d",
            spec.run_id,
            spec.run_dir,
            self.request.baseline_k,
            self.request.max_k,
        )
        started = time.perf_counter()
        session = _FrozenRunSession(self, spec)
        session.verification["model_and_data_setup_seconds"] = (
            time.perf_counter() - started
        )
        try:
            session.evaluate()
            if self.request.benchmark:
                session.benchmark()
            self.verification["runs"].append(
                session.verification | {"status": "complete"}
            )
        finally:
            session.close()
        self.logger.info("Finished %s", spec.run_id)

    def _attach_timing_summaries(self, rows: list[CsvRow]) -> None:
        for row in rows:
            for scope in ("tangent_only", "prediction_forward"):
                values = [
                    item["seconds"]
                    for item in self.timings
                    if item["run_id"] == row["run_id"]
                    and item["evaluation_k"] == row["evaluation_k"]
                    and item["scope"] == scope
                ]
                row[f"{scope}_seconds_median"] = (
                    float(np.median(values)) if values else None
                )
                row[f"{scope}_seconds_p95"] = (
                    float(np.quantile(values, 0.95)) if values else None
                )

    def _write_readme(self) -> None:
        (self.outdir / "README.md").write_text(
            "# Frozen tangent CSV audit\n\n"
            "Use tables only when metadata.json and verification.json say complete.\n"
            "All error, reach and gain values are fractions, not percentages. Blank cells mean NA.\n"
            "response_cost = hx*hy*sum((u_phi-u_psi)^2), without a half factor.\n"
            "rel_sol uses the configured weak blend; rel_sol_equal_mean uses the equal mean.\n"
            "rel_flux is the existing mean of the two directional-source relative errors, not flux-vector error.\n"
            "correction_pair_norm is ||(delta,-delta)||_2; correction_ratio divides by the original symmetric pair.\n"
            "direction_index is zero-based; direction_active concerns the last direction of the current prefix.\n"
            "effective_dimension counts all active directions in that prefix.\n"
            "Per-seed *_ratio_previous and *_ratio_baseline are ratios of sample means.\n"
            "*_ratio_previous_mean and *_ratio_baseline_mean are means of sample ratios.\n"
            "Gain is 1-ratio. No ratio is defined for a zero denominator.\n"
            "Summary uses equally weighted seed summaries and sample SD (ddof=1), never pooled test replicas.\n"
            "The same reference test set is shared across seeds; it is not independent replicated data.\n"
            "Accuracy computes nested prefixes once; timings independently recompute each K.\n"
            "One timing repeat is the sum of batch calls over the entire test set, in seconds.\n"
            "Transfers, setup, reference metrics and CSV writes are outside measured calls.\n"
            "tangent_only starts from prepared mismatch/gradient; prediction_forward includes the network and weak blend.\n"
            "Production MGS, finite checks and small K-by-K orthogonality diagnostics remain included.\n"
            "CUDA memory is allocated MiB (not reserved memory); CPU peak memory is unmeasured, not zero.\n"
            "Topology reach is a geometry-only correlation proxy, not an accuracy bound.\n"
            "No reference-selected best K, adaptive stopping, retraining or timing comparison to training logs.\n"
            "Baseline evaluator logs are retained in baseline_logs/. Input files are read-only.\n"
        )


class _FrozenRunSession:
    def __init__(self, audit: FrozenTangentCsvAudit, spec: FrozenRunSpec) -> None:
        self.audit, self.spec = audit, spec
        self.request, self.device = audit.request, audit.device
        self.verification: dict[str, Any] = dict(
            run_id=spec.run_id,
            seed=spec.seed,
            baseline_samples=0,
            baseline_max_abs=0.0,
            balance_max_abs=0.0,
            status="running",
        )
        self.evaluator: ComplexCouplingEvaluator | None = None
        geometry = load_complex_geometry(spec.geometry, dtype=torch.float64)
        self.dataset = ComplexCouplingDataset(
            spec.test,
            geometry,
            load_coefficient_functions(spec.coefficients),
            branch_input_dim=spec.configs.coupling_model.branch_input_dim,
            dtype=torch.float64,
            coefficient_terms=spec.configs.coupling_model.coefficient_terms,
            integration_rule=spec.configs.coupling_training.integration_rule,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.request.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=complex_coupling_collate_fn,
        )
        loader = ComplexCouplingArtifactExporter(
            CouplingArtifactRequest(
                config=spec.run_dir / "config_used.json",
                coupling_checkpoint=spec.run_dir / BEST_CHECKPOINT,
                green_checkpoint=spec.green,
                outdir=audit.outdir,
                device=str(self.device),
            ),
            logger=audit.logger,
        )
        self.model = loader._load_complex_model(spec.configs, self.device)
        self.green = loader._load_green_model(spec.configs, self.device)
        for model in (self.model, self.green):
            model.eval()
            for parameter in model.parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.float64:
                    raise ValueError("Loaded checkpoint is not float64.")
                parameter.requires_grad_(False)
        self.projection = BalanceProjectionConfig.from_raw(
            spec.configs.coupling_model.balance_projection
        )
        self.tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            self.projection.symmetric_tangent_green_response
        )
        self.cross_axis = ComplexCrossAxisReconstructor(
            spec.configs.coupling_model.cross_axis_reconstruction
        )
        self.boundary = build_boundary_energy_context(geometry)
        self.canonical = ComplexCanonicalEnergyConfig.from_raw(
            spec.configs.coupling_training.canonical_energy
        )

    def _initialize_context(self, batch: ComplexCouplingBatch) -> None:
        sidecar = self.spec.run_dir / "tangent_response_context.safetensors"
        persistence = TangentContextCheckpointConfig(
            enabled=True, load_policy="if_available", save_after_build=False
        )
        own_cache = SymmetricTangentGreenResponseContextCache(
            self.tangent, checkpoint=persistence, checkpoint_path=sidecar
        )
        key = (
            (
                _sha256(sidecar),
                _sha256(sidecar.with_suffix(".json"))
                if sidecar.with_suffix(".json").exists()
                else None,
            )
            if sidecar.exists()
            else None
        )
        if self.audit.shared_cache is None or (
            key is not None and key not in self.audit.validated_sidecars
        ):
            own_cache.get_or_build(
                green_model=self.green,
                geometry=batch.geometry,
                x_green_branch=batch.x_green_branch,
                y_green_branch=batch.y_green_branch,
            )
            if key is not None:
                self.audit.validated_sidecars.add(key)
            if self.audit.shared_cache is None:
                self.audit.shared_cache = own_cache
        cache = self.audit.shared_cache
        assert cache is not None
        self.context = cache.get_or_build(
            green_model=self.green,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        self.projection_by_k = {
            k: replace(
                self.projection,
                symmetric_tangent_green_response=replace(
                    self.tangent,
                    subspace_dimension=k,
                    max_subspace_dimension=max(k, self.tangent.max_subspace_dimension),
                ),
            )
            for k in range(self.request.baseline_k, self.request.max_k + 1)
        }
        self.context_by_k = {
            k: replace(self.context, subspace_dimension=k)
            for k in range(self.request.baseline_k, self.request.max_k + 1)
        }
        self.verification["context"] = cache.telemetry() | {
            "reused_across_runs": cache is not own_cache,
            "this_sidecar_validated": key is not None,
            "identity": cache.identity.as_dict()
            if cache.identity is not None
            else None,
        }
        training = replace(
            self.spec.configs.coupling_training, tangent_context_checkpoint=persistence
        )
        self.evaluator = ComplexCouplingEvaluator(
            model=self.model,
            green_model=self.green,
            config=training,
            device=self.device,
            work_dir=self.audit.outdir / "baseline_logs" / self.spec.run_id,
            tangent_context_path=sidecar,
        )
        self.evaluator._tangent_context_cache = cache

    def _prepare(self, batch: ComplexCouplingBatch) -> PreparedTangentBatch:
        return prepare_tangent_audit_batch(
            model=self.model, context=self.context, batch=batch
        )

    def _krylov(
        self, prepared: PreparedTangentBatch, k: int
    ) -> KrylovSubspaceStepResult:
        return matrix_free_krylov_subspace_step(
            context=self.context,
            mismatch=prepared.mismatch,
            gradient=prepared.gradient,
            max_dimension=k,
            relative_eps=self.tangent.line_search_relative_eps,
            monotonicity_relative_tol=1.0e-10,
        )

    def _candidate(
        self,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
        result: KrylovSubspaceStepResult,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        delta = result.deltas[k - 1]
        physical = torch.stack(
            (
                prepared.symmetric_physical[:, 0] + delta,
                prepared.symmetric_physical[:, 1] - delta,
            ),
            dim=1,
        )
        solution = self.context.response_operator.forward_pair(physical)
        cross = self.cross_axis.reconstruct(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            projected_physical=physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        return physical, solution, cross

    def _rows(
        self,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
        result: KrylovSubspaceStepResult,
        k: int,
    ) -> list[CsvRow]:
        physical, solution, cross = self._candidate(batch, prepared, result, k)
        for value in (physical, solution, cross.u_pred_valid):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{self.spec.run_id} K={k}: non-finite prediction.")
        balance = (physical.sum(dim=1) - batch.rhs_valid).abs().amax(dim=1)
        scale = physical.abs().sum(dim=1).amax(dim=1) + batch.rhs_valid.abs().amax(
            dim=1
        )
        if torch.any(
            balance > 64 * torch.finfo(torch.float64).eps * scale.clamp_min(1.0)
        ):
            raise RuntimeError(f"{self.spec.run_id} K={k}: physical balance failed.")
        self.verification["balance_max_abs"] = max(
            self.verification["balance_max_abs"], float(balance.max())
        )
        energy = canonical_complex_energy_loss(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            a_valid=batch.a_valid,
            geometry=batch.geometry,
            boundary_context=self.boundary,
        )
        optimized = (
            energy.bulk_per_sample
            + self.canonical.boundary_weight * energy.boundary_per_sample
        )
        rows: list[CsvRow] = []
        for i, sample_id in enumerate(batch.sample_indices.tolist()):
            target = batch.sol_valid[i : i + 1]
            pair_norm = float(prepared.symmetric_physical[i].norm())
            correction_norm = float(result.deltas[k - 1, i].norm()) * math.sqrt(2.0)
            row: CsvRow = dict(
                run_id=self.spec.run_id,
                seed=self.spec.seed,
                training_k=self.request.baseline_k,
                evaluation_k=k,
                baseline_k=self.request.baseline_k,
                sample_id=sample_id,
                file_stem=batch.file_stems[i],
                response_cost=float(result.costs[k - 1, i]),
                response_cost_previous=float(result.costs[k - 2, i]),
                rel_sol=float(relative_l2_valid(cross.u_pred_valid[i : i + 1], target)),
                rel_sol_equal_mean=float(
                    relative_l2_valid(cross.u_equal_mean_valid[i : i + 1], target)
                ),
                rel_u_phi=float(relative_l2_valid(solution[i : i + 1, 0], target)),
                rel_u_psi=float(relative_l2_valid(solution[i : i + 1, 1], target)),
                rel_flux=float(
                    relative_l2_valid(physical[i : i + 1], batch.flux_valid[i : i + 1])
                ),
                canonical_bulk_energy=float(energy.bulk_per_sample[i]),
                canonical_boundary_energy=float(energy.boundary_per_sample[i]),
                loss_energy_optimized=float(optimized[i]),
                correction_pair_norm=correction_norm,
                correction_ratio=correction_norm
                / max(pair_norm, torch.finfo(torch.float64).eps),
                direction_index=k - 1,
                coefficient=float(result.coefficients[k - 1, i]),
                direction_active=int(result.direction_active[k - 1, i]),
                effective_dimension=int(result.direction_active[:k, i].sum()),
                balance_max_abs=float(balance[i]),
                response_orthogonality_max=float(
                    result.response_orthogonality_max[k - 1, i]
                ),
            )
            row.update(self.audit.reach[k])
            if any(
                not math.isfinite(value)
                for value in row.values()
                if isinstance(value, float)
            ):
                raise RuntimeError("Non-finite metric.")
            rows.append(row)
        return rows

    def _compare(self, actual: float, expected: float, label: str) -> None:
        if not (math.isfinite(actual) and math.isfinite(expected)) or not math.isclose(
            actual, expected, rel_tol=1e-8, abs_tol=1e-14
        ):
            raise RuntimeError(
                f"Baseline mismatch {self.spec.run_id} {label}: {actual:.17g} != {expected:.17g}"
            )
        self.verification["baseline_max_abs"] = max(
            self.verification["baseline_max_abs"], abs(actual - expected)
        )

    @torch.no_grad()
    def evaluate(self) -> None:
        seen: set[tuple[int, str]] = set()
        prepared_batches: list[tuple[PreparedTangentBatch, list[CsvRow]]] = []
        for cpu_batch in self.loader:
            batch = cpu_batch.to(self.device)
            if not bool(batch.has_solution.all() and batch.has_flux.all()):
                raise ValueError(
                    "Every test sample requires sol, phi and psi references."
                )
            if self.evaluator is None:
                self._initialize_context(batch)
            prepared = self._prepare(batch)
            baseline = self._krylov(prepared, self.request.baseline_k)
            base_rows = self._rows(batch, prepared, baseline, self.request.baseline_k)
            assert self.evaluator is not None
            production = self.evaluator.predict_batch(batch)
            physical, solution, cross = self._candidate(
                batch, prepared, baseline, self.request.baseline_k
            )
            for actual, expected in (
                (physical, production.projection.projected_physical),
                (solution[:, 0], production.reconstruction.u_phi_valid),
                (solution[:, 1], production.reconstruction.u_psi_valid),
                (cross.u_pred_valid, production.cross_axis_reconstruction.u_pred_valid),
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
            for i, row in enumerate(base_rows):
                key = (row["sample_id"], row["file_stem"])
                if key in seen or key not in self.spec.artifacts:
                    raise ValueError(
                        f"Baseline artifact sample identity mismatch: {key}"
                    )
                seen.add(key)
                stored = self.spec.artifacts[key]
                prod = self.evaluator._sample_metric_row(production, i)
                mapping = {
                    "rel_sol": "rel_sol",
                    "rel_flux": "rel_flux",
                    "loss_energy_optimized": "loss_energy_optimized",
                    "canonical_bulk_energy": "loss_energy_bulk",
                    "canonical_boundary_energy": "loss_energy_boundary",
                    "response_cost": f"tangent_response_cost_k{self.request.baseline_k}",
                }
                if "rel_sol_equal_mean" in prod:
                    mapping["rel_sol_equal_mean"] = "rel_sol_equal_mean"
                for name, original in mapping.items():
                    self._compare(
                        row[name],
                        float(prod[original]),
                        f"sample={key} production {name}",
                    )
                    self._compare(
                        row[name],
                        float(stored[original]),
                        f"sample={key} artifact {name}",
                    )
                self.verification["baseline_samples"] += 1
            del production, physical, solution, cross, baseline
            prepared_batches.append(
                (
                    PreparedTangentBatch(
                        raw_physical=prepared.raw_physical.cpu(),
                        symmetric_physical=prepared.symmetric_physical.cpu(),
                        mismatch=prepared.mismatch.cpu(),
                        gradient=prepared.gradient.cpu(),
                    ),
                    base_rows,
                )
            )
        if seen != set(self.spec.artifacts):
            raise ValueError(
                "Test dataset and best-energy artifact sample sets differ."
            )
        self.audit.logger.info(
            "%s all %d baseline samples verified before K sweep",
            self.spec.run_id,
            len(seen),
        )
        evaluated = 0
        for cpu_batch, (cpu_prepared, base_rows) in zip(
            self.loader, prepared_batches, strict=True
        ):
            batch = cpu_batch.to(self.device)
            prepared = PreparedTangentBatch(
                raw_physical=cpu_prepared.raw_physical.to(self.device),
                symmetric_physical=cpu_prepared.symmetric_physical.to(self.device),
                mismatch=cpu_prepared.mismatch.to(self.device),
                gradient=cpu_prepared.gradient.to(self.device),
            )
            result = self._krylov(prepared, self.request.max_k)
            initial_cost = self.context.point_mass * prepared.mismatch.square().sum(
                dim=1
            )
            previous = torch.cat((initial_cost.unsqueeze(0), result.costs[:-1]), dim=0)
            tolerance = (
                1e-10 * torch.maximum(previous, initial_cost.unsqueeze(0))
                + torch.finfo(torch.float64).tiny
            )
            if torch.any(result.costs > previous + tolerance):
                raise RuntimeError(
                    "Nested response cost increased beyond production-scaled tolerance."
                )
            for k in range(self.request.baseline_k, self.request.max_k + 1):
                rows = self._rows(batch, prepared, result, k)
                if k == self.request.baseline_k:
                    for row, base in zip(rows, base_rows, strict=True):
                        for metric_key in (
                            "response_cost",
                            "rel_sol",
                            "rel_flux",
                            "loss_energy_optimized",
                        ):
                            self._compare(
                                row[metric_key],
                                base[metric_key],
                                f"prefix {metric_key}",
                            )
                    # Use the verified prefix as its own exact reporting baseline.
                    base_rows = [row.copy() for row in rows]
                for row, base in zip(rows, base_rows, strict=True):
                    add_paired_metrics(row, base)
                self.audit.rows.extend(rows)
            evaluated += len(base_rows)
            self.audit.logger.info(
                "%s evaluated %d/%d samples",
                self.spec.run_id,
                evaluated,
                len(self.dataset),
            )
        self.verification.update(
            response_nonincrease=True,
            finite=True,
            production_baseline_verified=True,
            artifact_baseline_verified=True,
        )

    def _prediction_forward(self, batch: ComplexCouplingBatch, k: int) -> object:
        raw, _ = self.model.forward_with_fusion_diagnostics(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
            rhs_phys=batch.rhs_valid,
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

    @torch.no_grad()
    def benchmark(self) -> None:
        clock = FrozenTangentBenchmark(self.device)
        for k in range(self.request.baseline_k, self.request.max_k + 1):
            for scope in ("tangent_only", "prediction_forward"):
                for repeat in range(
                    -self.request.warmup_repeats, self.request.timing_repeats
                ):
                    elapsed = 0.0
                    starts: list[float] = []
                    peaks: list[float] = []
                    for cpu_batch in self.loader:
                        batch = cpu_batch.to(self.device)
                        call: Callable[[], object]
                        if scope == "tangent_only":
                            prepared = self._prepare(batch)
                            call = partial(self._krylov, prepared, k)
                        else:
                            call = partial(self._prediction_forward, batch, k)
                        if repeat < 0:
                            result = call()
                            clock.synchronize()
                            del result
                        else:
                            measured = clock.measure(call)
                            elapsed += measured["seconds"]
                            if measured["start_allocated_mib"] is not None:
                                starts.append(measured["start_allocated_mib"])
                                peaks.append(measured["peak_allocated_mib"])
                        del call, batch
                        if scope == "tangent_only":
                            del prepared
                    if repeat >= 0:
                        self.audit.timings.append(
                            dict(
                                run_id=self.spec.run_id,
                                seed=self.spec.seed,
                                evaluation_k=k,
                                scope=scope,
                                repeat=repeat,
                                sample_count=len(self.dataset),
                                seconds=elapsed,
                                seconds_per_sample=elapsed / len(self.dataset),
                                start_allocated_mib=max(starts) if starts else None,
                                peak_allocated_mib=max(peaks) if peaks else None,
                                memory_status="cuda_allocated"
                                if peaks
                                else "cpu_unmeasured",
                            )
                        )
                self.audit.logger.info(
                    "%s benchmark K=%d %s complete", self.spec.run_id, k, scope
                )

    def close(self) -> None:
        if self.evaluator is not None:
            for handler in list(self.evaluator.logger.handlers):
                self.evaluator.logger.removeHandler(handler)
                handler.close()


def run_frozen_tangent_csv(request: FrozenTangentCsvRequest) -> dict[str, Any]:
    return FrozenTangentCsvAudit(request).run()
