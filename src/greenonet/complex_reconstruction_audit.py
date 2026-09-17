"""Read-only Annulus reconstruction audit using the production evaluator."""

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

Row = dict[str, Any]
BEST = "complex_coupling_model_best_energy.safetensors"
FIELDS = ("u_phi", "u_psi", "equal", "weak")


@dataclass(frozen=True)
class ReconstructionAuditRequest:
    run_dirs: tuple[Path, ...]
    outdir: Path
    device: str
    batch_size: int = 10
    num_threads: int = 4

    def __post_init__(self) -> None:
        if not self.run_dirs or self.device not in {"cpu", "cuda:1"}:
            raise ValueError(
                "Explicit cpu or cuda:1 and at least one run are required."
            )
        if self.batch_size < 1 or self.num_threads < 1:
            raise ValueError("Batch size and thread count must be positive.")


def region_masks(xy: np.ndarray, width: int) -> dict[str, np.ndarray]:
    horizontal = np.abs(np.abs(xy[:, 1]) - 0.2) <= width / 128
    vertical = np.abs(np.abs(xy[:, 0]) - 0.2) <= width / 128
    return dict(
        transition=horizontal | vertical,
        outside=~(horizontal | vertical),
        horizontal_only=horizontal & ~vertical,
        vertical_only=vertical & ~horizontal,
        overlap=horizontal & vertical,
    )


def sample_metrics(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    weight: np.ndarray,
    xy: np.ndarray,
) -> tuple[Row, list[Row], list[Row]]:
    if any(
        not np.isfinite(x).all() for x in [target, weight, xy, *predictions.values()]
    ):
        raise ValueError("Non-finite prediction or input.")
    errors = {name: predictions[name] - target for name in FIELDS}
    global_row: Row = {
        name: float(np.linalg.norm(e) / max(np.linalg.norm(target), 1e-12))
        for name, e in errors.items()
    }
    global_row["absolute_change"] = global_row["weak"] - global_row["equal"]
    global_row["relative_reduction"] = (
        1 - global_row["weak"] / global_row["equal"]
        if global_row["equal"] > 1e-12
        else None
    )
    global_row["improved"] = int(global_row["weak"] < global_row["equal"])
    regions, weights = [], []
    eps = np.finfo(target.dtype).eps
    error_scale = max(
        float(np.max(errors["u_phi"] ** 2 + errors["u_psi"] ** 2)),
        np.finfo(target.dtype).tiny,
    )
    advantage = errors["u_psi"] ** 2 - errors["u_phi"] ** 2
    tied = (np.abs(weight - 0.5) <= 64 * eps) | (
        np.abs(advantage) <= 64 * eps * error_scale
    )
    aligned = (weight - 0.5) * advantage > 0
    for width in (1, 2, 4):
        for name, mask in region_masks(xy, width).items():
            count = int(mask.sum())
            row: Row = dict(width_h=width, region=name, count=count)
            row.update(
                {
                    key: float(np.sqrt(np.mean(e[mask] ** 2))) if count else None
                    for key, e in errors.items()
                }
            )
            row["absolute_change"] = row["weak"] - row["equal"] if count else None
            row["relative_reduction"] = (
                1 - row["weak"] / row["equal"]
                if count and row["equal"] > 1e-12
                else None
            )
            row["improved"] = int(row["weak"] < row["equal"]) if count else None
            regions.append(row)
            valid = mask & ~tied
            nvalid = int(valid.sum())
            wr: Row = dict(
                width_h=width,
                region=name,
                count=count,
                valid_count=nvalid,
                tie_count=int((mask & tied).sum()),
                alignment_fraction=float(aligned[valid].mean()) if nvalid else None,
                opposite_error_fraction=float(
                    (errors["u_phi"][mask] * errors["u_psi"][mask] < 0).mean()
                )
                if count
                else None,
            )
            for label, w in [("phi", weight), ("psi", 1 - weight)]:
                for stat in ("mean", "median", "p05", "p95"):
                    wr[f"weight_{label}_{stat}"] = _stat(w[mask], stat)
                wr[f"weight_{label}_above_half_fraction"] = (
                    float((w[mask] > 0.5 + 64 * eps).mean()) if count else None
                )
            weights.append(wr)
    return global_row, regions, weights


def _stat(values: np.ndarray, name: str) -> float | None:
    if not len(values):
        return None
    if name == "mean":
        return float(np.mean(values))
    if name == "max":
        return float(np.max(values))
    return float(np.quantile(values, {"median": 0.5, "p05": 0.05, "p95": 0.95}[name]))


def summarize(
    rows: list[Row], metrics: list[str], groups: tuple[str, ...] = ()
) -> tuple[list[Row], list[Row]]:
    seeds, aggregate = [], []
    combinations = sorted({tuple(r[k] for k in groups) for r in rows})
    for combination in combinations:
        subset = [r for r in rows if tuple(r[k] for k in groups) == combination]
        for metric in metrics:
            means = []
            for seed in sorted({r["seed"] for r in subset}):
                values = np.array(
                    [
                        r[metric]
                        for r in subset
                        if r["seed"] == seed and r[metric] is not None
                    ],
                    dtype=float,
                )
                row = dict(zip(groups, combination, strict=True)) | dict(
                    seed=seed, metric=metric, n=len(values)
                )
                row.update(
                    {k: _stat(values, k) for k in ("mean", "median", "p95", "max")}
                )
                row["paired_win_count"] = (
                    int(values.sum()) if metric == "improved" else None
                )
                seeds.append(row)
                if len(values):
                    means.append(float(values.mean()))
            aggregate.append(
                dict(zip(groups, combination, strict=True))
                | dict(
                    metric=metric,
                    n_seeds=len(means),
                    mean=float(np.mean(means)) if means else None,
                    seed_sd=float(np.std(means, ddof=1)) if len(means) > 1 else None,
                )
            )
    return seeds, aggregate


def _read_csv(path: Path) -> list[Row]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReconstructionAudit:
    def __init__(self, request: ReconstructionAuditRequest) -> None:
        self.request = request
        self.out = request.outdir.resolve()
        self.hashes: dict[str, str] = {}
        self.global_rows: list[Row] = []
        self.region_rows: list[Row] = []
        self.weight_rows: list[Row] = []
        self.verification: Row = dict(status="running", runs=[])

    def _track(self, path: Path) -> None:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    self._track(child)
        else:
            self.hashes[str(path.resolve())] = _sha(path)

    def _json(self, name: str, value: Any) -> None:
        (self.out / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    def _preflight(self) -> None:
        common = None
        ids = None
        seeds = set()
        for run in self.request.run_dirs:
            raw = json.loads((run / "config_used.json").read_text())
            configs = load_coupling_artifact_configs(run / "config_used.json")
            summary = json.loads(
                (run / "artifacts_best_energy/summary.json").read_text()
            )
            if (
                summary["checkpoint_selector"] != "best_energy"
                or Path(summary["coupling_checkpoint"]).name != BEST
            ):
                raise ValueError("Baseline is not best-energy.")
            keys = [
                (int(r["sample_id"]), r["file_stem"])
                for r in _read_csv(
                    run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
                )
            ]
            if (
                len(keys) != 100
                or len(set(keys)) != 100
                or (ids is not None and ids != keys)
            ):
                raise ValueError("Expected the same 100 unique sample identities.")
            ids = keys
            training = {
                k: v for k, v in raw["coupling_training"].items() if k != "seed"
            }
            contract = [
                raw["dataset"],
                raw["coupling_model"],
                raw["pipeline"],
                training,
            ]
            if common is not None and contract != common:
                raise ValueError("Run configurations differ beyond training seed.")
            common = contract
            seed = configs.coupling_training.seed
            if seed is None or seed in seeds:
                raise ValueError("Distinct explicit seeds required.")
            seeds.add(seed)
            projection = summary["balance_projection"]
            if (
                projection["mode"] != "symmetric_tangent_green_response"
                or projection["symmetric_tangent_green_response"]["subspace_dimension"]
                != 4
            ):
                raise ValueError("Expected K4 tangent projection.")
            if (
                summary["cross_axis_reconstruction"]["affects_training_objective"]
                or not summary["cross_axis_reconstruction"]["enabled"]
            ):
                raise ValueError("Expected reference-free weak reconstruction enabled.")
            geometry = Path(raw["dataset"]["geometry_path"])
            with np.load(geometry) as data:
                for key, value in [
                    ("inner_radius", 0.2),
                    ("outer_radius", 0.5),
                    ("hx", 1 / 128),
                    ("hy", 1 / 128),
                ]:
                    if not np.isclose(data[key].item(), value, rtol=0, atol=1e-14):
                        raise ValueError(f"Unexpected geometry {key}")
                if str(data["domain_type"].item()) != "annulus" or not np.array_equal(
                    data["center"], np.zeros(2)
                ):
                    raise ValueError("Expected centered Annulus geometry.")
            for path in [
                run / "config_used.json",
                run / BEST,
                run / "artifacts_best_energy",
                geometry,
                Path(raw["pipeline"]["green_pretrained_path"]),
                Path(raw["dataset"]["coefficient_functions_path"]),
                Path(raw["dataset"]["test_path"]),
            ]:
                self._track(path)
            for suffix in ["safetensors", "json"]:
                path = run / f"tangent_response_context.{suffix}"
                if path.exists():
                    self._track(path)
        self._json(
            "provenance.json",
            dict(
                device=self.request.device,
                dtype="float64",
                batch_size=self.request.batch_size,
                num_threads=self.request.num_threads,
                input_hashes=self.hashes,
                torch_version=torch.__version__,
                python_version=platform.python_version(),
                inference_mode="eval_no_grad_eager",
                main_band_width_h=2,
                sensitivity_widths_h=[1, 4],
                reference_used_only_for_metrics=True,
                baseline_rtol=1e-8,
                baseline_atol=1e-10,
                tie_policy="64*dtype_eps, error difference relative to sample max sum directional error squared",
            ),
        )

    def run(self) -> None:
        if self.out.exists() and any(self.out.iterdir()):
            raise ValueError("Output must be empty.")
        if any(self.out.is_relative_to(p.resolve()) for p in self.request.run_dirs):
            raise ValueError("Output cannot be inside a source run.")
        self.out.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"annulus_audit.{self.out}")
        console = RichHandler(
            show_path=True, omit_repeated_times=False, rich_tracebacks=True
        )
        file = logging.FileHandler(self.out / "audit.log")
        for h in [console, file]:
            h.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        self.logger.handlers = [console, file]
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        try:
            self._preflight()
            torch.set_num_threads(self.request.num_threads)
            device = torch.device(self.request.device)
            if device.type == "cuda":
                if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
                    raise RuntimeError("cuda:1 unavailable; no fallback.")
                torch.cuda.set_device(device)
                self.verification["device_name"] = torch.cuda.get_device_name(device)
            for run in self.request.run_dirs:
                self.logger.info("Evaluating %s", run)
                with torch.no_grad():
                    self._run_one(run, device)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for path, digest in self.hashes.items():
                if _sha(Path(path)) != digest:
                    raise RuntimeError(f"Source modified: {path}")
            self._outputs()
            self.verification.update(
                status="complete",
                sources_unchanged=True,
                sample_evaluations=len(self.global_rows),
            )
        except Exception as exc:
            self.verification.update(status="failed", error=str(exc))
            self.logger.exception("Audit stopped; no fallback or tolerance change")
            raise
        finally:
            self._json("verification.json", self.verification)
            for h in self.logger.handlers:
                h.close()

    def _run_one(self, run: Path, device: torch.device) -> None:
        configs = load_coupling_artifact_configs(run / "config_used.json")
        raw = configs.raw
        seed = configs.coupling_training.seed
        geometry = load_complex_geometry(
            Path(raw["dataset"]["geometry_path"]), dtype=torch.float64
        )
        dataset = ComplexCouplingDataset(
            Path(raw["dataset"]["test_path"]),
            geometry,
            load_coefficient_functions(
                Path(raw["dataset"]["coefficient_functions_path"])
            ),
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=torch.float64,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        if len(dataset) != 100:
            raise ValueError("Test dataset must contain 100 samples.")
        loader = ComplexCouplingArtifactExporter(
            CouplingArtifactRequest(
                config=run / "config_used.json",
                coupling_checkpoint=run / BEST,
                green_checkpoint=Path(raw["pipeline"]["green_pretrained_path"]),
                outdir=self.out,
                device=str(device),
            ),
            logger=self.logger,
        )
        model = loader._load_complex_model(configs, device)
        green = loader._load_green_model(configs, device)
        for network in [model, green]:
            network.eval()
            for parameter in network.parameters():
                if parameter.dtype != torch.float64:
                    raise ValueError("Expected float64 checkpoint.")
                parameter.requires_grad_(False)
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
            work_dir=self.out / f"seed{seed}_logs",
            tangent_context_path=run / "tangent_response_context.safetensors",
        )
        baseline = {
            r["file_stem"]: r
            for r in _read_csv(
                run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
            )
        }
        verification: Row = dict(
            seed=seed,
            samples=0,
            selected_checked=0,
            baseline_max_abs=0.0,
            first_batch_passed=False,
        )
        self.verification["runs"].append(verification)
        with np.load(
            run / "artifacts_best_energy/data/selected_raw_arrays.npz"
        ) as saved:
            selected = {
                k[:-4].split("_", 2)[2]: k[:-4]
                for k in saved.files
                if k.endswith("_sol")
            }
            seen = set()
            for cpu_batch in DataLoader(
                dataset,
                batch_size=self.request.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=complex_coupling_collate_fn,
            ):
                batch = cpu_batch.to(device)
                if not bool(batch.has_solution.all() and batch.has_flux.all()):
                    raise ValueError("Missing test reference.")
                result = evaluator.predict_batch(batch)
                cross = result.cross_axis_reconstruction
                if cross.reliability is None:
                    raise ValueError("Missing reliability diagnostics.")
                arrays = {
                    "u_phi": result.reconstruction.u_phi_valid,
                    "u_psi": result.reconstruction.u_psi_valid,
                    "equal": cross.u_equal_mean_valid,
                    "weak": cross.u_pred_valid,
                }
                for i, stem in enumerate(batch.file_stems):
                    if stem in seen or stem not in baseline:
                        raise ValueError("Sample identity mismatch.")
                    seen.add(stem)
                    ident = dict(
                        seed=seed,
                        sample_id=int(batch.sample_indices[i]),
                        file_stem=stem,
                    )
                    if ident["sample_id"] != int(baseline[stem]["sample_id"]):
                        raise ValueError("Sample index mismatch.")
                    target = batch.sol_valid[i].cpu().numpy()
                    predictions = {k: v[i].cpu().numpy() for k, v in arrays.items()}
                    xy = batch.geometry.coords_valid.cpu().numpy()
                    weight = cross.reliability.w_phi[i].cpu().numpy()
                    g, regions, weights = sample_metrics(
                        target, predictions, weight, xy
                    )
                    for name, key in [
                        ("weak", "rel_sol"),
                        ("equal", "rel_sol_equal_mean"),
                    ]:
                        expected = float(baseline[stem][key])
                        np.testing.assert_allclose(
                            g[name],
                            expected,
                            rtol=1e-8,
                            atol=1e-10,
                            err_msg=f"seed{seed}/{stem}/{name}",
                        )
                        verification["baseline_max_abs"] = max(
                            verification["baseline_max_abs"], abs(g[name] - expected)
                        )
                    physical = result.projection.projected_physical[i].cpu().numpy()
                    reference = batch.flux_valid[i].cpu().numpy()
                    g["rel_flux"] = float(
                        (
                            np.linalg.norm(physical - reference, axis=-1)
                            / np.maximum(np.linalg.norm(reference, axis=-1), 1e-12)
                        ).mean()
                    )
                    np.testing.assert_allclose(
                        g["rel_flux"],
                        float(baseline[stem]["rel_flux"]),
                        rtol=1e-8,
                        atol=1e-10,
                    )
                    if stem in selected:
                        prefix = selected[stem]
                        old = {
                            k: saved[prefix + "_" + name]
                            for k, name in [
                                ("u_phi", "u_phi"),
                                ("u_psi", "u_psi"),
                                ("equal", "u_equal_mean"),
                                ("weak", "u_pred"),
                            ]
                        }
                        sg, sr, _ = sample_metrics(
                            saved[prefix + "_sol"],
                            old,
                            saved[prefix + "_weak_reliability_weight_phi"],
                            saved[prefix + "_coords_valid"],
                        )
                        for k in FIELDS:
                            np.testing.assert_allclose(
                                g[k], sg[k], rtol=1e-8, atol=1e-10
                            )
                        for new, previous in zip(regions, sr, strict=True):
                            for k in FIELDS:
                                if previous[k] is not None:
                                    np.testing.assert_allclose(
                                        new[k], previous[k], rtol=1e-8, atol=1e-10
                                    )
                        verification["selected_checked"] += 1
                    self.global_rows.append(ident | g)
                    self.region_rows.extend(ident | r for r in regions)
                    self.weight_rows.extend(ident | r for r in weights)
                    verification["samples"] += 1
                if not verification["first_batch_passed"]:
                    verification["first_batch_passed"] = True
                    self.logger.info(
                        "seed%s first batch baseline passed; continuing full test", seed
                    )
            if seen != set(baseline) or verification["selected_checked"] != len(
                selected
            ):
                raise ValueError("Incomplete coverage.")
        verification["context"] = (
            evaluator.symmetric_tangent_green_response_context_telemetry
        )
        verification["status"] = "complete"
        self.logger.info(
            "seed%s: 100/100 verified; selected=%s",
            seed,
            verification["selected_checked"],
        )

    def _outputs(self) -> None:
        for name, rows in [
            ("per_sample_global.csv", self.global_rows),
            ("per_sample_regions.csv", self.region_rows),
            ("per_sample_weights.csv", self.weight_rows),
        ]:
            _write_csv(self.out / name, rows)
        seeds: list[Row] = []
        aggregates: list[Row] = []
        for scope, rows, metrics, groups in [
            (
                "global",
                self.global_rows,
                [
                    *FIELDS,
                    "rel_flux",
                    "absolute_change",
                    "relative_reduction",
                    "improved",
                ],
                (),
            ),
            (
                "regions",
                self.region_rows,
                [*FIELDS, "absolute_change", "relative_reduction", "improved"],
                ("width_h", "region"),
            ),
            (
                "weights",
                self.weight_rows,
                [
                    k
                    for k in self.weight_rows[0]
                    if k not in {"seed", "sample_id", "file_stem", "width_h", "region"}
                ],
                ("width_h", "region"),
            ),
        ]:
            s, a = summarize(rows, metrics, groups)
            seeds.extend(dict(scope=scope, width_h="", region="") | r for r in s)
            aggregates.extend(dict(scope=scope, width_h="", region="") | r for r in a)
        _write_csv(self.out / "seed_summary.csv", seeds)
        _write_csv(self.out / "aggregate_summary.csv", aggregates)
        self._report(seeds, aggregates)

    def _report(self, seeds: list[Row], aggregate: list[Row]) -> None:
        def value(
            scope: str, metric: str, region: str = "", width: int | str = ""
        ) -> float:
            return float(
                next(
                    r["mean"]
                    for r in aggregate
                    if r["scope"] == scope
                    and r["metric"] == metric
                    and r["region"] == region
                    and r["width_h"] == width
                )
            )

        lines = [
            "# Annulus Full-Test Directional / Transition Audit",
            "",
            "## Evidence and scope",
            "",
            f"Evaluated {len(self.global_rows)} sample/model pairs: the same 100 test sources across four training seeds. These are NOT 400 independent sources.",
            "Frozen best-energy checkpoints; production K4 separable tangent projection and original weak reliability settings. No training, tuning, benchmark, or paper-section writing.",
            "",
            "All values below come from the full test, not selected example fields. Relative errors use the existing valid-point L2 norm per sample before averaging. SD denotes sample SD of seed means (ddof=1), not a confidence interval.",
            "",
            "## Global relative errors",
            "",
            "| Prediction | Mean (%) | Seed SD (percentage points) |",
            "|---|---:|---:|",
        ]
        for name in (*FIELDS, "rel_flux"):
            row = next(
                r for r in aggregate if r["scope"] == "global" and r["metric"] == name
            )
            lines.append(
                f"| {name} | {100 * row['mean']:.6f} | {100 * row['seed_sd']:.6f} |"
            )
        reduction = 100 * (1 - value("global", "weak") / value("global", "equal"))
        wins = sum(r["improved"] for r in self.global_rows)
        lines += [
            "",
            f"Weak versus equal: ratio-of-means reduction {reduction:.4f}%; {wins}/{len(self.global_rows)} paired wins. The mean of samplewise reductions is a different statistic, explicitly named relative_reduction in the CSV.",
            "rel_flux is (rel_phi + rel_psi)/2, not error in a*grad(u); blending does not change directional sources.",
            "",
            "## Region errors",
            "",
            "Transition Bx: abs(abs(y)-0.2) <= w*h; By: abs(abs(x)-0.2) <= w*h, h=1/128. Main width w=2; w=1,4 are predeclared sensitivity checks. Transition is the union (counted once). Horizontal-only, vertical-only, and overlap are disjoint. Outside is the complement.",
            "RMS_B = sqrt(mean_B((prediction-reference)^2)); this is not a trace-jump metric. Reduction below is 100*(1-mean RMS_weak/mean RMS_equal), not the mean sample reduction.",
            "",
            "| Width | Region | Points | Equal RMS | Weak RMS | Reduction (%) | Wins / pairs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for width in (1, 2, 4):
            for region in (
                "transition",
                "outside",
                "horizontal_only",
                "vertical_only",
                "overlap",
            ):
                rows = [
                    r
                    for r in self.region_rows
                    if r["width_h"] == width and r["region"] == region
                ]
                e, w = (
                    value("regions", "equal", region, width),
                    value("regions", "weak", region, width),
                )
                lines.append(
                    f"| {width}h | {region} | {rows[0]['count']} | {e:.7e} | {w:.7e} | {100 * (1 - w / e):.4f} | {sum(r['improved'] for r in rows)}/{len(rows)} |"
                )
        lines += [
            "",
            "## Weight and error alignment",
            "",
            "Horizontal-only tests transverse psi weight; vertical-only tests transverse phi weight. Neither the band nor the reference solution is supplied to the production weighting rule.",
            "Alignment: (w_phi-0.5)*((u_psi-u)^2-(u_phi-u)^2)>0. Weight ties use 64*machine_eps; squared-error ties use 64*machine_eps times the sample maximum directional squared-error sum. Alignment excludes ties and CSVs preserve valid/tie counts. Opposite-error fraction counts (u_phi-u)*(u_psi-u)<0.",
            "",
            "| Region (2h) | Mean w_phi | Mean w_psi | Alignment (%) | Opposite signs (%) |",
            "|---|---:|---:|---:|---:|",
        ]
        for region in ("horizontal_only", "vertical_only", "overlap", "outside"):
            metrics = [
                value("weights", m, region, 2)
                for m in (
                    "weight_phi_mean",
                    "weight_psi_mean",
                    "alignment_fraction",
                    "opposite_error_fraction",
                )
            ]
            lines.append(
                f"| {region} | {metrics[0]:.6f} | {metrics[1]:.6f} | {100 * metrics[2]:.3f} | {100 * metrics[3]:.3f} |"
            )
        lines += [
            "",
            "## Exceptions",
            "",
            "The sign-based alignment is not an error oracle. Opposite-sign directional errors can cancel, so choosing the smaller individual error is not equivalent to finding the best blend. Report actual blended errors independently.",
            "",
            "| Region (2h) | Worst seed | Worst source | Relative change in RMS (%) |",
            "|---|---:|---|---:|",
        ]
        for region in (
            "horizontal_only",
            "vertical_only",
            "overlap",
            "transition",
            "outside",
        ):
            rows = [
                r
                for r in self.region_rows
                if r["width_h"] == 2 and r["region"] == region
            ]
            worst = min(rows, key=lambda r: r["relative_reduction"])
            lines.append(
                f"| {region} | {worst['seed']} | {worst['file_stem']} | {-100 * worst['relative_reduction']:.4f} |"
            )
        lines += [
            "",
            "## Interpretation and limitations",
            "",
            "Use global improvement, transition concentration, and orientation alignment as separate empirical statements. Regional exceptions remain visible even if the union improves. No continuity guarantee, optimal smoothing claim, or cross-domain guarantee follows from these measurements.",
            "Checkpoint selection used validation energy, not reference errors. All four models use the same GP source seed, so training seeds measure initialization/batch-order variation, not independently generated datasets.",
            "No timing benchmark was performed; this report does not claim an inference-speed advantage. The paper preparation document is deliberately deferred.",
            "",
            "## Reproduction and verification",
            "",
            "Run cli/audit_annulus_reconstruction.py with explicit --run-dirs, --outdir, --device cuda:1, --batch-size 10, --num-threads 4 from the project root with PYTHONPATH=src. Output must be new/empty. No overwrite or CPU fallback is performed.",
            "verification.json records first-batch gating, all baseline checks (rtol=1e-8, atol=1e-10), selected raw directional and region comparisons, and context load/build/save counts. provenance.json records exact source paths and SHA256 hashes, settings and metric conventions. Sources are rehashed after evaluation.",
            "CSV files: per_sample_global.csv, per_sample_regions.csv, per_sample_weights.csv, seed_summary.csv, aggregate_summary.csv. Empty cells are NA; no empty-region or near-zero-denominator fake improvements are inserted.",
            "See audit.log and per-seed evaluator logs for execution details.",
            "",
        ]
        (self.out / "report.md").write_text("\n".join(lines))
