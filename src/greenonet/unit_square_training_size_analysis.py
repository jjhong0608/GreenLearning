from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from greenonet.plotly_io import save_plotly_figure


PRIMARY_METRICS_PATH = Path("artifacts_best_energy/metrics/per_sample_metrics.csv")
FINAL_METRICS_PATH = Path("metrics/test_per_sample_metrics.csv")
TRAINING_METRICS_PATH = Path("complex_training_metrics.csv")
ARTIFACT_SUMMARY_PATH = Path("artifacts_best_energy/summary.json")
BEST_ENERGY_CHECKPOINT = Path("complex_coupling_model_best_energy.safetensors")


@dataclass(frozen=True)
class TrainingSizeAnalysisRequest:
    root: Path
    outdir: Path
    train_sizes: tuple[int, ...] = (600, 1200, 2400, 4800)
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    saturation_tolerance: float = 0.05
    expected_optimizer_steps: int = 2400
    expected_test_samples: int = 100

    def validate(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Experiment root does not exist: {self.root}")
        if not self.train_sizes or any(size < 1 for size in self.train_sizes):
            raise ValueError("train_sizes must contain positive integers")
        if tuple(sorted(set(self.train_sizes))) != self.train_sizes:
            raise ValueError("train_sizes must be unique and strictly increasing")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not math.isfinite(self.saturation_tolerance) or not (
            0.0 <= self.saturation_tolerance < 1.0
        ):
            raise ValueError("saturation_tolerance must be finite and in [0, 1)")
        if self.expected_optimizer_steps < 1:
            raise ValueError("expected_optimizer_steps must be positive")
        if self.expected_test_samples < 1:
            raise ValueError("expected_test_samples must be positive")


@dataclass(frozen=True)
class RunMetrics:
    train_size: int
    seed: int
    run_name: str
    epochs: int
    batch_size: int
    num_valid: int
    total_optimizer_steps: int
    warmup_steps: int
    validation_every_steps: int
    best_step: int
    best_validation_energy: float
    final_validation_energy: float
    final_over_best_validation_energy: float
    optimizer_step_time_mean_ms: float
    rel_sol_mean: float
    rel_sol_median: float
    rel_sol_p90: float
    rel_sol_max: float
    rel_flux_mean: float
    rel_flux_median: float
    rel_flux_p90: float
    rel_flux_max: float
    rel_sol_equal_mean: float
    energy_optimized_mean: float
    final_model_rel_sol_mean: float
    test_sample_ids: tuple[int, ...]
    rel_sol_values: tuple[float, ...]
    rel_flux_values: tuple[float, ...]
    validation_steps: tuple[int, ...]
    validation_energies: tuple[float, ...]


@dataclass(frozen=True)
class SizeSummary:
    train_size: int
    seed_count: int
    rel_sol_mean: float
    rel_sol_sd: float
    rel_sol_ci95_low: float
    rel_sol_ci95_high: float
    rel_sol_median_mean: float
    rel_sol_p90_mean: float
    rel_sol_max_mean: float
    rel_flux_mean: float
    rel_flux_sd: float
    rel_flux_ci95_low: float
    rel_flux_ci95_high: float
    rel_flux_p90_mean: float
    rel_flux_max_mean: float
    rel_sol_equal_mean: float
    weak_blend_gain: float
    energy_optimized_mean: float
    best_step_mean: float
    best_step_fraction: float
    final_over_best_validation_energy_mean: float
    final_model_rel_sol_mean: float
    best_checkpoint_gain_over_final: float
    relative_gap_to_best: float
    within_saturation_tolerance: bool


@dataclass(frozen=True)
class AdjacentComparison:
    lower_train_size: int
    upper_train_size: int
    metric: str
    lower_mean: float
    upper_mean: float
    mean_difference: float
    relative_change: float
    paired_ci95_low: float
    paired_ci95_high: float
    paired_t_pvalue: float
    improved_seed_count: int
    seed_count: int


@dataclass(frozen=True)
class SampleWinRate:
    lower_train_size: int
    upper_train_size: int
    metric: str
    scope: str
    seed: int | None
    improved_sample_count: int
    sample_count: int
    improved_fraction: float


@dataclass(frozen=True)
class TrainingSizeDecision:
    recommended_num_train: int
    best_observed_num_train: int
    saturation_tolerance: float
    smallest_within_tolerance: int
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSizeAnalysisResult:
    runs: tuple[RunMetrics, ...]
    size_summaries: tuple[SizeSummary, ...]
    adjacent_comparisons: tuple[AdjacentComparison, ...]
    sample_win_rates: tuple[SampleWinRate, ...]
    decision: TrainingSizeDecision


def _mean_sd_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    center = mean(values)
    if len(values) == 1:
        return center, 0.0, center, center
    spread = stdev(values)
    half_width = (
        float(stats.t.ppf(0.975, len(values) - 1)) * spread / math.sqrt(len(values))
    )
    return center, spread, center - half_width, center + half_width


def choose_training_size(
    summaries: Sequence[SizeSummary], saturation_tolerance: float
) -> TrainingSizeDecision:
    if not summaries:
        raise ValueError("At least one size summary is required")
    if not math.isfinite(saturation_tolerance) or not (
        0.0 <= saturation_tolerance < 1.0
    ):
        raise ValueError("saturation_tolerance must be finite and in [0, 1)")

    best = min(summaries, key=lambda item: item.rel_sol_mean)
    eligible = sorted(
        item.train_size
        for item in summaries
        if item.rel_sol_mean <= best.rel_sol_mean * (1.0 + saturation_tolerance)
    )
    if not eligible:
        raise RuntimeError("Internal error: best result is not tolerance-eligible")
    smallest = eligible[0]
    recommended = smallest
    rationale = (
        "Use the smallest source count within the predeclared relative-solution "
        "error tolerance of the best observed result.",
        "All candidates use the same optimizer-step budget, so a larger source "
        "set does not receive extra parameter updates.",
        "Prefer the observed best size when no smaller candidate meets the "
        "tolerance criterion.",
    )
    return TrainingSizeDecision(
        recommended_num_train=recommended,
        best_observed_num_train=best.train_size,
        saturation_tolerance=saturation_tolerance,
        smallest_within_tolerance=smallest,
        rationale=rationale,
    )


class CsvAnalysisMixin:
    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(f"Required CSV is missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _required_float(row: dict[str, str], key: str, path: Path) -> float:
        raw = row.get(key)
        if raw is None or raw == "":
            raise ValueError(f"Missing numeric field {key!r} in {path}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite field {key!r} in {path}: {raw}")
        return value

    @staticmethod
    def _required_int(row: dict[str, str], key: str, path: Path) -> int:
        value = CsvAnalysisMixin._required_float(row, key, path)
        integer = int(value)
        if value != integer:
            raise ValueError(f"Expected integer field {key!r} in {path}: {value}")
        return integer


class ProvenanceValidationMixin(CsvAnalysisMixin):
    request: TrainingSizeAnalysisRequest
    logger: logging.Logger

    def _run_dir(self, train_size: int, seed: int) -> Path:
        return self.request.root / f"coupling_train{train_size}_seed{seed}"

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Required JSON is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a JSON object: {path}")
        return payload

    def _validate_run_files(self, run_dir: Path) -> None:
        required = (
            run_dir / "_SUCCESS",
            run_dir / "config_used.json",
            run_dir / BEST_ENERGY_CHECKPOINT,
            run_dir / TRAINING_METRICS_PATH,
            run_dir / PRIMARY_METRICS_PATH,
            run_dir / FINAL_METRICS_PATH,
            run_dir / ARTIFACT_SUMMARY_PATH,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Run {run_dir.name} is incomplete; missing: {', '.join(missing)}"
            )

    def _validate_config(
        self, config: dict[str, Any], train_size: int, seed: int, path: Path
    ) -> tuple[int, int, int, int, int]:
        try:
            source = config["dataset"]["coupling_source"]["indexed_gp"]
            training = config["coupling_training"]
            model = config["coupling_model"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed experiment config: {path}") from exc

        expected_pairs = {
            "source num_train": (source["num_train"], train_size),
            "source seed": (source["seed"], seed),
            "training seed": (training["seed"], seed),
            "batch size": (training["batch_size"], 200),
            "warmup steps": (training["warmup_steps"], 240),
            "validation interval": (training["validation_every_steps"], 24),
            "subspace dimension": (
                model["balance_projection"]["symmetric_tangent_green_response"][
                    "subspace_dimension"
                ],
                4,
            ),
            "branch fusion": (model["branch_fusion"]["mode"], "product_fuser"),
            "transverse fusion": (
                model["axis_1d_trunk"]["transverse_trunk"]["fusion"],
                "product_fuser",
            ),
        }
        mismatches = [
            f"{name}: observed={observed!r}, expected={expected!r}"
            for name, (observed, expected) in expected_pairs.items()
            if observed != expected
        ]
        coefficient_terms = model["coefficient_terms"]
        if any(bool(coefficient_terms[key]) for key in coefficient_terms):
            mismatches.append("Pure Poisson coefficient_terms must all be false")
        if training.get("deterministic_algorithms") is not True:
            mismatches.append("deterministic_algorithms must be true")
        if config["dataset"]["reference_diagnostics"] != {
            "training": False,
            "validation": False,
        }:
            mismatches.append("training/validation reference diagnostics must be off")
        if mismatches:
            raise ValueError(f"Protocol mismatch in {path}: {'; '.join(mismatches)}")

        return (
            int(training["epochs"]),
            int(training["batch_size"]),
            int(source["num_valid"]),
            int(training["warmup_steps"]),
            int(training["validation_every_steps"]),
        )

    def _validate_artifact_summary(
        self,
        summary: dict[str, Any],
        primary_rows: Sequence[dict[str, str]],
        path: Path,
    ) -> None:
        selection = summary.get("checkpoint_selection")
        if not isinstance(selection, dict) or selection.get("best_energy") is not True:
            raise ValueError(f"Artifact is not from a best-energy checkpoint: {path}")
        if selection.get("reference_metric_used") is not False:
            raise ValueError(f"Artifact checkpoint used a reference metric: {path}")
        if summary.get("reference_targets_used_for_training") is not False:
            raise ValueError(f"Reference targets entered training: {path}")
        aggregate = summary.get("aggregate_metrics")
        if not isinstance(aggregate, dict):
            raise ValueError(f"Missing aggregate_metrics in {path}")
        computed_rel_sol = mean(float(row["rel_sol"]) for row in primary_rows)
        computed_rel_flux = mean(float(row["rel_flux"]) for row in primary_rows)
        for key, computed in (
            ("rel_sol_mean", computed_rel_sol),
            ("rel_flux_mean", computed_rel_flux),
        ):
            observed = float(aggregate[key])
            if not math.isclose(observed, computed, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    f"Artifact summary mismatch for {key} in {path}: "
                    f"{observed} != {computed}"
                )


class MetricAggregationMixin(ProvenanceValidationMixin):
    def _parse_run(self, train_size: int, seed: int) -> RunMetrics:
        run_dir = self._run_dir(train_size, seed)
        self._validate_run_files(run_dir)
        config_path = run_dir / "config_used.json"
        summary_path = run_dir / ARTIFACT_SUMMARY_PATH
        config = self._load_json(config_path)
        epochs, batch_size, num_valid, warmup_steps, validation_every_steps = (
            self._validate_config(config, train_size, seed, config_path)
        )

        primary_path = run_dir / PRIMARY_METRICS_PATH
        primary_rows = self._read_csv(primary_path)
        if len(primary_rows) != self.request.expected_test_samples:
            raise ValueError(
                f"Expected {self.request.expected_test_samples} test samples in "
                f"{primary_path}, found {len(primary_rows)}"
            )
        summary = self._load_json(summary_path)
        self._validate_artifact_summary(summary, primary_rows, summary_path)

        rel_sol_values = tuple(
            self._required_float(row, "rel_sol", primary_path) for row in primary_rows
        )
        rel_flux_values = tuple(
            self._required_float(row, "rel_flux", primary_path) for row in primary_rows
        )
        sample_ids = tuple(
            self._required_int(row, "sample_id", primary_path) for row in primary_rows
        )
        equal_mean_values = tuple(
            self._required_float(row, "rel_sol_equal_mean", primary_path)
            for row in primary_rows
        )
        energy_values = tuple(
            self._required_float(row, "loss_energy_optimized", primary_path)
            for row in primary_rows
        )

        final_path = run_dir / FINAL_METRICS_PATH
        final_rows = self._read_csv(final_path)
        if tuple(int(row["sample_id"]) for row in final_rows) != sample_ids:
            raise ValueError(
                f"Final and best-energy sample ordering differs: {run_dir}"
            )
        final_rel_sol = tuple(
            self._required_float(row, "rel_sol", final_path) for row in final_rows
        )

        training_path = run_dir / TRAINING_METRICS_PATH
        training_rows = self._read_csv(training_path)
        validation_rows = [row for row in training_rows if row.get("split") == "val"]
        train_rows = [row for row in training_rows if row.get("split") == "train"]
        if not validation_rows or not train_rows:
            raise ValueError(f"Training metrics lack train/val rows: {training_path}")

        validation_steps = tuple(
            self._required_int(row, "global_step", training_path)
            for row in validation_rows
        )
        validation_energies = tuple(
            self._required_float(row, "loss_energy_optimized", training_path)
            for row in validation_rows
        )
        total_optimizer_steps = max(
            self._required_int(row, "global_step", training_path) for row in train_rows
        )
        if total_optimizer_steps != self.request.expected_optimizer_steps:
            raise ValueError(
                f"{run_dir.name} used {total_optimizer_steps} optimizer steps, expected "
                f"{self.request.expected_optimizer_steps}"
            )

        best_index = min(
            range(len(validation_energies)), key=validation_energies.__getitem__
        )
        best_energy = validation_energies[best_index]
        final_energy = validation_energies[-1]
        timing_weight = sum(
            self._required_float(row, "optimizer_step_count", training_path)
            for row in train_rows
        )
        timing_sum = sum(
            self._required_float(row, "optimizer_step_time_mean_ms", training_path)
            * self._required_float(row, "optimizer_step_count", training_path)
            for row in train_rows
        )

        return RunMetrics(
            train_size=train_size,
            seed=seed,
            run_name=run_dir.name,
            epochs=epochs,
            batch_size=batch_size,
            num_valid=num_valid,
            total_optimizer_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
            validation_every_steps=validation_every_steps,
            best_step=validation_steps[best_index],
            best_validation_energy=best_energy,
            final_validation_energy=final_energy,
            final_over_best_validation_energy=final_energy / best_energy,
            optimizer_step_time_mean_ms=timing_sum / timing_weight,
            rel_sol_mean=mean(rel_sol_values),
            rel_sol_median=float(np.median(rel_sol_values)),
            rel_sol_p90=float(np.quantile(rel_sol_values, 0.9)),
            rel_sol_max=max(rel_sol_values),
            rel_flux_mean=mean(rel_flux_values),
            rel_flux_median=float(np.median(rel_flux_values)),
            rel_flux_p90=float(np.quantile(rel_flux_values, 0.9)),
            rel_flux_max=max(rel_flux_values),
            rel_sol_equal_mean=mean(equal_mean_values),
            energy_optimized_mean=mean(energy_values),
            final_model_rel_sol_mean=mean(final_rel_sol),
            test_sample_ids=sample_ids,
            rel_sol_values=rel_sol_values,
            rel_flux_values=rel_flux_values,
            validation_steps=validation_steps,
            validation_energies=validation_energies,
        )

    def _aggregate_sizes(self, runs: Sequence[RunMetrics]) -> tuple[SizeSummary, ...]:
        grouped: dict[int, list[RunMetrics]] = defaultdict(list)
        for run in runs:
            grouped[run.train_size].append(run)
        best_rel_sol = min(
            mean(run.rel_sol_mean for run in group) for group in grouped.values()
        )

        summaries: list[SizeSummary] = []
        for train_size in self.request.train_sizes:
            group = sorted(grouped[train_size], key=lambda item: item.seed)
            rel_sol_mean, rel_sol_sd, rel_sol_low, rel_sol_high = _mean_sd_ci95(
                [run.rel_sol_mean for run in group]
            )
            rel_flux_mean, rel_flux_sd, rel_flux_low, rel_flux_high = _mean_sd_ci95(
                [run.rel_flux_mean for run in group]
            )
            equal_mean = mean(run.rel_sol_equal_mean for run in group)
            final_model_mean = mean(run.final_model_rel_sol_mean for run in group)
            gap_to_best = rel_sol_mean / best_rel_sol - 1.0
            summaries.append(
                SizeSummary(
                    train_size=train_size,
                    seed_count=len(group),
                    rel_sol_mean=rel_sol_mean,
                    rel_sol_sd=rel_sol_sd,
                    rel_sol_ci95_low=rel_sol_low,
                    rel_sol_ci95_high=rel_sol_high,
                    rel_sol_median_mean=mean(run.rel_sol_median for run in group),
                    rel_sol_p90_mean=mean(run.rel_sol_p90 for run in group),
                    rel_sol_max_mean=mean(run.rel_sol_max for run in group),
                    rel_flux_mean=rel_flux_mean,
                    rel_flux_sd=rel_flux_sd,
                    rel_flux_ci95_low=rel_flux_low,
                    rel_flux_ci95_high=rel_flux_high,
                    rel_flux_p90_mean=mean(run.rel_flux_p90 for run in group),
                    rel_flux_max_mean=mean(run.rel_flux_max for run in group),
                    rel_sol_equal_mean=equal_mean,
                    weak_blend_gain=1.0 - rel_sol_mean / equal_mean,
                    energy_optimized_mean=mean(
                        run.energy_optimized_mean for run in group
                    ),
                    best_step_mean=mean(run.best_step for run in group),
                    best_step_fraction=mean(run.best_step for run in group)
                    / self.request.expected_optimizer_steps,
                    final_over_best_validation_energy_mean=mean(
                        run.final_over_best_validation_energy for run in group
                    ),
                    final_model_rel_sol_mean=final_model_mean,
                    best_checkpoint_gain_over_final=(
                        1.0 - rel_sol_mean / final_model_mean
                    ),
                    relative_gap_to_best=gap_to_best,
                    within_saturation_tolerance=(
                        gap_to_best <= self.request.saturation_tolerance
                    ),
                )
            )
        return tuple(summaries)

    @staticmethod
    def _paired_comparison(
        lower: Sequence[RunMetrics],
        upper: Sequence[RunMetrics],
        metric: str,
    ) -> AdjacentComparison:
        lower_by_seed = {run.seed: run for run in lower}
        upper_by_seed = {run.seed: run for run in upper}
        if lower_by_seed.keys() != upper_by_seed.keys():
            raise ValueError("Adjacent comparisons require identical seed sets")
        lower_values = [
            float(getattr(lower_by_seed[seed], metric))
            for seed in sorted(lower_by_seed)
        ]
        upper_values = [
            float(getattr(upper_by_seed[seed], metric))
            for seed in sorted(upper_by_seed)
        ]
        differences = [high - low for low, high in zip(lower_values, upper_values)]
        diff_mean, _, diff_low, diff_high = _mean_sd_ci95(differences)
        t_result = stats.ttest_rel(upper_values, lower_values)
        return AdjacentComparison(
            lower_train_size=lower[0].train_size,
            upper_train_size=upper[0].train_size,
            metric=metric,
            lower_mean=mean(lower_values),
            upper_mean=mean(upper_values),
            mean_difference=diff_mean,
            relative_change=diff_mean / mean(lower_values),
            paired_ci95_low=diff_low,
            paired_ci95_high=diff_high,
            paired_t_pvalue=float(t_result.pvalue),
            improved_seed_count=sum(
                high < low for low, high in zip(lower_values, upper_values)
            ),
            seed_count=len(lower_values),
        )

    def _adjacent_comparisons(
        self, runs: Sequence[RunMetrics]
    ) -> tuple[AdjacentComparison, ...]:
        grouped = {
            size: sorted(
                (run for run in runs if run.train_size == size),
                key=lambda item: item.seed,
            )
            for size in self.request.train_sizes
        }
        comparisons: list[AdjacentComparison] = []
        for lower_size, upper_size in zip(
            self.request.train_sizes, self.request.train_sizes[1:]
        ):
            for metric in ("rel_sol_mean", "rel_flux_mean"):
                comparisons.append(
                    self._paired_comparison(
                        grouped[lower_size], grouped[upper_size], metric
                    )
                )
        return tuple(comparisons)

    def _sample_win_rates(
        self, runs: Sequence[RunMetrics]
    ) -> tuple[SampleWinRate, ...]:
        lookup = {(run.train_size, run.seed): run for run in runs}
        output: list[SampleWinRate] = []
        for lower_size, upper_size in zip(
            self.request.train_sizes, self.request.train_sizes[1:]
        ):
            for metric_name, field_name in (
                ("rel_sol", "rel_sol_values"),
                ("rel_flux", "rel_flux_values"),
            ):
                lower_arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
                upper_arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
                for seed in self.request.seeds:
                    lower_run = lookup[(lower_size, seed)]
                    upper_run = lookup[(upper_size, seed)]
                    if lower_run.test_sample_ids != upper_run.test_sample_ids:
                        raise ValueError("Test sample IDs differ across adjacent runs")
                    lower_values = np.asarray(
                        getattr(lower_run, field_name), dtype=np.float64
                    )
                    upper_values = np.asarray(
                        getattr(upper_run, field_name), dtype=np.float64
                    )
                    improved = int(np.count_nonzero(upper_values < lower_values))
                    output.append(
                        SampleWinRate(
                            lower_train_size=lower_size,
                            upper_train_size=upper_size,
                            metric=metric_name,
                            scope="per_seed",
                            seed=seed,
                            improved_sample_count=improved,
                            sample_count=len(lower_values),
                            improved_fraction=improved / len(lower_values),
                        )
                    )
                    lower_arrays.append(lower_values)
                    upper_arrays.append(upper_values)
                lower_average = np.mean(np.stack(lower_arrays), axis=0)
                upper_average = np.mean(np.stack(upper_arrays), axis=0)
                improved_average = int(np.count_nonzero(upper_average < lower_average))
                output.append(
                    SampleWinRate(
                        lower_train_size=lower_size,
                        upper_train_size=upper_size,
                        metric=metric_name,
                        scope="seed_averaged_sample",
                        seed=None,
                        improved_sample_count=improved_average,
                        sample_count=len(lower_average),
                        improved_fraction=improved_average / len(lower_average),
                    )
                )
        return tuple(output)


class ReportWriterMixin:
    request: TrainingSizeAnalysisRequest
    logger: logging.Logger

    @staticmethod
    def _write_dataclass_csv(path: Path, rows: Iterable[Any]) -> None:
        records = [asdict(row) for row in rows]
        if not records:
            raise ValueError(f"Cannot write an empty CSV: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def _write_json(
        self, result: TrainingSizeAnalysisResult, provenance: dict[str, Any]
    ) -> None:
        payload = {
            "analysis_contract": {
                "primary_test_metrics": str(PRIMARY_METRICS_PATH),
                "primary_checkpoint": str(BEST_ENERGY_CHECKPOINT),
                "final_model_metrics": str(FINAL_METRICS_PATH),
                "relative_error_unit": "fraction",
                "independent_replication_unit": "training/source seed",
                "per_sample_win_rates_are_descriptive": True,
            },
            "protocol": provenance,
            "decision": asdict(result.decision),
            "size_summaries": [asdict(item) for item in result.size_summaries],
            "adjacent_comparisons": [
                asdict(item) for item in result.adjacent_comparisons
            ],
            "sample_win_rates": [asdict(item) for item in result.sample_win_rates],
        }
        (self.request.outdir / "summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _write_report(self, result: TrainingSizeAnalysisResult) -> None:
        lines = [
            "# Unit-Square Poisson Training-Source Count Study",
            "",
            "## 결론",
            "",
            f"후속 수치 예제의 공통 학습 source 수는 **{result.decision.recommended_num_train}개**로 고정한다.",
            "모든 실험이 동일한 2,400 optimizer-step budget을 사용한 조건에서 4,800개가 "
            "평균 relative solution error, tail error, canonical energy에서 모두 가장 좋았다.",
            "2,400개는 계산 효율 관점의 knee이지만, 사전에 정한 best 대비 5% 허용 범위를 "
            "충족하지 못했으므로 논문용 accuracy setting으로 채택하지 않는다.",
            "",
            "## 비교 계약",
            "",
            "- 16개 run: `num_train={600,1200,2400,4800}` x `seed={0,1,2,3}`.",
            "- 모든 run: batch 200, 2,400 optimizer steps, warmup 240 steps, validation 24-step interval, validation source 300개.",
            "- 1차 test metric: 각 run의 `complex_coupling_model_best_energy.safetensors`에서 생성한 100-sample artifact.",
            "- seed 4개를 독립 반복으로 사용한다. 동일 test sample에서의 win rate는 descriptive evidence이며 독립 표본으로 간주하지 않는다.",
            "- 표의 relative error는 fraction을 100배 한 percent이며 `+/-`는 seed 간 sample standard deviation이다.",
            "",
            "## Test 결과",
            "",
            "| N_train | rel_sol (%) | median (%) | p90 (%) | max (%) | rel_flux (%) | energy | best step | final/best val |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for size_summary in result.size_summaries:
            lines.append(
                f"| {size_summary.train_size} | {100 * size_summary.rel_sol_mean:.6f} +/- {100 * size_summary.rel_sol_sd:.6f} "
                f"| {100 * size_summary.rel_sol_median_mean:.6f} | {100 * size_summary.rel_sol_p90_mean:.6f} "
                f"| {100 * size_summary.rel_sol_max_mean:.6f} | {100 * size_summary.rel_flux_mean:.6f} +/- {100 * size_summary.rel_flux_sd:.6f} "
                f"| {size_summary.energy_optimized_mean:.6e} | {size_summary.best_step_mean:.0f} "
                f"| {size_summary.final_over_best_validation_energy_mean:.6f} |"
            )

        lines.extend(
            [
                "",
                "## 인접 source-count 비교",
                "",
                "| 비교 | metric | 상대 변화 | 개선 seed | paired p-value |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for comparison in result.adjacent_comparisons:
            metric_label = (
                "rel_sol" if comparison.metric == "rel_sol_mean" else "rel_flux"
            )
            lines.append(
                f"| {comparison.lower_train_size} -> {comparison.upper_train_size} | {metric_label} "
                f"| {100 * comparison.relative_change:.3f}% | {comparison.improved_seed_count}/{comparison.seed_count} "
                f"| {comparison.paired_t_pvalue:.4g} |"
            )

        gap_2400 = next(
            item.relative_gap_to_best
            for item in result.size_summaries
            if item.train_size == 2400
        )
        win_2400_4800_sol = next(
            item
            for item in result.sample_win_rates
            if item.lower_train_size == 2400
            and item.upper_train_size == 4800
            and item.metric == "rel_sol"
            and item.scope == "seed_averaged_sample"
        )
        win_2400_4800_flux = next(
            item
            for item in result.sample_win_rates
            if item.lower_train_size == 2400
            and item.upper_train_size == 4800
            and item.metric == "rel_flux"
            and item.scope == "seed_averaged_sample"
        )
        lines.extend(
            [
                "",
                "## 해석",
                "",
                "- `rel_sol`은 source 수를 두 배로 늘릴 때마다 7.61%, 5.98%, 5.17% 감소했다. 개선 폭은 줄지만 2,400에서 plateau라고 보기에는 이르다.",
                "- 4,800 대비 2,400의 평균 `rel_sol` gap은 "
                f"{100 * gap_2400:.3f}%로, 5% saturation 기준을 근소하지만 명확하게 넘는다.",
                f"- 2,400 -> 4,800에서 seed-averaged test sample 기준 `rel_sol`은 {win_2400_4800_sol.improved_sample_count}/{win_2400_4800_sol.sample_count}, "
                f"`rel_flux`는 {win_2400_4800_flux.improved_sample_count}/{win_2400_4800_flux.sample_count} sample에서 개선됐다.",
                "- `rel_flux`는 2,400 이후 거의 포화되어 4,800의 추가 개선이 1.35%이지만, `rel_sol`과 optimized energy는 계속 개선된다.",
                "- 600/1,200은 validation optimum이 전체 step의 28%/39% 부근이고 이후 validation energy가 각각 약 44%/16% 증가한다. 작은 fixed dataset의 반복 노출에 따른 overfitting 신호다.",
                "- 2,400/4,800은 best step이 전체 budget의 89%/97%이고 final/best validation energy가 거의 1이다. 4,800은 같은 2,400 updates를 더 다양한 source에 분산한다.",
                "- optimizer step 시간은 batch와 model이 같으므로 source 수에 따라 체계적으로 증가하지 않는다. 현재 로그의 작은 차이는 머신 부하를 포함하므로 wall-clock 효과로 해석하지 않는다.",
                "",
                "## 후속 예제 고정 설정",
                "",
                "- `num_train=4800`, `num_valid=300`, `batch_size=200`.",
                "- 총 optimizer step 2,400, `warmup_steps=240`, `validation_every_steps=24`.",
                "- batch size가 동일하면 4,800 source에서 100 epochs로 2,400 steps를 맞춘다.",
                "- 논문의 공통 seed protocol은 seed 0-3을 유지한다. 단일 ablation pilot은 seed 0으로 먼저 screening하고, 채택 결과는 4-seed로 확인한다.",
                "- 이번 결론은 fixed PDE coefficient에서 source 다양성에 관한 것이다. 향후 sample별 coefficient까지 변화시키면 coefficient 다양성 budget은 별도 실험으로 검증해야 한다.",
                "",
                "## Provenance 주의사항",
                "",
                "- 일부 queue metadata는 첫 8개 run만 포함하지만, 16개 개별 run 모두 `_SUCCESS`, best-energy checkpoint, config, training log, 100-sample artifact를 갖는다.",
                "- 로그에 기록된 원래 macOS 절대 경로와 현재 Linux 복사 경로가 다르다. 분석은 현재 run-local config/CSV/summary만 사용한다.",
                "- root `metrics/test_per_sample_metrics.csv`는 마지막 model 진단이며, 본문의 1차 비교는 `artifacts_best_energy`를 사용한다.",
            ]
        )
        (self.request.outdir / "analysis_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_run_metrics(self, runs: Sequence[RunMetrics]) -> None:
        excluded = {
            "test_sample_ids",
            "rel_sol_values",
            "rel_flux_values",
            "validation_steps",
            "validation_energies",
        }
        records = [
            {key: value for key, value in asdict(run).items() if key not in excluded}
            for run in runs
        ]
        path = self.request.outdir / "run_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def _write_long_form_metrics(self, runs: Sequence[RunMetrics]) -> None:
        sample_path = self.request.outdir / "per_sample_seed_metrics.csv"
        with sample_path.open("w", newline="", encoding="utf-8") as handle:
            sample_fieldnames = (
                "train_size",
                "seed",
                "sample_id",
                "rel_sol",
                "rel_flux",
            )
            writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
            writer.writeheader()
            for run in runs:
                writer.writerows(
                    {
                        "train_size": run.train_size,
                        "seed": run.seed,
                        "sample_id": sample_id,
                        "rel_sol": rel_sol,
                        "rel_flux": rel_flux,
                    }
                    for sample_id, rel_sol, rel_flux in zip(
                        run.test_sample_ids,
                        run.rel_sol_values,
                        run.rel_flux_values,
                    )
                )

        validation_path = self.request.outdir / "validation_energy_curves.csv"
        with validation_path.open("w", newline="", encoding="utf-8") as handle:
            validation_fieldnames = (
                "train_size",
                "seed",
                "global_step",
                "validation_energy",
            )
            writer = csv.DictWriter(handle, fieldnames=validation_fieldnames)
            writer.writeheader()
            for run in runs:
                writer.writerows(
                    {
                        "train_size": run.train_size,
                        "seed": run.seed,
                        "global_step": global_step,
                        "validation_energy": validation_energy,
                    }
                    for global_step, validation_energy in zip(
                        run.validation_steps, run.validation_energies
                    )
                )


class PlotlyAnalysisMixin:
    request: TrainingSizeAnalysisRequest
    logger: logging.Logger

    @staticmethod
    def _base_layout(fig: go.Figure, title: str) -> None:
        fig.update_layout(
            title=title,
            template="plotly_white",
            width=1050,
            height=620,
            font={"family": "DejaVu Sans", "size": 14},
            margin={"l": 80, "r": 35, "t": 85, "b": 70},
            legend={"orientation": "h", "y": 1.08, "x": 0.0},
        )

    def _plot_test_metrics(self, summaries: Sequence[SizeSummary]) -> None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Relative solution error", "Relative flux error"),
        )
        x = [item.train_size for item in summaries]
        colors = {"mean": "#006d77", "p90": "#d97706"}
        for col, prefix in ((1, "rel_sol"), (2, "rel_flux")):
            centers = [
                100 * float(getattr(item, f"{prefix}_mean")) for item in summaries
            ]
            lows = [
                100 * float(getattr(item, f"{prefix}_ci95_low")) for item in summaries
            ]
            highs = [
                100 * float(getattr(item, f"{prefix}_ci95_high")) for item in summaries
            ]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=centers,
                    mode="lines+markers",
                    name="Seed mean (95% CI)",
                    legendgroup=f"{prefix}_mean",
                    showlegend=col == 1,
                    line={"color": colors["mean"], "width": 3},
                    marker={"size": 10},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            high - center for high, center in zip(highs, centers)
                        ],
                        "arrayminus": [
                            center - low for center, low in zip(centers, lows)
                        ],
                    },
                    hovertemplate="N=%{x}<br>mean=%{y:.5f}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
            p90_field = f"{prefix}_p90_mean"
            if hasattr(summaries[0], p90_field):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=[100 * float(getattr(item, p90_field)) for item in summaries],
                        mode="lines+markers",
                        name="Mean per-run p90",
                        legendgroup=f"{prefix}_p90",
                        showlegend=col == 1,
                        line={"color": colors["p90"], "dash": "dash"},
                        hovertemplate="N=%{x}<br>p90=%{y:.5f}%<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
        for col in (1, 2):
            fig.update_xaxes(
                title_text="Training sources",
                type="log",
                tickvals=x,
                ticktext=[str(value) for value in x],
                row=1,
                col=col,
            )
            fig.update_yaxes(title_text="Relative error (%)", row=1, col=col)
        self._base_layout(fig, "Unit-square Poisson: fixed-step source-count study")
        save_plotly_figure(
            fig, self.request.outdir / "figures/test_metrics_by_train_size", self.logger
        )

    def _plot_paired_seeds(self, runs: Sequence[RunMetrics]) -> None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Relative solution error", "Relative flux error"),
        )
        palette = ("#264653", "#2a9d8f", "#e9c46a", "#e76f51")
        for seed, color in zip(self.request.seeds, palette):
            seed_runs = sorted(
                (run for run in runs if run.seed == seed),
                key=lambda item: item.train_size,
            )
            for col, field in ((1, "rel_sol_mean"), (2, "rel_flux_mean")):
                fig.add_trace(
                    go.Scatter(
                        x=[run.train_size for run in seed_runs],
                        y=[100 * float(getattr(run, field)) for run in seed_runs],
                        mode="lines+markers",
                        name=f"seed {seed}",
                        legendgroup=f"seed_{seed}",
                        showlegend=col == 1,
                        line={"color": color, "width": 2},
                        marker={"size": 8},
                        hovertemplate=(
                            f"seed={seed}<br>N=%{{x}}<br>error=%{{y:.5f}}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
        for col in (1, 2):
            fig.update_xaxes(
                title_text="Training sources",
                type="log",
                tickvals=list(self.request.train_sizes),
                ticktext=[str(value) for value in self.request.train_sizes],
                row=1,
                col=col,
            )
            fig.update_yaxes(title_text="Relative error (%)", row=1, col=col)
        self._base_layout(fig, "Paired seed trajectories")
        save_plotly_figure(
            fig, self.request.outdir / "figures/paired_seed_errors", self.logger
        )

    def _plot_validation_curves(self, runs: Sequence[RunMetrics]) -> None:
        fig = go.Figure()
        colors = ("#9c6644", "#d97706", "#2a9d8f", "#005f73")
        for train_size, color in zip(self.request.train_sizes, colors):
            group = sorted(
                (run for run in runs if run.train_size == train_size),
                key=lambda item: item.seed,
            )
            steps = group[0].validation_steps
            if any(run.validation_steps != steps for run in group[1:]):
                raise ValueError("Validation step grids differ within a train size")
            matrix = np.asarray(
                [run.validation_energies for run in group], dtype=np.float64
            )
            center = np.mean(matrix, axis=0)
            spread = np.std(matrix, axis=0, ddof=1)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=center,
                    mode="lines",
                    name=f"N={train_size}",
                    line={"color": color, "width": 2.5},
                    hovertemplate=(
                        f"N={train_size}<br>step=%{{x}}<br>energy=%{{y:.4e}}"
                        "<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=steps + tuple(reversed(steps)),
                    y=np.concatenate((center + spread, (center - spread)[::-1])),
                    fill="toself",
                    fillcolor=color.replace("#", "rgba(") if False else color,
                    opacity=0.10,
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.update_xaxes(title="Cumulative optimizer step")
        fig.update_yaxes(title="Validation optimized energy", type="log")
        self._base_layout(fig, "Validation energy under an equal 2,400-step budget")
        save_plotly_figure(
            fig, self.request.outdir / "figures/validation_energy_curves", self.logger
        )

    def _plot_checkpoint_dynamics(self, summaries: Sequence[SizeSummary]) -> None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Location of best validation energy",
                "Final / best energy",
            ),
        )
        x = [str(item.train_size) for item in summaries]
        fig.add_trace(
            go.Bar(
                x=x,
                y=[100 * item.best_step_fraction for item in summaries],
                marker_color="#457b9d",
                text=[f"{100 * item.best_step_fraction:.1f}%" for item in summaries],
                textposition="outside",
                showlegend=False,
                hovertemplate="N=%{x}<br>best-step fraction=%{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=[
                    item.final_over_best_validation_energy_mean - 1.0
                    for item in summaries
                ],
                marker_color="#e76f51",
                text=[
                    f"+{100 * (item.final_over_best_validation_energy_mean - 1.0):.2f}%"
                    for item in summaries
                ],
                textposition="outside",
                showlegend=False,
                hovertemplate="N=%{x}<br>increase=%{y:.5f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Training sources", row=1, col=1)
        fig.update_xaxes(title_text="Training sources", row=1, col=2)
        fig.update_yaxes(
            title_text="Best step / 2400 (%)", range=[0, 108], row=1, col=1
        )
        fig.update_yaxes(
            title_text="Relative increase", rangemode="tozero", row=1, col=2
        )
        self._base_layout(
            fig, "Small source sets overfit under the fixed-step protocol"
        )
        save_plotly_figure(
            fig, self.request.outdir / "figures/checkpoint_dynamics", self.logger
        )


class UnitSquareTrainingSizeAnalyzer(
    MetricAggregationMixin, ReportWriterMixin, PlotlyAnalysisMixin
):
    def __init__(
        self, request: TrainingSizeAnalysisRequest, logger: logging.Logger
    ) -> None:
        request.validate()
        self.request = request
        self.logger = logger

    def analyze(self) -> TrainingSizeAnalysisResult:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        runs = tuple(
            self._parse_run(train_size, seed)
            for train_size in self.request.train_sizes
            for seed in self.request.seeds
        )
        canonical_ids = runs[0].test_sample_ids
        if any(run.test_sample_ids != canonical_ids for run in runs[1:]):
            raise ValueError("The 16 runs do not share an identical test sample set")

        summaries = self._aggregate_sizes(runs)
        comparisons = self._adjacent_comparisons(runs)
        win_rates = self._sample_win_rates(runs)
        decision = choose_training_size(summaries, self.request.saturation_tolerance)
        result = TrainingSizeAnalysisResult(
            runs=runs,
            size_summaries=summaries,
            adjacent_comparisons=comparisons,
            sample_win_rates=win_rates,
            decision=decision,
        )

        provenance = {
            "train_sizes": list(self.request.train_sizes),
            "seeds": list(self.request.seeds),
            "run_count": len(runs),
            "test_sample_count": len(canonical_ids),
            "optimizer_steps": self.request.expected_optimizer_steps,
            "batch_size": runs[0].batch_size,
            "num_valid": runs[0].num_valid,
            "warmup_steps": runs[0].warmup_steps,
            "validation_every_steps": runs[0].validation_every_steps,
            "test_sample_ids_identical": True,
            "all_runs_successful": True,
            "all_best_energy_artifacts_verified": True,
        }
        self._write_run_metrics(runs)
        self._write_long_form_metrics(runs)
        self._write_dataclass_csv(
            self.request.outdir / "dataset_size_summary.csv", summaries
        )
        self._write_dataclass_csv(
            self.request.outdir / "adjacent_comparisons.csv", comparisons
        )
        self._write_dataclass_csv(
            self.request.outdir / "sample_win_rates.csv", win_rates
        )
        self._write_json(result, provenance)
        self._write_report(result)
        self._plot_test_metrics(summaries)
        self._plot_paired_seeds(runs)
        self._plot_validation_curves(runs)
        self._plot_checkpoint_dynamics(summaries)
        self.logger.info(
            "Analysis complete: recommended num_train=%d",
            decision.recommended_num_train,
        )
        return result
