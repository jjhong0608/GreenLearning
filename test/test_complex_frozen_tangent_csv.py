from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from greenonet.complex_frozen_tangent_csv import (
    BEST_CHECKPOINT,
    FrozenTangentCsvRequest,
    FrozenTangentCsvAudit,
    FrozenTangentBenchmark,
    run_frozen_tangent_csv,
)
from greenonet.frozen_tangent_csv_metrics import aggregate_samples, aggregate_seeds
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.coefficients import load_coefficient_functions
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from test.test_complex_projection_response_audit import _write_fixture
from test.test_complex_tangent_subspace_audit import _context
from greenonet.complex_tangent_projection import matrix_free_krylov_subspace_step
from greenonet.complex_tangent_context_io import TangentResponseContextStore


def test_request_defaults_and_validation(tmp_path: Path) -> None:
    args = dict(run_dirs=(tmp_path / "seed0",), outdir=tmp_path / "audit", device="cpu")
    request = FrozenTangentCsvRequest(**args)
    assert (request.baseline_k, request.max_k) == (10, 64)
    assert (request.batch_size, request.num_threads) == (10, 4)
    assert not request.benchmark
    for invalid in (
        {"baseline_k": 1},
        {"max_k": 9},
        {"batch_size": 0},
        {"num_threads": True},
        {"timing_repeats": 0},
        {"warmup_repeats": -1},
        {"device": "cuda"},
        {"device": "mps"},
    ):
        with pytest.raises((ValueError, TypeError)):
            FrozenTangentCsvRequest(**(args | invalid))


def test_ratio_of_means_and_seed_sd() -> None:
    rows = [
        dict(
            run_id="a",
            seed=0,
            training_k=10,
            evaluation_k=10,
            sample_id=i,
            response_cost=value,
            response_cost_previous=previous,
            response_cost_baseline=value,
            rel_sol=value,
            rel_sol_baseline=value,
        )
        for i, (value, previous) in enumerate(((1.0, 2.0), (9.0, 10.0)))
    ]
    aggregated = aggregate_samples(rows)[0]
    assert aggregated["response_cost_ratio_previous"] == pytest.approx(10.0 / 12.0)
    assert aggregated["response_cost_gain_baseline"] == 0.0
    assert aggregate_seeds([aggregated])[0]["response_cost_mean_seed_std"] is None
    other = aggregated | {"run_id": "b", "seed": 1, "response_cost_mean": 7.0}
    summary = aggregate_seeds([aggregated, other])[0]
    assert summary["response_cost_mean_seed_mean"] == 6.0
    assert summary["response_cost_mean_seed_std"] == pytest.approx(2.0**0.5)
    assert summary["num_seeds"] == 2


def test_zero_denominator_is_missing_not_a_fake_gain() -> None:
    row = dict(
        run_id="a",
        seed=0,
        training_k=10,
        evaluation_k=10,
        sample_id=0,
        response_cost=0.0,
        response_cost_previous=0.0,
        response_cost_baseline=0.0,
    )
    result = aggregate_samples([row])[0]
    assert result["response_cost_ratio_previous"] is None
    assert result["response_cost_gain_baseline"] is None


def _make_runs(tmp_path: Path, *, k: int = 10, count: int = 2) -> tuple[Path, ...]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config, checkpoint, green, geometry, test, coefficients = _write_fixture(inputs)
    with np.load(test / "sample_0000.npz") as archive:
        second = {key: value * 0.7 for key, value in archive.items()}
    np.savez(test / "sample_0001.npz", **second)
    payload = json.loads(config.read_text())
    payload["coupling_model"]["balance_projection"] = dict(
        enabled=True,
        mode="symmetric_tangent_green_response",
        symmetric_tangent_green_response=dict(
            subspace_dimension=k,
            max_subspace_dimension=max(k, 8),
            eta_strategy="closed_loop_exact_line_search",
            eta_cap_enabled=False,
        ),
    )
    payload["pipeline"] = {"green_pretrained_path": str(green)}
    runs = []
    for seed in range(count):
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        payload["coupling_training"]["seed"] = seed
        path = run / "config_used.json"
        path.write_text(json.dumps(payload))
        shutil.copyfile(checkpoint, run / BEST_CHECKPOINT)
        configs = load_coupling_artifact_configs(path)
        dataset = ComplexCouplingDataset(
            test,
            load_complex_geometry(geometry, dtype=torch.float64),
            load_coefficient_functions(coefficients),
            branch_input_dim=4,
            dtype=torch.float64,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        loader = ComplexCouplingArtifactExporter(
            CouplingArtifactRequest(
                config=path,
                coupling_checkpoint=run / BEST_CHECKPOINT,
                green_checkpoint=green,
                outdir=tmp_path / "unused",
            )
        )
        evaluator = ComplexCouplingEvaluator(
            model=loader._load_complex_model(configs, torch.device("cpu")),
            green_model=loader._load_green_model(configs, torch.device("cpu")),
            config=configs.coupling_training,
            device=torch.device("cpu"),
            work_dir=tmp_path / f"fixture_baseline_{seed}",
        )
        rows = []
        with torch.no_grad():
            for batch in DataLoader(
                dataset, batch_size=1, collate_fn=complex_coupling_collate_fn
            ):
                prediction = evaluator.predict_batch(batch)
                row = evaluator._sample_metric_row(prediction, 0)
                row.update(
                    sample_id=int(batch.sample_indices[0]),
                    file_stem=batch.file_stems[0],
                )
                rows.append(row)
        artifacts = run / "artifacts_best_energy"
        (artifacts / "metrics").mkdir(parents=True)
        (artifacts / "summary.json").write_text(
            json.dumps({"coupling_checkpoint": str(run / BEST_CHECKPOINT)})
        )
        with (artifacts / "metrics" / "per_sample_metrics.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for handler in list(evaluator.logger.handlers):
            evaluator.logger.removeHandler(handler)
            handler.close()
        runs.append(run)
    return tuple(runs)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_sequential_checkpoints_baseline_csv_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _make_runs(tmp_path)
    snapshots = {
        path: path.read_bytes()
        for run in runs
        for path in run.rglob("*")
        if path.is_file()
    }
    import plotly.graph_objects as go

    def no_plot(*args: object, **kwargs: object) -> None:
        raise AssertionError("CSV audit must not export figures")

    monkeypatch.setattr(go.Figure, "write_html", no_plot)
    monkeypatch.setattr(go.Figure, "write_image", no_plot)
    order = []
    original = FrozenTangentCsvAudit._run_one

    def capture(self: FrozenTangentCsvAudit, spec: object) -> None:
        order.append(spec.seed)
        original(self, spec)

    monkeypatch.setattr(FrozenTangentCsvAudit, "_run_one", capture)
    request = FrozenTangentCsvRequest(
        run_dirs=runs, outdir=tmp_path / "csv", device="cpu", max_k=11, batch_size=2
    )
    metadata = run_frozen_tangent_csv(request)
    assert metadata["status"] == "complete"
    assert order == [0, 1]
    assert metadata["environment"]["compile"] is False
    rows = _read(request.outdir / "posthoc_per_sample.csv")
    assert len(rows) == 8
    assert {row["evaluation_k"] for row in rows} == {"10", "11"}
    assert all(float(row["balance_max_abs"]) < 1e-12 for row in rows)
    summary = _read(request.outdir / "posthoc_summary.csv")
    assert all(row["num_seeds"] == "2" for row in summary)
    checks = json.loads((request.outdir / "verification.json").read_text())
    assert all(run["baseline_samples"] == 2 for run in checks["runs"])
    assert checks["runs"][1]["context"]["reused_across_runs"]
    assert checks["runs"][1]["context"]["build_count"] == 1
    assert _read(request.outdir / "posthoc_timing.csv") == []
    assert not any(
        path.suffix in {".html", ".png", ".pdf", ".npz"}
        for path in request.outdir.rglob("*")
    )
    assert all(path.read_bytes() == original for path, original in snapshots.items())
    assert not any(
        (run / "tangent_response_context.safetensors").exists() for run in runs
    )
    second = replace(request, outdir=tmp_path / "batch1", batch_size=1)
    run_frozen_tangent_csv(second)
    other = _read(second.outdir / "posthoc_per_sample.csv")
    by_key = {(r["run_id"], r["evaluation_k"], r["sample_id"]): r for r in other}
    for row in rows:
        same = by_key[(row["run_id"], row["evaluation_k"], row["sample_id"])]
        for key in ("response_cost", "rel_sol", "rel_u_phi", "rel_u_psi"):
            assert float(row[key]) == pytest.approx(
                float(same[key]), rel=1e-9, abs=1e-12
            )


def test_benchmark_cpu_small_fixture(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path, k=2, count=1)
    request = FrozenTangentCsvRequest(
        run_dirs=runs,
        outdir=tmp_path / "timing",
        device="cpu",
        baseline_k=2,
        max_k=3,
        benchmark=True,
        warmup_repeats=1,
        timing_repeats=2,
    )
    run_frozen_tangent_csv(request)
    rows = _read(request.outdir / "posthoc_timing.csv")
    assert len(rows) == 8
    assert {r["scope"] for r in rows} == {"tangent_only", "prediction_forward"}
    assert all(float(row["seconds"]) > 0 for row in rows)
    assert all(
        row["peak_allocated_mib"] == "" and row["memory_status"] == "cpu_unmeasured"
        for row in rows
    )


def test_cuda_timer_synchronizes_and_retains_result_through_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("sync", str(device)))
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 3 * 2**20)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append(("reset", str(device))),
    )
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(
        "greenonet.complex_frozen_tangent_csv.time.perf_counter", lambda: next(clock)
    )
    measured = FrozenTangentBenchmark(torch.device("cuda:1")).measure(
        lambda: events.append(("call", ""))
    )
    assert measured["seconds"] == 2.0
    assert (
        measured["start_allocated_mib"] == 1.0 and measured["peak_allocated_mib"] == 3.0
    )
    assert events == [
        ("sync", "cuda:1"),
        ("reset", "cuda:1"),
        ("call", ""),
        ("sync", "cuda:1"),
    ]


def test_k64_all_prefixes_match_independent_calls_and_degenerate_directions() -> None:
    context = _context()
    mismatch = torch.tensor([[1.0, -0.4, 0.2], [0.0, 0.0, 0.0]], dtype=torch.float64)
    gradient = context.tangent_gradient(mismatch)
    extended = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=64,
        relative_eps=1e-12,
        monotonicity_relative_tol=1e-10,
    )
    assert torch.isfinite(extended.deltas).all()
    assert not extended.direction_active[:, 1].any()
    assert torch.count_nonzero(extended.coefficients[:, 1]) == 0
    for k in range(2, 65):
        independent = matrix_free_krylov_subspace_step(
            context=context,
            mismatch=mismatch,
            gradient=gradient,
            max_dimension=k,
            relative_eps=1e-12,
            monotonicity_relative_tol=1e-10,
        )
        torch.testing.assert_close(
            extended.deltas[k - 1], independent.final_delta, rtol=0, atol=0
        )
        torch.testing.assert_close(
            extended.costs[k - 1], independent.costs[-1], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_run",
        "duplicate_seed",
        "different_model",
        "missing_checkpoint",
        "missing_artifact",
        "missing_reference",
        "bad_artifact",
        "bad_sidecar",
    ],
)
def test_input_failures_do_not_publish_success(tmp_path: Path, failure: str) -> None:
    runs = _make_runs(
        tmp_path,
        k=2,
        count=2 if failure in {"duplicate_seed", "different_model"} else 1,
    )
    if failure == "duplicate_run":
        runs = (runs[0], runs[0])
    elif failure in {"duplicate_seed", "different_model"}:
        path = runs[1] / "config_used.json"
        payload = json.loads(path.read_text())
        if failure == "duplicate_seed":
            payload["coupling_training"]["seed"] = 0
        else:
            payload["coupling_model"]["hidden_dim"] = 5
        path.write_text(json.dumps(payload))
    elif failure == "missing_checkpoint":
        (runs[0] / BEST_CHECKPOINT).unlink()
    elif failure == "missing_artifact":
        (runs[0] / "artifacts_best_energy" / "summary.json").unlink()
    elif failure == "missing_reference":
        path = tmp_path / "inputs" / "test" / "sample_0001.npz"
        with np.load(path) as archive:
            payload = {key: value for key, value in archive.items() if key != "sol"}
        np.savez(path, **payload)
    elif failure == "bad_artifact":
        path = runs[0] / "artifacts_best_energy" / "metrics" / "per_sample_metrics.csv"
        rows = _read(path)
        rows[-1]["rel_sol"] = "123.0"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (runs[0] / "tangent_response_context.safetensors").write_bytes(
            b"corrupt sidecar"
        )
    request = FrozenTangentCsvRequest(
        run_dirs=runs, outdir=tmp_path / "failed", device="cpu", baseline_k=2, max_k=3
    )
    with pytest.raises(Exception):
        run_frozen_tangent_csv(request)
    assert (
        json.loads((request.outdir / "metadata.json").read_text())["status"] == "failed"
    )
    assert _read(request.outdir / "posthoc_summary.csv") == []
    assert _read(request.outdir / "posthoc_per_sample.csv") == []


def test_valid_sidecar_is_read_only_and_reference_independent(tmp_path: Path) -> None:
    from greenonet.complex_frozen_tangent_csv import _FrozenRunSession

    runs = _make_runs(tmp_path, k=2, count=1)
    request = FrozenTangentCsvRequest(
        run_dirs=runs,
        outdir=tmp_path / "read_only",
        device="cpu",
        baseline_k=2,
        max_k=3,
    )
    audit = FrozenTangentCsvAudit(replace(request, outdir=tmp_path / "sidecar_setup"))
    spec = audit._preflight()[0]
    session = _FrozenRunSession(audit, spec)
    try:
        batch = next(iter(session.loader))
        session._initialize_context(batch)
        cache = audit.shared_cache
        assert (
            cache is not None
            and cache.context is not None
            and cache.identity is not None
        )
        sidecar = runs[0] / "tangent_response_context.safetensors"
        TangentResponseContextStore.save(
            path=sidecar, context=cache.context, identity=cache.identity
        )
        before = session._prepare(batch)
        altered = replace(
            batch, sol_valid=batch.sol_valid + 10, flux_valid=batch.flux_valid * -20
        )
        after = session._prepare(altered)
        for field in ("symmetric_physical", "mismatch", "gradient"):
            assert torch.equal(getattr(before, field), getattr(after, field))
        assert torch.equal(
            session._krylov(before, 3).final_delta,
            session._krylov(after, 3).final_delta,
        )
    finally:
        session.close()
    snapshot = sidecar.read_bytes()
    run_frozen_tangent_csv(request)
    assert sidecar.read_bytes() == snapshot
    verification = json.loads((request.outdir / "verification.json").read_text())
    assert verification["runs"][0]["context"]["source"] == "loaded"
    assert verification["runs"][0]["context"]["build_count"] == 0
    with pytest.raises(FileExistsError):
        run_frozen_tangent_csv(request)
    run_frozen_tangent_csv(replace(request, overwrite=True))
    assert sidecar.read_bytes() == snapshot


def test_cli_help_exposes_explicit_device_and_defaults(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).parents[1] / "cli" / "audit_frozen_tangent_csv.py"
    spec = importlib.util.spec_from_file_location("audit_frozen_tangent_csv", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.AuditFrozenTangentCsvCLI().parser.parse_args(
        ["--run-dirs", "a", "b", "--outdir", str(tmp_path), "--device", "cpu"]
    )
    assert args.max_k == 64 and args.baseline_k == 10
    assert not args.benchmark


def test_default_k10_through_k64_csv_contract(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path, count=1)
    request = FrozenTangentCsvRequest(
        run_dirs=runs, outdir=tmp_path / "default64", device="cpu"
    )
    run_frozen_tangent_csv(request)
    rows = _read(request.outdir / "posthoc_per_sample.csv")
    assert len(rows) == 2 * 55
    assert {int(row["evaluation_k"]) for row in rows} == set(range(10, 65))
    baseline = [row for row in rows if row["evaluation_k"] == "10"]
    assert all(float(row["response_cost_gain_baseline"]) == 0.0 for row in baseline)
    assert all(float(row["response_cost"]) >= 0.0 for row in rows)
    assert all(
        int(row["effective_dimension"]) <= int(row["evaluation_k"]) for row in rows
    )
    assert all(
        float(row["minimum_reach"]) <= float(row["global_reach"]) for row in rows
    )


def test_benchmark_repeats_are_full_passes_excluding_metric_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from greenonet.complex_frozen_tangent_csv import _FrozenRunSession

    runs = _make_runs(tmp_path, k=2, count=1)
    request = FrozenTangentCsvRequest(
        run_dirs=runs,
        outdir=tmp_path / "repeat_audit",
        device="cpu",
        baseline_k=2,
        max_k=3,
        batch_size=1,
        benchmark=True,
        warmup_repeats=3,
        timing_repeats=5,
    )
    count = {"measure": 0, "rows": 0, "krylov": 0}
    original_rows = _FrozenRunSession._rows
    original_krylov = _FrozenRunSession._krylov

    def capture_rows(*args: object, **kwargs: object) -> object:
        count["rows"] += 1
        return original_rows(*args, **kwargs)

    def capture_krylov(*args: object, **kwargs: object) -> object:
        count["krylov"] += 1
        return original_krylov(*args, **kwargs)

    def measure(self: FrozenTangentBenchmark, call: object) -> dict[str, object]:
        before = count["rows"]
        call()
        assert count["rows"] == before
        count["measure"] += 1
        return dict(
            seconds=1.0,
            start_allocated_mib=None,
            peak_allocated_mib=None,
            memory_status="cpu_unmeasured",
        )

    monkeypatch.setattr(_FrozenRunSession, "_rows", capture_rows)
    monkeypatch.setattr(_FrozenRunSession, "_krylov", capture_krylov)
    monkeypatch.setattr(FrozenTangentBenchmark, "measure", measure)
    run_frozen_tangent_csv(request)
    assert count["measure"] == 2 * 2 * 5 * 2
    assert count["krylov"] == 4 + 2 * (3 + 5) * 2
    assert all(
        float(row["seconds"]) == 2.0
        for row in _read(request.outdir / "posthoc_timing.csv")
    )


def test_cuda_csv_smoke_when_gpu1_is_available(tmp_path: Path) -> None:
    import os

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA unavailable; CPU and synchronization mock tests cover this environment."
        )
    if torch.cuda.device_count() >= 2:
        device = "cuda:1"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "1":
        device = "cuda:0"
    else:
        pytest.skip("GPU:1 unavailable; do not allocate on reserved physical GPU:0.")
    runs = _make_runs(tmp_path, k=10, count=1)
    request = FrozenTangentCsvRequest(
        run_dirs=runs,
        outdir=tmp_path / "cuda",
        device=device,
        max_k=11,
        benchmark=True,
        warmup_repeats=1,
        timing_repeats=1,
    )
    run_frozen_tangent_csv(request)
    assert all(
        float(row["peak_allocated_mib"]) > 0
        for row in _read(request.outdir / "posthoc_timing.csv")
    )
