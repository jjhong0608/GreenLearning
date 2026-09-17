from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from docs.paper.numerical_examples.example_04_pentagram.build_tables import (
    PaperEvidenceBuilder,
    aggregate_seed_means,
    gain,
    percentage,
    reach_for_k,
    relative_directional_errors,
    validate_sample_grid,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper"


def test_seed_summary_is_not_pooled_or_population_sd() -> None:
    result = aggregate_seed_means([1.0, 2.0, 4.0, 5.0])
    assert result["mean"] == 3.0
    assert result["sd"] == pytest.approx(np.sqrt(10.0 / 3.0))
    assert gain(1.0, 4.0) == 0.75
    assert gain(0.0, 0.0) is None
    assert percentage(0.008) == pytest.approx(0.8)


def test_relative_flux_is_axis_mean_not_joint_norm() -> None:
    target = np.array([[1.0, 0.0], [0.0, 10.0]])
    predicted = np.array([[2.0, 0.0], [0.0, 10.0]])
    phi, psi, average, joint = relative_directional_errors(predicted, target)
    assert (phi, psi, average) == (1.0, 0.0, 0.5)
    assert joint == pytest.approx(1 / np.sqrt(101))
    assert average != pytest.approx(joint)


def test_k0_is_not_assigned_a_geometry_reach() -> None:
    reach = {1: {"global_reach": 0.001}}
    assert all(value is None for value in reach_for_k(0, reach).values())
    assert reach_for_k(1, reach)["global_reach"] == 0.001
    with pytest.raises(KeyError):
        reach_for_k(2, reach)


def test_sample_grid_rejects_duplicate_and_missing_ids() -> None:
    validate_sample_grid([{"sample_id": "0"}, {"sample_id": "1"}], 2)
    with pytest.raises(ValueError, match="sample"):
        validate_sample_grid([{"sample_id": "0"}, {"sample_id": "0"}], 2)
    with pytest.raises(ValueError, match="sample"):
        validate_sample_grid([{"sample_id": "0"}], 2)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("paper_tables")
    plan_hash = hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest()
    result = PaperEvidenceBuilder(ROOT, out).build()
    assert hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest() == plan_hash
    return out, result


def test_real_evidence_is_complete_and_reconciled(built: tuple[Path, dict]) -> None:
    _, result = built
    assert result["status"] == "complete"
    assert result["training_runs"] == 32
    assert result["training_sample_rows"] == 3200
    assert result["frozen_checkpoints"] == 4
    assert result["posthoc_sample_rows"] == 22000
    assert result["input_files_unchanged"] is True
    assert result["rel_flux_axis_mean_max_abs"] < 1e-12
    assert result["saved_field_samples_checked"] >= 32
    assert result["saved_field_axis_mean_max_abs"] < 1e-12
    assert result["reference_test_files_verified"] == 100
    assert result["cross_device_checkpoint_pairs"] == 0
    assert result["inference_executed"] is False
    assert result["training_executed"] is False
    assert result["holds"] == []


def test_saved_tables_reproduce_without_changing_their_bytes(
    built: tuple[Path, dict],
) -> None:
    out, _ = built
    saved = PAPER / "numerical_examples/example_04_pentagram"
    rebuilt_names = {p.name for p in (out / "tables").glob("*.csv")}
    assert rebuilt_names == {p.name for p in (saved / "tables").glob("*.csv")}
    for name in rebuilt_names:
        if name == "input_manifest.csv":
            # This source evolves; its archived hash remains historical provenance.
            with (out / "tables" / name).open() as handle:
                current = {r["path"]: r for r in csv.DictReader(handle)}
            with (saved / "tables" / name).open() as handle:
                archived = {r["path"]: r for r in csv.DictReader(handle)}
            for source in (
                "src/greenonet/complex_coupling_model.py",
                "src/greenonet/complex_coupling_evaluator.py",
                "src/greenonet/complex_tangent_projection.py",
                "src/greenonet/complex_projection.py",
                "src/greenonet/complex_tangent_preconditioner.py",
            ):
                live = current.pop(source)
                assert (
                    live["sha256"]
                    == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
                )
                assert int(live["bytes"]) == (ROOT / source).stat().st_size
                archived.pop(source)
            assert current == archived
        else:
            assert (out / "tables" / name).read_bytes() == (
                saved / "tables" / name
            ).read_bytes()
    assert (out / "verification.json").read_bytes() == (
        saved / "verification.json"
    ).read_bytes()


def test_training_posthoc_and_timing_tables_stay_separate(
    built: tuple[Path, dict],
) -> None:
    out, _ = built

    def rows(name: str) -> list[dict[str, str]]:
        with (out / "tables" / name).open() as handle:
            return list(csv.DictReader(handle))

    training = rows("training_by_k.csv")
    assert {int(r["K"]) for r in training} == {0, 1, 2, 3, 4, 5, 9, 10}
    assert next(r for r in training if r["K"] == "0")["global_reach"] == ""
    posthoc = rows("posthoc_by_k.csv")
    assert {int(r["K"]) for r in posthoc} == set(range(10, 65))
    assert all(r["training_K"] == "10" for r in posthoc)
    assert {r["hardware"] for r in rows("training_cost_by_device.csv")} == {
        "nvidia_a40",
        "mac_studio",
    }
    assert all(
        r["peak_allocated_mib_max"] == ""
        for r in rows("posthoc_cost_by_device.csv")
        if r["hardware"] == "mac_studio"
    )


def test_reading_only_builder_has_no_model_execution_imports() -> None:
    code = (
        PAPER / "numerical_examples/example_04_pentagram/build_tables.py"
    ).read_text()
    for forbidden in (
        "import torch",
        "import plotly",
        "predict_batch(",
        "subprocess",
        "exporter",
        "load_model(",
    ):
        assert forbidden not in code


def test_markdown_inventory_and_links() -> None:
    expected = [
        "README.md",
        "shared/notation.md",
        "shared/experiment_protocol.md",
        "sections/README.md",
        "manuscript/README.md",
        "numerical_examples/README.md",
        *[
            f"numerical_examples/example_04_pentagram/{name}.md"
            for name in (
                "writing_brief",
                "experiment_setup",
                "results_and_interpretation",
                "figures_and_tables",
                "provenance",
            )
        ],
    ]
    for name in expected:
        path = PAPER / name
        assert path.is_file(), name
        text = path.read_text()
        assert "TODO" not in text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (path.parent / target.split("#")[0]).resolve().exists(), (
                name,
                target,
            )
    provenance = (
        PAPER / "numerical_examples/example_04_pentagram/provenance.md"
    ).read_text()
    assert "rel_flux" in provenance and "joint" in provenance
    brief = (
        PAPER / "numerical_examples/example_04_pentagram/writing_brief.md"
    ).read_text()
    assert "Structural coverage" in brief
    assert "validation" in brief


def test_displayed_result_tables_match_unrounded_csv(built: tuple[Path, dict]) -> None:
    out, _ = built
    text = (
        PAPER / "numerical_examples/example_04_pentagram/results_and_interpretation.md"
    ).read_text()

    def source(name: str) -> list[dict[str, str]]:
        with (out / "tables" / name).open() as handle:
            return list(csv.DictReader(handle))

    def display(start: str, end: str) -> dict[int, list[str]]:
        block = text.split(start, 1)[1].split(end, 1)[0]
        cells = [
            line.strip("|").split("|")
            for line in block.splitlines()
            if line.startswith("|")
        ]
        return {
            int(row[0]): [c.strip() for c in row[1:]]
            for row in cells
            if row[0].strip().isdigit()
        }

    def mean_sd(row: dict[str, str], metric: str, places: int) -> str:
        return f"{100 * float(row[f'{metric}_mean']):.{places}f} +/- {100 * float(row[f'{metric}_sd']):.{places}f}"

    solution = display("### 1.1", "### 1.2")
    diagnostic = display("### 1.2", "## 2.")
    for row in source("training_by_k.csv"):
        k = int(row["K"])
        assert solution[k] == [
            mean_sd(row, "rel_sol", 4),
            f"{100 * float(row['rel_sol_equal_mean_mean']):.4f}",
            mean_sd(row, "rel_u_phi", 4),
            mean_sd(row, "rel_u_psi", 4),
        ]
        assert diagnostic[k][0] == mean_sd(row, "rel_flux", 4)
        assert float(diagnostic[k][1]) == pytest.approx(
            float(row["energy_bulk_mean"]), rel=5e-6
        )
        assert diagnostic[k][2:] == [
            f"{100 * float(row['rel_sol_p95_seed_mean']):.4f}",
            f"{100 * float(row['rel_sol_max_seed_mean']):.4f}",
        ]
    reach = display("## 2.", "## 3.")
    for row in source("geometry_reach.csv"):
        assert reach[int(row["K"])][:3] == [
            "NA" if row["K"] == "0" else f"{100 * float(row[key]):.6f}"
            for key in ("global_reach", "lower_5pct_reach", "minimum_reach")
        ]
    posthoc = display("### 5.1", "### 5.2")
    for row in source("posthoc_by_k.csv"):
        k = int(row["K"])
        if k not in posthoc:
            continue
        assert posthoc[k] == [
            mean_sd(row, "rel_sol", 5),
            *[
                f"{100 * float(row[f'{key}_mean']):.5f}"
                for key in ("rel_sol_equal_mean", "rel_u_phi", "rel_u_psi")
            ],
            f"{1 - float(row['response_gain_k10']):.5f}",
        ]
    training_time = display("## 4.", "## 5.")
    for row in source("training_cost_by_device.csv"):
        i = 0 if row["hardware"] == "nvidia_a40" else 1
        cells = training_time[int(row["K"])]
        assert cells[i] == f"{float(row['steady_epoch_seconds']):.2f}"
        assert cells[i + 2] == f"{float(row['trainer_wall_hours']):.3f}"
    post_time = display("### 5.3", "## 6.")
    for row in source("posthoc_cost_by_device.csv"):
        if row["scope"] != "prediction_forward" or int(row["K"]) not in post_time:
            continue
        i = 0 if row["hardware"] == "nvidia_a40" else 1
        assert (
            post_time[int(row["K"])][i]
            == f"{float(row['seconds_median_seed_mean']):.3f}"
        )


def test_build_rejects_output_in_original_evidence_directory() -> None:
    with pytest.raises(ValueError, match="docs/paper"):
        PaperEvidenceBuilder(ROOT, ROOT / "checkpoints/numerical_examples/pentagram")


def test_historical_hash_conflict_is_not_silently_accepted(tmp_path: Path) -> None:
    folder = tmp_path / "docs/analysis/pentagram_paper_20260905"
    folder.mkdir(parents=True)
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("changed")
    (folder / "input_manifest.json").write_text(
        '[{"path":"evidence.csv","sha256":"not-the-current-hash"}]'
    )
    builder = PaperEvidenceBuilder(tmp_path, tmp_path / "docs/paper")
    with pytest.raises(ValueError, match="Historical input hash changed"):
        builder.check_old_evidence()


def test_written_claims_match_paired_and_frozen_statistics(
    built: tuple[Path, dict],
) -> None:
    out, _ = built
    document = (
        PAPER / "numerical_examples/example_04_pentagram/results_and_interpretation.md"
    ).read_text()

    def rows(name: str) -> list[dict[str, str]]:
        with (out / "tables" / name).open() as handle:
            return list(csv.DictReader(handle))

    training = rows("training_runs.csv")
    for seed in range(4):
        group = sorted(
            [r for r in training if int(r["seed"]) == seed], key=lambda r: int(r["K"])
        )
        for key in ("rel_sol_mean", "rel_u_phi_mean", "rel_u_psi_mean"):
            assert np.all(np.diff([float(r[key]) for r in group]) < 0)
    by_k = {int(r["K"]): r for r in rows("training_by_k.csv")}
    for start, end in ((0, 4), (4, 9), (9, 10)):
        improvement = 1 - float(by_k[end]["rel_sol_mean"]) / float(
            by_k[start]["rel_sol_mean"]
        )
        assert f"{100 * improvement:.2f}%" in document
    paired = rows("training_paired_changes.csv")
    for start, end, expected in (
        (0, 4, 1.0),
        (4, 5, 0.6975),
        (4, 9, 0.77),
        (9, 10, 0.8175),
    ):
        selected = [
            r
            for r in paired
            if r["metric"] == "rel_sol"
            and int(r["start_K"]) == start
            and int(r["end_K"]) == end
        ]
        assert sum(int(r["improved_count"]) for r in selected) / 400 == expected
    for seed in range(4):
        group = [r for r in rows("posthoc_by_seed.csv") if int(r["seed"]) == seed]
        assert int(min(group, key=lambda r: float(r["rel_sol_mean"]))["K"]) == 29
    paired_post = [
        r
        for r in rows("posthoc_paired_changes.csv")
        if r["metric"] == "rel_sol" and r["start_K"] == "29" and r["end_K"] == "64"
    ]
    assert 1 - sum(int(r["improved_count"]) for r in paired_post) / 400 == 0.72
    post = {int(r["K"]): r for r in rows("posthoc_by_k.csv")}
    for key in (
        "response_cost_mean",
        "canonical_bulk_energy_mean",
        "rel_sol_mean",
        "rel_sol_equal_mean_mean",
        "rel_u_phi_mean",
        "rel_u_psi_mean",
    ):
        improvement = 1 - float(post[64][key]) / float(post[10][key])
        assert f"{100 * improvement:.2f}%" in document
    for start, end in ((11, 16), (17, 24), (25, 32), (33, 48), (49, 64)):
        mean_gain = np.mean(
            [float(post[k]["response_gain_previous"]) for k in range(start, end + 1)]
        )
        assert f"{100 * mean_gain:.3f}%" in document
