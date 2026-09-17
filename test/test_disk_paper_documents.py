import copy
import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from docs.paper.numerical_examples.example_02_disk.build_tables import (
    DiskEvidenceBuilder,
    gain,
    representative_samples,
    statistics,
    validate_grid,
    validate_pair,
    validate_selector,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper/numerical_examples/example_02_disk"


def test_statistics():
    values = list(range(50))
    s = statistics(values)
    assert s["mean"] == s["median"] == 24.5
    assert s["p95"] == pytest.approx(np.quantile(values, 0.95, method="linear"))
    assert s["maximum"] == 49
    assert gain(4, 1) == 0.75 and gain(0, 0) is None
    for bad in ([], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            statistics(bad)


def test_grid_and_ties():
    rows = [
        dict(method=m, seed=s, sample_id=i, file_stem=f"sample_{i:06d}", rel_sol=1.0)
        for m in ("identity", "separable")
        for s in range(4)
        for i in range(50)
    ]
    validate_grid(rows)
    assert [r["sample_id"] for r in representative_samples(rows)] == [0, 0]
    for bad in (rows[:-1], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            validate_grid(bad)
    rows[0]["file_stem"] = "wrong"
    with pytest.raises(ValueError):
        validate_grid(rows)


def test_config_and_selector_rejections():
    a = {
        "coupling_model": {
            "balance_projection": {
                "symmetric_tangent_green_response": {
                    "preconditioner_variant": "identity"
                }
            }
        }
    }
    b = copy.deepcopy(a)
    b["coupling_model"]["balance_projection"]["symmetric_tangent_green_response"][
        "preconditioner_variant"
    ] = "separable"
    validate_pair(a, b)
    b["other"] = 2
    with pytest.raises(ValueError):
        validate_pair(a, b)
    s = dict(
        checkpoint_selector="best_energy",
        coupling_checkpoint="/run/complex_coupling_model_best_energy.safetensors",
        checkpoint_selection={"reference_metric_used": False},
    )
    validate_selector(s, "run")
    for field, value in (
        ("checkpoint_selector", "final"),
        ("coupling_checkpoint", "/run/coupling_model.safetensors"),
    ):
        bad = dict(s, **{field: value})
        with pytest.raises(ValueError):
            validate_selector(bad, "run")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("disk_paper")
    result = DiskEvidenceBuilder(ROOT, out).build()
    return out, result


def test_real_reconciliation(built):
    out, result = built
    assert (result["runs"], result["rows"], result["unique_test_sources"]) == (
        8,
        400,
        50,
    )
    assert result["max_numeric_difference"] < 1e-10
    assert result["input_files_unchanged"] and result["test_activity_all_directions"]
    assert not result["inference_executed"] and not result["training_executed"]
    assert not result["figures_generated"]
    with (out / "tables/accuracy_and_tails_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = {(r["method"], r["metric"]): float(r["mean"]) for r in rows}
    assert values["identity", "rel_sol"] == pytest.approx(0.017342896431963645)
    assert values["separable", "rel_sol"] == pytest.approx(0.011428995556500848)


def test_repeat_build_and_saved_tables(built, tmp_path):
    first, _ = built
    DiskEvidenceBuilder(ROOT, tmp_path).build()
    for source in (first / "tables").glob("*.csv"):
        assert source.read_bytes() == (tmp_path / "tables" / source.name).read_bytes()
        if source.name != "input_manifest.csv":
            assert source.read_bytes() == (PAPER / "tables" / source.name).read_bytes()
    assert json.loads((PAPER / "verification.json").read_text())["status"] == "complete"


def test_no_inference_imports():
    text = (PAPER / "build_tables.py").read_text()
    for forbidden in ("import torch", "import plotly", "predict_batch(", "subprocess"):
        assert forbidden not in text


def test_document_links_and_semantics():
    expected = {
        "writing_brief.md",
        "experiment_setup.md",
        "motivation_and_preconditioning.md",
        "results_and_interpretation.md",
        "figures_and_tables.md",
        "provenance.md",
        "acceptance.md",
    }
    assert expected <= {p.name for p in PAPER.glob("*.md")}
    for path in PAPER.glob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("https://", "http://", "#")):
                assert (path.parent / target.split("#")[0]).exists(), (
                    path.name,
                    target,
                )
    provenance = (PAPER / "provenance.md").read_text()
    for term in (
        "ddof=1",
        "linear",
        "ratio_of_means",
        "mean_of_sample_ratios",
        "NA",
        "소급",
    ):
        assert term in provenance
    motivation = (PAPER / "motivation_and_preconditioning.md").read_text()
    assert "cross term을 제외하지만" in motivation
    assert "row norm" in motivation
    assert "normalization" not in (PAPER / "writing_brief.md").read_text()


def test_activity_timing_candidates(built):
    out, _ = built

    def rows(name):
        with (out / "tables" / name).open() as handle:
            return list(csv.DictReader(handle))

    for r in rows("optimizer_timing.csv"):
        assert r["scope"] == "optimizer_only_not_training_wall_time"
        assert int(r["optimizer_calls"]) == 2400
        if r["device"] == "cpu":
            assert r["cuda_peak_allocated_mib"] == ""
    candidates = rows("figure_candidates.csv")
    assert [int(r["sample_id"]) for r in candidates] == [2, 27]
    assert all(
        r["status"] == "future_export_required_no_substitution" for r in candidates
    )
    for r in rows("reference_free_diagnostics.csv"):
        assert (
            float(r["direction0_active_fraction"])
            == float(r["direction1_active_fraction"])
            == 1
        )


def test_current_manifest_integrity(built):
    import hashlib

    out, _ = built
    with (out / "tables/input_manifest.csv").open() as handle:
        manifest = list(csv.DictReader(handle))
    assert sum(r["role"] == "current_test_source_reference" for r in manifest) == 50
    for row in manifest:
        assert (
            hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )


def test_saved_context_pairing(built):
    out, _ = built
    with (out / "tables/saved_context_provenance.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for seed in range(4):
        a, b = [r for r in rows if int(r["seed"]) == seed]
        assert a["context_id"] == b["context_id"]
        assert a["tensor_payload_sha256"] == b["tensor_payload_sha256"]
    assert len({r["green_state_dict_sha256"] for r in rows}) == 1
    assert len({r["geometry_semantic_sha256"] for r in rows}) == 1


def test_failed_build_cannot_leave_stale_complete(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "verification.json").write_text('{"status": "complete"}')
    with pytest.raises(FileNotFoundError):
        DiskEvidenceBuilder(tmp_path / "missing_inputs", out).build()
    assert json.loads((out / "verification.json").read_text())["status"] != "complete"
