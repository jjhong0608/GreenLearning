import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from docs.paper.numerical_examples.example_01_unit_square.build_tables import (
    gain,
    representative_samples,
    statistics,
    UnitSquareEvidenceBuilder,
    validate_grid,
    worst_ids,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper/numerical_examples/example_01_unit_square"
AUDIT = ROOT / "docs/analysis/unit_square_full_test_directional"


def test_statistics_definitions():
    values = list(range(1, 101))
    row = statistics(values)
    assert row["mean"] == row["median"] == 50.5
    assert row["p95"] == pytest.approx(np.quantile(values, 0.95, method="linear"))
    assert row["worst5_mean"] == 98
    assert row["maximum"] == 100
    assert gain(0, 0) is None
    assert gain(4, 1) == 0.75
    with pytest.raises(ValueError):
        statistics([np.nan])


def test_worst_ids_ties():
    rows = [dict(sample_id=i, error=2) for i in reversed(range(10))]
    assert worst_ids(rows, "error") == [0, 1, 2, 3, 4]


def test_grid_rejects_missing_duplicate_identity():
    rows = [
        dict(kind=k, seed=s, sample_id=i, file_stem=f"sample_{i:06d}")
        for k in ("off", "on", "wide_off")
        for s in range(4)
        for i in range(100)
    ]
    validate_grid(rows)
    for bad in [rows[:-1], rows[:-1] + [rows[0]]]:
        with pytest.raises(ValueError):
            validate_grid(bad)
    rows[0]["file_stem"] = "wrong"
    with pytest.raises(ValueError):
        validate_grid(rows)


def test_builder_has_no_inference():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/paper/numerical_examples/example_01_unit_square/build_tables.py"
    )
    text = path.read_text()
    for forbidden in ("import torch", "import plotly", "predict_batch(", "subprocess"):
        assert forbidden not in text


def test_representative_tie():
    rows = [
        dict(sample_id=i, rel_sol=v)
        for i, v in [(77, 0.0158630597720228), (72, 0.015547804037680328)]
    ]
    assert representative_samples(rows)[0]["sample_id"] == 72


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("unit_square_paper")
    result = UnitSquareEvidenceBuilder(ROOT, out).build()
    return out, result


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_real_reconciliation_and_context(built):
    out, result = built
    assert result["status"] == "complete"
    assert (result["runs"], result["samples"], result["unique_test_sources"]) == (
        12,
        1200,
        100,
    )
    assert result["summary_max_abs_difference"] < 1e-10
    assert result["input_files_unchanged"]
    assert not result["inference_executed"] and not result["training_executed"]
    assert not result["figures_generated"]
    audit = json.loads((AUDIT / "verification.json").read_text())
    assert audit["device"] == "cpu" and audit["dtype"] == "float64"
    for run in audit["runs"]:
        assert run["samples"] == 100
        assert run["context"]["load_count"] + run["context"]["build_count"] == 1
        assert run["context"]["save_count"] == 0
        assert run["max_baseline_abs_error"] < 1e-10
    validate_grid(read_rows(out / "tables/directional_per_sample.csv"))


def test_repeat_build_and_saved_numeric_tables(built):
    out, _ = built
    before = {p.name: p.read_bytes() for p in (out / "tables").glob("*.csv")}
    check = (out / "verification.json").read_bytes()
    UnitSquareEvidenceBuilder(ROOT, out).build()
    assert check == (out / "verification.json").read_bytes()
    for name, content in before.items():
        assert (out / "tables" / name).read_bytes() == content
        if name != "input_manifest.csv":
            assert (PAPER / "tables" / name).read_bytes() == content


def test_tail_scopes_and_paired_identity(built):
    out, _ = built
    rows = read_rows(out / "tables/directional_per_sample.csv")
    summary = read_rows(out / "tables/accuracy_and_tails_summary.csv")
    for model in ("off", "on", "wide_off"):
        values = [
            [
                float(r["rel_sol"])
                for r in rows
                if r["kind"] == model and int(r["seed"]) == s
            ]
            for s in range(4)
        ]
        item = next(
            r
            for r in summary
            if r["kind"] == model
            and r["metric"] == "rel_sol"
            and r["statistic"] == "maximum"
        )
        assert float(item["seed_mean"]) == pytest.approx(
            np.mean([max(x) for x in values])
        )
        assert float(item["observed_global_max"]) == max(max(x) for x in values)
        assert float(item["seed_sd"]) == pytest.approx(
            np.std([max(x) for x in values], ddof=1)
        )
    fixed = read_rows(out / "tables/baseline_fixed_worst5.csv")
    for r in fixed:
        subset = [
            x for x in rows if x["kind"] == r["baseline"] and x["seed"] == r["seed"]
        ]
        assert list(map(int, r["sample_ids"].split(";"))) == worst_ids(
            subset, r["metric"]
        )
    candidates = read_rows(out / "tables/figure_candidates.csv")
    assert {r["sample_id"] for r in candidates} == {"72", "88"}


def test_documents_links_and_reported_means(built):
    out, _ = built
    for stem in (
        "writing_brief",
        "experiment_setup",
        "motivation_and_architecture",
        "results_and_interpretation",
        "figures_and_tables",
        "provenance",
        "acceptance",
    ):
        doc = PAPER / f"{stem}.md"
        assert len(doc.read_text()) > 300
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if not target.startswith(("https://", "http://", "#")):
                assert (doc.parent / target.split("#")[0]).exists(), target
    summary = read_rows(out / "tables/accuracy_and_tails_summary.csv")
    for line in (PAPER / "results_and_interpretation.md").read_text().splitlines():
        if line.startswith("| ") and " +/- " in line:
            columns = [c.strip() for c in line.strip("|").split("|")]
            kind = {"256/off": "off", "256/on": "on", "428/off": "wide_off"}[columns[0]]
            for metric, text in zip(
                ("rel_u_phi", "rel_u_psi", "rel_sol_equal_mean", "rel_sol"),
                columns[1:],
                strict=True,
            ):
                mean, sd = map(float, text.split(" +/- "))
                row = next(
                    r
                    for r in summary
                    if r["kind"] == kind
                    and r["metric"] == metric
                    and r["statistic"] == "mean"
                )
                assert mean == pytest.approx(100 * float(row["seed_mean"]), abs=5.1e-7)
                assert sd == pytest.approx(100 * float(row["seed_sd"]), abs=5.1e-7)
