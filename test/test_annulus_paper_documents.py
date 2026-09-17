from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from docs.paper.numerical_examples.example_03_annulus.build_tables import (
    AnnulusEvidenceBuilder,
    gain,
    representative,
    stats,
    validate_grid,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper/numerical_examples/example_03_annulus"


def test_units_ratios_and_na() -> None:
    assert gain(0, 0) is None
    assert gain(1e-14, 0) is None
    assert gain(4, 1) == 0.75
    assert stats([])["mean"] is None
    assert stats([1, 2, 3])["median"] == 2
    assert gain(5, 3) != np.mean([gain(1, 1), gain(9, 5)])
    with pytest.raises(ValueError, match="Non-finite"):
        stats([float("nan")])


def test_representative_uses_equal_and_smaller_id_on_tie() -> None:
    rows = [
        dict(seed=0, sample_id=i, equal=v, weak=100 - i)
        for i, v in ((4, 2), (3, 4), (2, 8), (1, 10))
    ]
    median, worst = representative(rows)
    assert median["sample_id"] == 2
    assert worst["sample_id"] == 1


def test_missing_duplicate_and_identity_are_rejected() -> None:
    rows = [
        dict(seed=s, sample_id=i, file_stem=f"sample_{i:06d}")
        for s in range(4)
        for i in range(100)
    ]
    validate_grid(rows)
    with pytest.raises(ValueError, match="coverage"):
        validate_grid(rows[:-1])
    with pytest.raises(ValueError, match="coverage"):
        validate_grid(rows[:-1] + [rows[0]])
    rows[0]["file_stem"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate_grid(rows)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("annulus_paper")
    before = hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest()
    result = AnnulusEvidenceBuilder(ROOT, out).build()
    assert hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest() == before
    return out, result


def test_real_reconciliation(built: tuple[Path, dict]) -> None:
    _, result = built
    assert result["status"] == "complete"
    assert (result["global_rows"], result["regional_rows"], result["weight_rows"]) == (
        400,
        6000,
        6000,
    )
    assert result["original_audit_hashes_verified"] == 3815
    assert result["summary_max_abs_difference"] < 1e-14
    assert result["input_files_unchanged"] and result["plan_unchanged"]
    assert not any(
        result[k]
        for k in ("training_executed", "inference_executed", "figures_generated")
    )


def test_repeated_build_is_deterministic(built: tuple[Path, dict]) -> None:
    out, _ = built
    paths = [*sorted((out / "tables").glob("*.csv")), out / "verification.json"]
    before = {p.relative_to(out): p.read_bytes() for p in paths}
    AnnulusEvidenceBuilder(ROOT, out).build()
    for name, content in before.items():
        assert (out / name).read_bytes() == content
        if name == Path("tables/input_manifest.csv"):
            # PLAN is a live implementation plan, not frozen numerical evidence.
            with (out / name).open() as handle:
                current = {r["path"]: r for r in csv.DictReader(handle)}
            with (PAPER / name).open() as handle:
                saved = {r["path"]: r for r in csv.DictReader(handle)}
            assert (
                current.pop("PLAN.md")["sha256"]
                == hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest()
            )
            saved.pop("PLAN.md")
            for source in ("src/greenonet/complex_projection.py",):
                assert (
                    current.pop(source)["sha256"]
                    == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
                )
                saved.pop(source)
            assert current == saved
        else:
            assert (PAPER / name).read_bytes() == content


def test_claims_and_missing_figure(built: tuple[Path, dict]) -> None:
    out, _ = built

    def rows(name: str) -> list[dict]:
        with (out / "tables" / f"{name}.csv").open() as handle:
            return list(csv.DictReader(handle))

    candidates = rows("figure_candidates")
    assert {r["sample_id"] for r in candidates if r["role"] == "equal_median"} == {"1"}
    assert all(
        r["stored_field"] == "False" for r in candidates if r["role"] == "equal_median"
    )
    assert all(r["composite_figure_status"] == "not_generated" for r in candidates)
    regions = {r["region"]: r for r in rows("transition_comparison")}
    assert int(regions["transition"]["paired_wins"]) == 400
    assert int(regions["overlap"]["paired_wins"]) == 297
    assert len(rows("local_deterioration")) == 129
    accuracy = {r["metric"]: r for r in rows("global_accuracy")}
    paired = rows("global_paired_comparison")[0]
    assert int(paired["paired_wins"]) == 400
    assert float(paired["reduction_ratio_of_means"]) == pytest.approx(
        0.1211699, rel=1e-5
    )
    assert float(paired["mean_sample_reduction"]) == pytest.approx(0.119841, rel=1e-5)
    assert float(accuracy["weak"]["mean_percent"]) == pytest.approx(1.0006962790462157)
    assert float(accuracy["equal"]["seed_sd_percentage_points"]) == pytest.approx(
        0.0084428, rel=1e-4
    )


def test_no_inference_imports_or_execution() -> None:
    text = (PAPER / "build_tables.py").read_text()
    for forbidden in (
        "import torch",
        "import plotly",
        "subprocess",
        "load_model(",
        "predict_batch(",
    ):
        assert forbidden not in text


def test_documents_and_links() -> None:
    for name in (
        "writing_brief",
        "experiment_setup",
        "motivation_and_reconstruction",
        "results_and_interpretation",
        "figures_and_tables",
        "provenance",
    ):
        path = PAPER / f"{name}.md"
        text = path.read_text()
        assert "TODO" not in text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http", "#")):
                continue
            assert (path.parent / target.split("#")[0]).resolve().exists(), target
    combined = "\n".join(p.read_text() for p in PAPER.glob("*.md"))
    for token in (
        "미입증",
        "100개",
        "103/400",
        "sample_000001",
        "sample_000034",
        "ddof=1",
        "trace-jump",
    ):
        assert token in combined
