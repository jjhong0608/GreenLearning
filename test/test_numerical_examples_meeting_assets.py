"""Focused tests for the new meeting evidence and asset contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DECK = Path(__file__).resolve().parents[1] / "docs/meeting/numerical_examples"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, DECK / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_verification_rejects_corruption(tmp_path):
    module = load_module("prepare_fields")
    preparation = module.FieldPreparation(tmp_path)
    fields = dict(sol=np.ones(3), weak=np.full(3, 1.1), equal=np.full(3, 1.2))
    preparation.verify_field(fields, dict(rel_sol="0.1", rel_sol_equal_mean="0.2"))
    fields["weak"][0] += 0.1
    with pytest.raises(AssertionError):
        preparation.verify_field(fields, dict(rel_sol="0.1", rel_sol_equal_mean="0.2"))


def test_field_selection_contract():
    module = load_module("prepare_fields")
    assert [(run.label, run.samples) for run in module.RUNS] == [
        ("disk_identity", (2, 27)),
        ("disk_separable", (2, 27)),
        ("annulus", (1, 34)),
    ]


def test_selected_field_reconciliation_and_pairing():
    verification = json.loads((DECK / "data/field_verification.json").read_text())
    assert len(verification["samples"]) == 6
    assert (
        sum(r["origin"] == "frozen_cpu_float64_replay" for r in verification["samples"])
        == 4
    )
    for r in verification["samples"]:
        assert r["rel_sol_abs_difference"] < 1e-10
        assert r["rel_sol_equal_mean_abs_difference"] < 1e-10
    for sample in (2, 27):
        with (
            np.load(DECK / f"data/disk_identity_{sample}.npz") as a,
            np.load(DECK / f"data/disk_separable_{sample}.npz") as b,
        ):
            for field in ("coords", "rhs", "sol"):
                np.testing.assert_array_equal(a[field], b[field])
    for sample in (1, 34):
        with np.load(DECK / f"data/annulus_{sample}.npz") as a:
            np.testing.assert_allclose(a["equal"], 0.5 * (a["u_phi"] + a["u_psi"]))
            np.testing.assert_allclose(
                a["weak"], a["w_phi"] * a["u_phi"] + (1 - a["w_phi"]) * a["u_psi"]
            )
            assert 0.25 <= a["w_phi"].min() <= a["w_phi"].max() <= 0.75


def test_manifest_stats_scales_and_immutable_sources():
    module = load_module("prepare_fields")
    manifest = json.loads((DECK / "assets/manifest.json").read_text())
    assert manifest["offline"] and not manifest["rendering_inference"]
    for name, digest in manifest["sources"].items():
        assert module.sha256(module.ROOT / name) == digest
    assets = manifest["assets"]
    assert assets["ex3_accuracy"]["categories"] == ["u_phi", "u_psi", "equal", "weak"]
    np.testing.assert_allclose(
        assets["ex3_accuracy"]["values"][0],
        [2.025867096470677, 2.04067566311509, 1.1386686029903275, 1.0006962790463374],
    )
    assert (
        assets["ex4_sample15_early"]["group_limits"]
        == assets["ex4_sample15_late"]["group_limits"]
    )
    for sample in (1, 34):
        assert assets[f"ex3_sample{sample}"]["group_limits"]["weight"] == [0.25, 0.75]
    for name, asset in assets.items():
        assert module.sha256(DECK / "assets" / asset["path"]) == asset["sha256"]
        if "sample" in name and "solution" not in name:
            assert asset["group_limits"]["error"][0] == 0
            assert "conforming triangle mesh" in asset["raster_policy"]
    e4 = module.rows(module.PAPER / "example_04_pentagram/tables/posthoc_by_k.csv")
    assert assets["ex4_posthoc"]["rows"][-1][0] == "64"
    assert float(assets["ex4_posthoc"]["rows"][-1][2]) == pytest.approx(
        100 * float(e4[-1]["rel_sol_mean"]), abs=5e-5
    )
    reach = assets["ex4_reach"]["rows"]
    assert reach[-2] == ["9", "100.0000", "100.0000", "100.0000"]
    assert reach[-1] == ["10", "100.0000", "100.0000", "100.0000"]


def test_mesh_payloads_and_artifact_reuse():
    load_module("prepare_fields")
    mesh_module = load_module("mesh_panels")
    manifest = json.loads((DECK / "assets/manifest.json").read_text())
    reused = 0
    for name, asset in manifest["assets"].items():
        if "mesh_panels" not in asset:
            continue
        payload = mesh_module.decode_arrays(
            json.loads((DECK / "assets" / (name + ".json")).read_text())
        )
        traces = [t for t in payload["data"] if t["type"] == "mesh3d"]
        assert len(traces) == len(asset["mesh_panels"])
        assert not any(t["type"] == "heatmap" for t in payload["data"])
        for trace, record in zip(traces, asset["mesh_panels"], strict=True):
            with np.load(DECK.parents[2] / record["mesh"]) as mesh:
                np.testing.assert_array_equal(
                    np.c_[trace["i"], trace["j"], trace["k"]], mesh["triangles"]
                )
                assert np.isfinite(trace["intensity"]).all()
                if "u_phi" in record["title"] or "u_psi" in record["title"]:
                    assert trace["intensitymode"] == "cell"
                    assert "unavailable" in record["boundary"]
            if record["origin"] == "reused_artifact":
                reused += 1
                original = mesh_module.decode_arrays(
                    json.loads((DECK.parents[2] / record["artifact"]).read_text())
                )
                np.testing.assert_array_equal(
                    trace["intensity"], original["data"][0]["intensity"]
                )
    assert reused > 0


def test_typical_example_uses_weak_max_without_clipping_values():
    load_module("prepare_fields")
    mesh_module = load_module("mesh_panels")
    with np.load(
        DECK.parents[2]
        / "docs/analysis/unit_square_full_test_directional/selected_fields.npz"
    ) as fields:
        limit = max(
            float(
                np.abs(
                    fields[f"{kind}_sample_000072_weak"]
                    - fields[f"{kind}_sample_000072_sol"]
                ).max()
            )
            for kind in ("off", "on", "wide_off")
        )
    payload = mesh_module.decode_arrays(
        json.loads((DECK / "assets/ex1_sample72.json").read_text())
    )
    meshes = [trace for trace in payload["data"] if trace["type"] == "mesh3d"]
    assert len(meshes) == 9
    assert all(trace["cmin"] == 0 and trace["cmax"] == limit for trace in meshes)
    assert any(np.max(trace["intensity"]) > limit for trace in meshes)


def test_all_error_slides_share_weak_based_limits():
    load_module("prepare_fields")
    mesh_module = load_module("mesh_panels")
    manifest = json.loads((DECK / "assets/manifest.json").read_text())
    for name, asset in manifest["assets"].items():
        if "sample" not in name or "solution" in name:
            continue
        payload = mesh_module.decode_arrays(
            json.loads((DECK / "assets" / f"{name}.json").read_text())
        )
        error_meshes = [
            t
            for t in payload["data"]
            if t["type"] == "mesh3d"
            and t["cmin"] == 0
            and t["cmax"] == asset["group_limits"]["error"][1]
        ]
        assert error_meshes
        assert all(t["colorscale"][0][1] == "rgb(255,255,204)" for t in error_meshes)
        assert "maximum weak error" in asset["error_scale_basis"]
    for sample in (2, 27):
        expected = 0.0
        for kind in ("identity", "separable"):
            with np.load(DECK / f"data/disk_{kind}_{sample}.npz") as data:
                expected = max(
                    expected, float(np.abs(data["weak"] - data["sol"]).max())
                )
        assert manifest["assets"][f"ex2_sample{sample}"]["group_limits"]["error"] == [
            0,
            expected,
        ]
    for sample in (1, 34):
        with np.load(DECK / f"data/annulus_{sample}.npz") as data:
            expected = float(np.abs(data["weak"] - data["sol"]).max())
        assert manifest["assets"][f"ex3_sample{sample}"]["group_limits"]["error"] == [
            0,
            expected,
        ]


def test_solution_panels_share_unclipped_limits():
    load_module("prepare_fields")
    mesh_module = load_module("mesh_panels")
    assets = json.loads((DECK / "assets/manifest.json").read_text())["assets"]
    solutions = {name: a for name, a in assets.items() if "solution_sample" in name}
    assert len(solutions) == 8
    for name, asset in solutions.items():
        payload = mesh_module.decode_arrays(
            json.loads((DECK / "assets" / f"{name}.json").read_text())
        )
        meshes = [t for t in payload["data"] if t["type"] == "mesh3d"]
        low, high = asset["group_limits"]["solution"]
        assert len(meshes) == (4 if name.startswith(("ex1", "ex4")) else 3)
        for trace in meshes:
            assert trace["cmin"] == low and trace["cmax"] == high
            assert (
                low <= np.min(trace["intensity"]) <= np.max(trace["intensity"]) <= high
            )
            assert trace["intensitymode"] == "vertex"
    assert (
        solutions["ex4_solution_sample15_early"]["group_limits"]
        == solutions["ex4_solution_sample15_late"]["group_limits"]
    )
