import numpy as np
import pytest
import json
import os
import subprocess
import sys
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from greenonet.complex_reconstruction_audit import (
    ReconstructionAuditRequest,
    ReconstructionAudit,
    region_masks,
    sample_metrics,
    summarize,
)


def test_partition_and_inclusive_band():
    xy = np.array([[0.3, 0.2], [0.2, 0.3], [0.2, 0.2], [0.4, 0.4], [0.3, 0.215625]])
    masks = region_masks(xy, 2)
    assert masks["transition"].tolist() == [True, True, True, False, True]
    partition = sum(
        masks[k].astype(int)
        for k in ["horizontal_only", "vertical_only", "overlap", "outside"]
    )
    assert np.all(partition == 1)


def test_perfect_and_ties():
    target = np.ones(4)
    predictions = {k: target.copy() for k in ["u_phi", "u_psi", "equal", "weak"]}
    global_row, regions, weights = sample_metrics(
        target, predictions, np.full(4, 0.5), np.array([[0.3, 0.2]] * 4)
    )
    assert global_row["weak"] == 0
    transition = next(
        r for r in regions if r["width_h"] == 2 and r["region"] == "transition"
    )
    assert transition["relative_reduction"] is None
    assert transition["absolute_change"] == 0
    assert weights[0]["tie_count"] == 4
    assert weights[0]["alignment_fraction"] is None
    empty = next(r for r in regions if r["region"] == "outside")
    assert empty["count"] == 0 and empty["weak"] is None


def test_alignment_and_cancellation():
    target = np.ones(2)
    predictions = dict(
        u_phi=np.array([1.0, 2.0]),
        u_psi=np.array([3.0, 0.0]),
        equal=np.array([2.0, 1.0]),
        weak=np.array([1.5, 1.5]),
    )
    g, _, w = sample_metrics(
        target, predictions, np.full(2, 0.75), np.array([[0.3, 0.2]] * 2)
    )
    assert g["equal"] == pytest.approx(1 / np.sqrt(2))
    assert g["weak"] == pytest.approx(0.5)
    assert w[0]["alignment_fraction"] == 1
    assert w[0]["tie_count"] == 1
    assert w[0]["opposite_error_fraction"] == 0.5


def test_seed_sd():
    rows = [dict(seed=s, metric=float(s + 1)) for s in range(4)]
    seed, aggregate = summarize(rows, ["metric"])
    assert len(seed) == 4
    assert aggregate[0]["seed_sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    summary, _ = summarize([dict(seed=0, improved=i) for i in [1, 0, 1]], ["improved"])
    assert summary[0]["paired_win_count"] == 2


def test_request_validation(tmp_path):
    with pytest.raises(ValueError):
        ReconstructionAuditRequest((), tmp_path, "cuda:0")
    with pytest.raises(ValueError):
        ReconstructionAuditRequest((tmp_path,), tmp_path, "cuda:1", batch_size=0)


def test_cli_rejects_gpu_zero(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "cli/audit_annulus_reconstruction.py",
            "--run-dirs",
            str(tmp_path),
            "--outdir",
            str(tmp_path / "out"),
            "--device",
            "cuda:0",
        ],
        env=os.environ | {"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "out").exists()


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        sample_metrics(
            np.array([np.nan]),
            {k: np.zeros(1) for k in ["u_phi", "u_psi", "equal", "weak"]},
            np.ones(1),
            np.zeros((1, 2)),
        )


def test_refuse_existing_output(tmp_path):
    (tmp_path / "keep.txt").write_text("unchanged")
    with pytest.raises(ValueError, match="empty"):
        ReconstructionAudit(
            ReconstructionAuditRequest((tmp_path / "run",), tmp_path, "cpu")
        ).run()
    assert (tmp_path / "keep.txt").read_text() == "unchanged"


def test_sequential_stops_on_failure(tmp_path, monkeypatch):
    seen = []
    audit = ReconstructionAudit(
        ReconstructionAuditRequest(
            tuple(tmp_path / str(i) for i in range(3)), tmp_path / "out", "cpu"
        )
    )
    monkeypatch.setattr(audit, "_preflight", lambda: None)

    def one(run, device):
        seen.append(run.name)
        if run.name == "1":
            raise ValueError("baseline mismatch")

    monkeypatch.setattr(audit, "_run_one", one)
    with pytest.raises(ValueError, match="baseline mismatch"):
        audit.run()
    assert seen == ["0", "1"]
    assert (
        json.loads((tmp_path / "out/verification.json").read_text())["status"]
        == "failed"
    )


@pytest.mark.parametrize("problem", ["checkpoint", "identities"])
def test_preflight_rejects_baseline_contract(tmp_path, monkeypatch, problem):
    from types import SimpleNamespace
    import csv
    import greenonet.complex_reconstruction_audit as audit_module

    run = tmp_path / "run"
    artifacts = run / "artifacts_best_energy"
    (artifacts / "metrics").mkdir(parents=True)
    (run / "config_used.json").write_text("{}")
    (artifacts / "summary.json").write_text(
        json.dumps(
            dict(
                checkpoint_selector="final"
                if problem == "checkpoint"
                else "best_energy",
                coupling_checkpoint=audit_module.BEST,
            )
        )
    )
    with (artifacts / "metrics/per_sample_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "file_stem"])
        writer.writeheader()
        writer.writerows([dict(sample_id=0, file_stem="duplicate")] * 100)
    monkeypatch.setattr(
        audit_module, "load_coupling_artifact_configs", lambda _: SimpleNamespace()
    )
    audit = ReconstructionAudit(
        ReconstructionAuditRequest((run,), tmp_path / "out", "cpu")
    )
    with pytest.raises(ValueError, match="best-energy|identities"):
        audit._preflight()


def test_production_predictions_reference_free(tmp_path):
    from test.test_complex_projection_response_audit import _write_fixture
    from greenonet.coefficients import load_coefficient_functions
    from greenonet.complex_geometry import load_complex_geometry
    from greenonet.complex_coupling_data import (
        ComplexCouplingDataset,
        complex_coupling_collate_fn,
    )
    from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
    from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
    from greenonet.coupling_artifacts import (
        CouplingArtifactRequest,
        load_coupling_artifact_configs,
    )
    from greenonet.config import TangentContextCheckpointConfig

    config, checkpoint, green, geometry, samples, coefficient = _write_fixture(tmp_path)
    raw = json.loads(config.read_text())
    raw["coupling_model"]["balance_projection"] = dict(
        enabled=True,
        mode="symmetric_tangent_green_response",
        symmetric_tangent_green_response=dict(
            subspace_dimension=4,
            max_subspace_dimension=4,
            eta_strategy="closed_loop_exact_line_search",
            eta_cap_enabled=False,
        ),
    )
    raw["coupling_model"]["cross_axis_reconstruction"] = dict(
        enabled=True, mode="local_weak_residual_reliability"
    )
    config.write_text(json.dumps(raw))
    configs = load_coupling_artifact_configs(config)
    dataset = ComplexCouplingDataset(
        samples,
        load_complex_geometry(geometry, dtype=torch.float64),
        load_coefficient_functions(coefficient),
        branch_input_dim=4,
        dtype=torch.float64,
        coefficient_terms=configs.coupling_model.coefficient_terms,
        integration_rule=configs.coupling_training.integration_rule,
    )
    batch = next(
        iter(DataLoader(dataset, batch_size=1, collate_fn=complex_coupling_collate_fn))
    )
    loader = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=config,
            coupling_checkpoint=checkpoint,
            green_checkpoint=green,
            outdir=tmp_path / "unused",
        )
    )
    sidecar = tmp_path / "absent_context.safetensors"
    evaluator = ComplexCouplingEvaluator(
        model=loader._load_complex_model(configs, torch.device("cpu")),
        green_model=loader._load_green_model(configs, torch.device("cpu")),
        config=replace(
            configs.coupling_training,
            tangent_context_checkpoint=TangentContextCheckpointConfig(
                enabled=True, load_policy="if_available", save_after_build=False
            ),
        ),
        device=torch.device("cpu"),
        work_dir=tmp_path / "audit",
        tangent_context_path=sidecar,
    )
    with torch.no_grad():
        original = evaluator.predict_batch(batch)
        changed = evaluator.predict_batch(
            replace(
                batch, sol_valid=batch.sol_valid + 17, flux_valid=batch.flux_valid - 9
            )
        )
    for attr in ["u_pred_valid", "u_equal_mean_valid"]:
        torch.testing.assert_close(
            getattr(original.cross_axis_reconstruction, attr),
            getattr(changed.cross_axis_reconstruction, attr),
            rtol=0,
            atol=0,
        )
    assert original.cross_axis_reconstruction.reliability is not None
    assert changed.cross_axis_reconstruction.reliability is not None
    torch.testing.assert_close(
        original.cross_axis_reconstruction.reliability.w_phi,
        changed.cross_axis_reconstruction.reliability.w_phi,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.projection.projected_physical,
        changed.projection.projected_physical,
        rtol=0,
        atol=0,
    )
    assert evaluator.symmetric_tangent_green_response_context_build_count == 1
    assert not sidecar.exists()
