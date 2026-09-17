import json
from pathlib import Path

import numpy as np
import pytest

from greenonet.unit_square_trunk_audit import (
    TrunkAuditRequest,
    check_baseline,
    relative_error,
    representative_samples,
    UnitSquareTrunkAudit,
    validate_samples,
)


def test_relative_error_and_batch_independence():
    target = np.arange(1, 13, dtype=float).reshape(3, 4)
    pred = target * np.array([1.1, 1.2, 1.3])[:, None]
    whole = [relative_error(p, t) for p, t in zip(pred, target, strict=True)]
    split = [relative_error(pred[i], target[i]) for i in range(3)]
    np.testing.assert_allclose(whole, [0.1, 0.2, 0.3])
    assert whole == split
    assert relative_error(np.zeros(4), np.zeros(4)) == 0
    with pytest.raises(ValueError, match="finite"):
        relative_error(np.array([np.nan]), np.ones(1))


def test_sample_contract_rejects_duplicate_missing_and_bad_identity():
    rows = [dict(sample_id=i, file_stem=f"sample_{i:06d}") for i in range(100)]
    validate_samples(rows)
    for invalid in [rows[:-1], rows[:-1] + [rows[0]], rows[::-1]]:
        with pytest.raises(ValueError, match="Sample"):
            validate_samples(invalid)


def test_baseline_strict_finite_tolerance():
    check_baseline({"rel_sol": 0.1 + 1e-12}, {"rel_sol": "0.1"}, "test")
    for val in [0.11, np.nan, np.inf]:
        with pytest.raises((AssertionError, ValueError)):
            check_baseline({"rel_sol": val}, {"rel_sol": "0.1"}, "test")


def test_request_validation(tmp_path: Path):
    for kwargs in [dict(batch_size=0), dict(num_threads=0), dict(device="cuda:0")]:
        with pytest.raises(ValueError):
            TrunkAuditRequest(tmp_path, tmp_path / "out", **kwargs)


def test_representative_machine_tie():
    rows = [
        dict(sample_id=i, rel_sol=v)
        for i, v in [(77, 0.0158630597720228), (72, 0.015547804037680328)]
    ]
    median, worst = representative_samples(rows)
    assert median["sample_id"] == 72
    assert worst["sample_id"] == 77


def test_output_and_checkpoint_protection(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    out = tmp_path / "audit"
    out.mkdir()
    (out / "existing").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        UnitSquareTrunkAudit(TrunkAuditRequest(root, out))._preflight()
    with pytest.raises(ValueError, match="outside"):
        UnitSquareTrunkAudit(TrunkAuditRequest(root, root / "audit"))._preflight()
    run = root / "unit_square_trunk_off_seed0"
    (run / "artifacts_best_energy").mkdir(parents=True)
    (run / "config_used.json").write_text(
        json.dumps(
            dict(
                coupling_training=dict(seed=0),
                dataset={},
                coupling_model=dict(
                    balance_projection=dict(
                        mode="symmetric_tangent_green_response",
                        symmetric_tangent_green_response=dict(subspace_dimension=2),
                    )
                ),
            )
        )
    )
    (run / "artifacts_best_energy/summary.json").write_text(
        json.dumps(dict(coupling_checkpoint="final.safetensors"))
    )
    audit = UnitSquareTrunkAudit(TrunkAuditRequest(root, tmp_path / "fresh"))
    monkeypatch.setattr(audit, "_record", lambda path: None)
    monkeypatch.setattr(
        "greenonet.unit_square_trunk_audit.read_csv",
        lambda path: [
            dict(sample_id=i, file_stem=f"sample_{i:06d}") for i in range(100)
        ],
    )
    with pytest.raises(ValueError, match="not a best-energy"):
        audit._preflight()
