import json
import logging
import shutil
from pathlib import Path

import pytest

from cli.audit_paper_tangent_compatibility import PaperCompatibilityAudit
from greenonet.unit_square_trunk_audit import write_csv
from test.test_complex_projection_response_audit import _write_fixture


@pytest.mark.parametrize("trained_k,posthoc_k", [(1, None), (2, None), (2, 4)])
def test_fixture_paired_frozen_compatibility(
    tmp_path, monkeypatch, trained_k, posthoc_k
):
    config, checkpoint, green, geometry, test, coefficients = _write_fixture(tmp_path)
    monkeypatch.setattr("cli.audit_paper_tangent_compatibility.local_path", Path)
    payload = json.loads(config.read_text())
    payload["pipeline"] = {"green_pretrained_path": str(green)}
    payload["dataset"].update(
        geometry_path=str(geometry),
        test_path=str(test),
        coefficient_functions_path=str(coefficients),
    )
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "subspace_dimension": trained_k,
            "eta_cap_enabled": False,
            "eta_strategy": "closed_loop_exact_line_search",
        },
    }
    config.write_text(json.dumps(payload))
    best = tmp_path / "complex_coupling_model_best_energy.safetensors"
    shutil.copyfile(checkpoint, best)
    metrics = tmp_path / "artifacts_best_energy/metrics"
    metrics.mkdir(parents=True)
    write_csv(
        metrics / "per_sample_metrics.csv",
        [{"file_stem": "sample_0000", "rel_sol": 0.0, "rel_sol_equal_mean": 0.0}],
    )
    before = best.read_bytes()
    audit = PaperCompatibilityAudit(tmp_path / "audit", logging.getLogger("test"))
    audit.posthoc_k = posthoc_k
    result = audit.run_one(config, "example1")
    assert result["samples"] == 1
    assert result["trained_k"] == trained_k
    assert result["k"] == (posthoc_k or trained_k)
    assert result["baseline_is_same_k"] is (posthoc_k is None)
    assert result["balance_max_abs"] < 1e-12
    assert result["relative_error_max_abs"] < 1e-8
    assert best.read_bytes() == before
