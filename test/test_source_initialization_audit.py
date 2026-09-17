from dataclasses import replace
from pathlib import Path

import pytest
import torch

from greenonet.complex_frozen_tangent_csv import (
    FrozenTangentCsvAudit,
    FrozenTangentCsvRequest,
)
from greenonet.complex_source_initialization_audit import (
    SourceInitializationRequest,
    SourceRun,
    SourceRunSession,
    operator_fingerprint,
)
from test.test_complex_frozen_tangent_csv import _make_runs


def test_request_restricts_scope(tmp_path: Path):
    args = dict(manifest=tmp_path / "manifest", outdir=tmp_path / "out")
    assert SourceInitializationRequest(**args).device == "cuda:1"
    for bad in (
        dict(device="cuda:0"),
        dict(device="cpu"),
        dict(max_k=65),
        dict(batch_size=0),
    ):
        with pytest.raises(ValueError):
            SourceInitializationRequest(**(args | bad))


@pytest.mark.parametrize("normalization", ["legacy", "response"])
def test_native_reference_equal_prefix_and_network_bypass(
    tmp_path: Path, normalization, monkeypatch
):
    runs = _make_runs(tmp_path, k=2, count=1, normalization=normalization)
    audit = FrozenTangentCsvAudit(
        FrozenTangentCsvRequest(
            run_dirs=runs,
            outdir=tmp_path / "out",
            device="cpu",
            baseline_k=2,
            max_k=4,
            batch_size=2,
        )
    )
    audit._initialize_output()
    spec = audit._preflight()[0]
    spec.fingerprint = operator_fingerprint(spec)
    altered = replace(
        spec, input_hashes=spec.input_hashes | {str(spec.green): "changed"}
    )
    assert operator_fingerprint(altered) != spec.fingerprint
    session = SourceRunSession(SourceRun("test", 2, audit, spec, {}))
    try:
        assert session.smoke()["equal_prediction_verified"]
        assert len(session.learned_rows()) == 2
        originals = [batch.flux_valid.clone() for batch in session.batches]
        refs = session.reference_rows()
        assert len(refs) == 4
        for batch, original in zip(session.batches, originals):
            assert torch.equal(batch.flux_valid, original)

        def forbidden(*args, **kwargs):
            raise AssertionError("equal split must bypass network")

        monkeypatch.setattr(session.model, "forward_with_fusion_diagnostics", forbidden)
        rows = session.equal_rows()
        assert len(rows) == 10
        session.validate_independent({0, 1, 2, 4}, rows)
        with torch.no_grad():
            for k in (0, 1, 2, 4):
                assert torch.isfinite(
                    session.equal_prediction(session.batches[0], k).u_pred_valid
                ).all()
    finally:
        session.close()


def test_independent_timing_recomputes_every_batch(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    import logging
    from greenonet.complex_source_initialization_audit import SourceInitializationAudit

    calls = []
    audit = SourceInitializationAudit(
        SourceInitializationRequest(
            manifest=tmp_path / "unused",
            outdir=tmp_path,
            warmup_repeats=1,
            timing_repeats=2,
        )
    )
    monkeypatch.setattr(audit, "_gpu_idle", lambda: None)

    class Clock:
        def __init__(self, device):
            pass

        def measure(self, call):
            call()
            return dict(seconds=1.0, start_allocated_mib=0.0, peak_allocated_mib=1.0)

    monkeypatch.setattr(
        "greenonet.complex_source_initialization_audit.FrozenTangentBenchmark", Clock
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    session = SimpleNamespace(
        device=torch.device("cpu"),
        run=SimpleNamespace(native_k=2, example="x"),
        spec=SimpleNamespace(fingerprint="f", run_id="s0"),
        batches=[0, 1],
        dataset=[0, 1],
        audit=SimpleNamespace(logger=logging.getLogger("test")),
        _prediction_forward=lambda batch, k: calls.append(("learned", k, batch)),
        equal_prediction=lambda batch, k: calls.append(("equal", k, batch)),
    )
    audit._benchmark(session, {0, 4}, {"s0": object()})
    assert len(calls) == 3 * 3 * 2
    assert len(audit.timings) == 2 * 3
    assert calls[:2] == [("equal", 4, 0), ("equal", 4, 1)]


def test_coverage_rejects_missing_and_duplicate_rows():
    from greenonet.source_initialization_report import verify_coverage

    runs = [dict(run_id="a", fingerprint="f", native_k=2, sample_count=1)]
    rows = [
        dict(
            run_id="shared",
            fingerprint="f",
            condition=condition,
            evaluation_k=0,
            sample_id=0,
        )
        for condition in ("reference_raw", "reference_balanced")
    ]
    assert (
        verify_coverage(rows, [], [], runs, "reference", 64, 5)["actual_sample_rows"]
        == 2
    )
    for wrong in (rows[:1], rows + rows):
        with pytest.raises(ValueError, match="coverage"):
            verify_coverage(wrong, [], [], runs, "reference", 64, 5)


def test_branch_identity_rebuild_is_explicit_and_other_identity_stays_strict(
    tmp_path: Path,
):
    from greenonet.complex_tangent_context_io import TangentResponseContextStore

    runs = _make_runs(tmp_path, k=2, count=1)
    audit = FrozenTangentCsvAudit(
        FrozenTangentCsvRequest(
            run_dirs=runs, outdir=tmp_path / "out", device="cpu", baseline_k=2, max_k=4
        )
    )
    audit._initialize_output()
    spec = audit._preflight()[0]
    run = SourceRun("test", 2, audit, spec, {})
    original = SourceRunSession(run)
    assert audit.shared_cache is not None and audit.shared_cache.identity is not None
    identity = audit.shared_cache.identity
    path = runs[0] / "tangent_response_context.safetensors"
    TangentResponseContextStore.save(
        path=path,
        context=original.context,
        identity=replace(identity, x_green_branch_sha256="sha256:historical_platform"),
    )
    snapshot = path.read_bytes()
    audit.shared_cache = None
    rebuilt = SourceRunSession(run)
    try:
        assert rebuilt.verification["runtime_context_validation"]["rebuilt"]
        assert (
            rebuilt.verification["runtime_context_validation"][
                "operator_max_abs_difference"
            ]
            == 0
        )
        assert len(rebuilt.learned_rows()) == 2
        assert path.read_bytes() == snapshot
    finally:
        rebuilt.close()
    TangentResponseContextStore.save(
        path=path,
        context=original.context,
        identity=replace(identity, geometry_semantic_sha256="sha256:wrong_geometry"),
    )
    audit.shared_cache = None
    with pytest.raises(ValueError, match="Non-branch"):
        SourceRunSession(run)
    original.close()
