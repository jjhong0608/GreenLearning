from pathlib import Path

import pytest
import torch

from cli.audit_learned_k_extension import LearnedExtensionSession, minimum_rows
from greenonet.complex_frozen_tangent_csv import (
    FrozenTangentCsvAudit,
    FrozenTangentCsvRequest,
)
from greenonet.complex_source_initialization_audit import (
    SourceRun,
    operator_fingerprint,
)
from greenonet.source_initialization_metrics import summarize
from test.test_complex_frozen_tangent_csv import _make_runs


def test_minimum_is_statistic_specific_and_ties_choose_smallest_k():
    rows = [
        dict(evaluation_k=k, rel_sol_mean=m, rel_sol_p95=p, rel_sol_max=x)
        for k, m, p, x in [(0, 3, 4, 5), (1, 1, 3, 6), (2, 1, 2, 4)]
    ]
    assert [r["evaluation_k"] for r in minimum_rows(rows)] == [1, 2, 2]
    with pytest.raises(ValueError):
        minimum_rows(rows[1:])


@pytest.mark.parametrize("normalization", ["legacy", "response"])
def test_learned_curve_native_and_independent(tmp_path: Path, normalization: str):
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
    session = LearnedExtensionSession(SourceRun("test", 2, audit, spec, {}))
    try:
        with torch.no_grad():
            native = session.learned_rows()
            rows = session.curve()
            assert len(rows) == 5 * len(native)
            assert len(summarize(rows)) == 5
            session.verify_curve(rows, {0, 1, 2, 4})
    finally:
        session.close()
