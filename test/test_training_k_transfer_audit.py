import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
spec = importlib.util.spec_from_file_location(
    "transfer", Path(__file__).resolve().parents[1] / "cli/audit_training_k_transfer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("target_k", [9, 10])
def test_native_validation_rejects_corruption(target_k):
    rows = pd.DataFrame(
        [
            dict(
                sample_id=0,
                K=k,
                rel_sol=0.1,
                rel_equal=0.2,
                energy=0.3,
                response_cost=1 / (k + 1),
            )
            for k in range(target_k + 1)
        ]
    )
    saved = pd.DataFrame(
        [
            dict(
                sample_id=0,
                rel_sol=0.1,
                rel_sol_equal_mean=0.2,
                loss_energy_optimized=0.3,
                tangent_response_cost_k4=0.2,
            )
        ]
    )
    assert module.validate_native(rows, saved, 4, target_k)["rel_sol"] == 0
    saved.loc[0, "rel_sol"] = 0.9
    with pytest.raises(AssertionError):
        module.validate_native(rows, saved, 4, target_k)
    assert np.isfinite(rows.response_cost).all()


def test_summary_requires_complete_paired_coverage(tmp_path):
    pd.DataFrame([dict(seed=0, training_k=4, K=10, sample_id=0)]).to_csv(
        tmp_path / "per_sample.csv", index=False
    )
    with pytest.raises(AssertionError):
        module.summarize(tmp_path)
