"""Static config validation only: no model execution or experiment launch."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cli.train import TrainCLI
from greenonet.complex_coupling_model import ComplexCouplingNet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "numerical_examples/unit_square"
CASES = [(seed, mode) for seed in range(4) for mode in ("off", "on")]


def config_path(seed: int, mode: str) -> Path:
    hardware = "nvidia_a40" if seed in (0, 2) else "mac_studio"
    return CONFIGS / hardware / f"seed{seed}/unit_square_trunk_{mode}_seed{seed}.json"


def test_exact_eight_configs_and_paired_differences() -> None:
    expected = {config_path(*case) for case in CASES}
    assert set(CONFIGS.glob("*/*/unit_square_trunk_*.json")) == expected
    baseline = None
    for seed, mode in CASES:
        raw = json.loads(config_path(seed, mode).read_text())
        normalized = copy.deepcopy(raw)
        training = normalized["coupling_training"]
        assert training["seed"] == seed
        assert training["device"] == ("cuda:1" if seed in (0, 2) else "cpu")
        trunk = normalized["coupling_model"]["axis_1d_trunk"]["transverse_trunk"]
        assert trunk["enabled"] is (mode == "on")
        training["seed"], training["device"], trunk["enabled"] = 0, "cpu", False
        if baseline is None:
            baseline = normalized
        assert normalized == baseline


@pytest.mark.parametrize(("seed", "mode"), CASES)
def test_static_parser_and_experiment_contract(seed: int, mode: str) -> None:
    path = config_path(seed, mode)
    dataset, _, _, model, training, pipeline, _ = TrainCLI()._build_configs(path)
    ComplexCouplingNet._validate_complex_config(model)
    raw = json.loads(path.read_text())
    m, t = raw["coupling_model"], raw["coupling_training"]
    assert dataset.geometry_mode == "complex"
    assert all(v is False for v in m["coefficient_terms"].values())
    assert m["branch_fusion"]["mode"] == "concat_fuser"
    assert not m["geometry_branch"]["enabled"]
    assert not m["axis_1d_trunk"]["fixed_line_transverse_branch"]["enabled"]
    assert not m["pre_projection_fusion"]["enabled"]
    assert m["axis_1d_trunk"]["transverse_trunk"]["fusion"] == "concat_fuser"
    tangent = m["balance_projection"]["symmetric_tangent_green_response"]
    assert tangent["subspace_dimension"] == tangent["max_subspace_dimension"] == 2
    assert tangent["preconditioner_variant"] == "separable"
    assert not tangent["geometry_k_selection"]["enabled"]
    assert not tangent["eta_cap_enabled"]
    assert tangent["relative_lambda"] == 0.01
    gp = raw["dataset"]["coupling_source"]["indexed_gp"]
    assert gp == dict(
        num_train=4800, num_valid=300, seed=0, lengthscale=0.15, amplitude=1, mean=0
    )
    assert training.epochs * (gp["num_train"] // training.batch_size) == 2400
    assert t["warmup_steps"] == 240 and "warmup_epochs" not in t
    assert t["validation_every_steps"] == 24
    assert t["canonical_energy"]["boundary_weight"] == 0
    assert not t["post_line_search_stationarity"]["enabled"]
    assert not t["response_trust"]["enabled"]
    assert t["best_energy_checkpoint"]["enabled"]
    assert not t["best_physics_checkpoint"]["enabled"]
    assert t["tangent_context_checkpoint"] == dict(
        enabled=True, path=None, load_policy="if_available", save_after_build=True
    )
    assert t["optimizer"]["name"] == "soap"
    assert t["optimizer"]["betas"] == [0.95, 0.99]
    assert m["cross_axis_reconstruction"]["mode"] == "local_weak_residual_reliability"
    assert raw["dataset"]["reference_diagnostics"] == dict(
        training=False, validation=False
    )
    assert raw["dataset"]["dtype"] == m["dtype"] == "float64"
    artifacts = raw["coupling_artifacts"]
    assert artifacts["enabled"] and artifacts["checkpoint"] == "best_energy"
    assert artifacts["device"] is None and artifacts["save_generated_data"]
    assert pipeline.run_coupling and not pipeline.run_green
    paths = [
        raw["dataset"][k]
        for k in ("geometry_path", "test_path", "coefficient_functions_path")
    ]
    paths += [raw["pipeline"]["green_pretrained_path"], artifacts["visualization_mesh"]]
    for value in paths:
        assert not Path(value).is_absolute()
        assert (ROOT / value).exists()
    assert (
        raw["pipeline"]["green_pretrained_path"]
        == "checkpoints/poisson_unit_square/green/model.safetensors"
    )


def test_poisson_coefficients_preserve_previous_definition() -> None:
    assert (CONFIGS / "coefficients.py").read_bytes() == (
        ROOT / "numerical_examples/unit_square_old/coefficients.py"
    ).read_bytes()
