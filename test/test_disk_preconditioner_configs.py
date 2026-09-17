"""Static paired config checks; never launch the eight training experiments."""

import copy
import json
from pathlib import Path

import pytest

from cli.train import TrainCLI
from greenonet.complex_coupling_model import ComplexCouplingNet

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "numerical_examples/disk"
CASES = [(seed, variant) for seed in range(4) for variant in ("identity", "separable")]


def config_path(seed, variant):
    hardware = "nvidia_a40" if seed in (0, 2) else "mac_studio"
    return DIRECTORY / hardware / f"seed{seed}/disk_{variant}_seed{seed}.json"


def test_eight_configs_only_preconditioner_seed_device_differ():
    assert set(DIRECTORY.rglob("*.json")) == {config_path(*case) for case in CASES}
    baseline = None
    for seed, variant in CASES:
        raw = json.loads(config_path(seed, variant).read_text())
        normalized = copy.deepcopy(raw)
        training = normalized["coupling_training"]
        assert training["seed"] == seed
        assert training["device"] == ("cuda:1" if seed in (0, 2) else "cpu")
        tangent = normalized["coupling_model"]["balance_projection"][
            "symmetric_tangent_green_response"
        ]
        assert tangent["preconditioner_variant"] == variant
        training["seed"], training["device"] = 0, "cpu"
        tangent["preconditioner_variant"] = "separable"
        if baseline is None:
            baseline = normalized
        assert normalized == baseline


@pytest.mark.parametrize("seed,variant", CASES)
def test_parser_and_common_experiment_contract(seed, variant):
    path = config_path(seed, variant)
    dataset, _, _, model, training, pipeline, _ = TrainCLI()._build_configs(path)
    ComplexCouplingNet._validate_complex_config(model)
    raw = json.loads(path.read_text())
    m, t = raw["coupling_model"], raw["coupling_training"]
    tangent = model.balance_projection.symmetric_tangent_green_response
    assert tangent.direction_normalization == "response"
    assert (
        tangent.direction_independence_relative_eps
        == tangent.line_search_relative_eps
        == 1e-12
    )
    assert tangent.preconditioner_variant == variant
    assert tangent.subspace_dimension == tangent.max_subspace_dimension == 2
    assert not tangent.geometry_k_selection.enabled
    assert not tangent.eta_cap_enabled
    assert tangent.eta_strategy == "closed_loop_exact_line_search"
    assert m["coefficient_terms"] == dict(
        diffusion=True, convection=False, reaction=False
    )
    assert m["branch_fusion"]["mode"] == "concat_fuser"
    assert m["axis_1d_trunk"]["transverse_trunk"] == dict(
        enabled=True, fusion="concat_fuser", length_context=True
    )
    assert not m["geometry_branch"]["enabled"]
    assert not m["axis_1d_trunk"]["fixed_line_transverse_branch"]["enabled"]
    assert not m["pre_projection_fusion"]["enabled"]
    gp = raw["dataset"]["coupling_source"]["indexed_gp"]
    assert gp == dict(
        num_train=4800, num_valid=300, seed=0, lengthscale=0.15, amplitude=1, mean=0
    )
    assert training.epochs * (gp["num_train"] // training.batch_size) == 2400
    assert t["warmup_steps"] == 240 and t["validation_every_steps"] == 24
    assert t["canonical_energy"]["boundary_weight"] == 0
    assert not t["response_trust"]["enabled"]
    assert not t["post_line_search_stationarity"]["enabled"]
    assert t["optimizer"]["name"] == "soap"
    assert t["best_energy_checkpoint"]["enabled"]
    assert t["tangent_context_checkpoint"]["enabled"]
    assert pipeline.run_coupling and not pipeline.run_green
    assert dataset.geometry_mode == "complex"
    artifacts = raw["coupling_artifacts"]
    assert artifacts["enabled"] and artifacts["checkpoint"] == "best_energy"
    paths = [
        raw["dataset"][key]
        for key in ("geometry_path", "test_path", "coefficient_functions_path")
    ]
    paths += [raw["pipeline"]["green_pretrained_path"], artifacts["visualization_mesh"]]
    for value in paths:
        assert not Path(value).is_absolute()
        assert (ROOT / value).exists()


def test_coefficient_definition_is_preserved():
    assert (DIRECTORY / "coefficients.py").read_bytes() == (
        ROOT / "numerical_examples/disk_old/coefficient.py"
    ).read_bytes()
