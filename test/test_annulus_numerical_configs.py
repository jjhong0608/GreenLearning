from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cli.train import TrainCLI


ROOT = Path(__file__).resolve().parents[1]
ANNULUS_DIR = ROOT / "numerical_examples" / "annulus"
SEEDS = (0, 1, 2, 3)


def test_annulus_green_config_preserves_soap_and_lbfgs_training_contract() -> None:
    config_path = ANNULUS_DIR / "annulus_green.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    with (ROOT / "configs" / "complex_green_soap.json").open(
        encoding="utf-8"
    ) as handle:
        expected = json.load(handle)
    expected["dataset"]["geometry_path"] = "data/geometry/annulus_02_05_1_128.npz"
    expected["dataset"]["coefficient_functions_path"] = (
        "coefficients/Smooth_Variable_Diffusion_Reaction.py"
    )
    expected["training"].pop("warmup_epochs")
    expected["training"]["warmup_steps"] = 20
    expected["training"]["validation_every_steps"] = 1
    assert config == expected

    dataset, model, training, _, _, pipeline, _ = TrainCLI()._build_configs(config_path)
    assert dataset.geometry_mode == "complex"
    assert model.branch_input_dim == 129
    assert training.seed == 0
    assert training.device == "cuda:1"
    assert pipeline.run_green is True
    assert pipeline.run_coupling is False
    assert pipeline.green_pretrained_path is None

    coupling_dataset = _load(0)["dataset"]
    for key in ("geometry_path", "coefficient_functions_path", "dtype"):
        assert config["dataset"][key] == coupling_dataset[key]
    assert "test_path" not in config["dataset"]
    assert "coupling_source" not in config["dataset"]
    assert "coupling_model" not in config
    assert "coupling_training" not in config
    assert "green_optimizer_provenance" not in config


def _load(seed: int) -> dict[str, Any]:
    path = ANNULUS_DIR / f"annulus_reconstruction_seed{seed}.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def test_annulus_has_one_shared_training_config_per_seed() -> None:
    expected = {
        ANNULUS_DIR / f"annulus_reconstruction_seed{seed}.json" for seed in SEEDS
    }
    assert set(ANNULUS_DIR.glob("annulus_reconstruction*.json")) == expected
    reference = copy.deepcopy(_load(0))
    for seed in SEEDS:
        payload = copy.deepcopy(_load(seed))
        assert payload["coupling_training"]["seed"] == seed
        payload["coupling_training"]["seed"] = 0
        assert payload == reference


@pytest.mark.parametrize("seed", SEEDS)
def test_annulus_uses_confirmed_diffusion_reaction_paths(seed: int) -> None:
    config = _load(seed)
    dataset = config["dataset"]
    assert dataset["coefficient_functions_path"] == (
        "coefficients/Smooth_Variable_Diffusion_Reaction.py"
    )
    assert dataset["test_path"] == (
        "data/complex_samples/annulus_02_05_1_128_reaction_diffusion/test"
    )
    assert config["pipeline"]["green_pretrained_path"] == (
        "checkpoints/numerical_examples/annulus/green/model.safetensors"
    )
    assert (ROOT / dataset["coefficient_functions_path"]).is_file()


@pytest.mark.parametrize("seed", SEEDS)
def test_annulus_preserves_fixed_k4_and_paper_training_protocol(seed: int) -> None:
    path = ANNULUS_DIR / f"annulus_reconstruction_seed{seed}.json"
    TrainCLI()._build_configs(path)
    config = _load(seed)
    dataset = config["dataset"]
    model = config["coupling_model"]
    training = config["coupling_training"]
    artifacts = config["coupling_artifacts"]
    source = dataset["coupling_source"]

    assert dataset["geometry_mode"] == "complex"
    assert dataset["geometry_path"] == "data/geometry/annulus_02_05_1_128.npz"
    assert dataset["dtype"] == model["dtype"] == "float64"
    assert dataset["reference_diagnostics"] == {
        "training": False,
        "validation": False,
    }
    assert source["mode"] == "indexed_gp"
    assert source["indexed_gp"] == {
        "num_train": 4800,
        "num_valid": 300,
        "seed": 0,
        "lengthscale": 0.15,
        "amplitude": 1.0,
        "mean": 0.0,
    }
    assert model["coefficient_terms"] == {
        "diffusion": True,
        "convection": False,
        "reaction": True,
    }
    assert model["branch_fusion"] == {"mode": "concat_fuser"}
    assert model["geometry_branch"]["enabled"] is False
    assert model["pre_projection_fusion"]["enabled"] is False
    axis = model["axis_1d_trunk"]
    assert axis["enabled"] is True
    assert axis["fixed_line_transverse_branch"]["enabled"] is False
    assert axis["transverse_trunk"] == {
        "enabled": True,
        "fusion": "concat_fuser",
        "length_context": True,
    }
    projection = model["balance_projection"]
    assert projection["enabled"] is True
    assert projection["mode"] == "symmetric_tangent_green_response"
    tangent = projection["symmetric_tangent_green_response"]
    assert tangent["subspace_dimension"] == tangent["max_subspace_dimension"] == 4
    assert tangent["geometry_k_selection"]["enabled"] is False
    assert tangent["preconditioner_variant"] == "separable"
    assert tangent["eta_cap_enabled"] is False
    assert training["epochs"] == 100
    assert training["batch_size"] == 200
    assert training["device"] == "cuda:1"
    assert training["epochs"] * (4800 // training["batch_size"]) == 2400
    assert training["warmup_steps"] == 240
    assert training["validation_every_steps"] == 24
    assert training["optimizer"]["name"] == "soap"
    assert training["canonical_energy"] == {"boundary_weight": 0.0}
    assert training["post_line_search_stationarity"]["enabled"] is False
    assert training["response_trust"]["enabled"] is False
    assert training["best_energy_checkpoint"]["enabled"] is True
    assert training["best_physics_checkpoint"]["enabled"] is False
    assert training["tangent_context_checkpoint"]["enabled"] is True
    assert model["cross_axis_reconstruction"] == {
        "enabled": True,
        "mode": "local_weak_residual_reliability",
        "gamma": 0.5,
        "smoothing_steps": 2,
        "smoothing_relaxation": 0.5,
        "relative_floor": 0.1,
        "eps": 1e-12,
    }
    assert artifacts["enabled"] is True
    assert artifacts["checkpoint"] == "best_energy"
    assert artifacts["visualization_mesh"] == (
        "data/visualization_mesh/annulus_02_05_1_128_mesh.npz"
    )
    assert config["pipeline"]["run_green"] is False
    assert config["pipeline"]["run_coupling"] is True
