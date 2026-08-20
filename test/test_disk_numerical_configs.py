from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from cli.train import TrainCLI
from greenonet.coefficients import load_coefficient_functions


ROOT = Path(__file__).resolve().parents[1]
DISK_DIR = ROOT / "numerical_examples" / "disk"
BASE_CONFIG_PATH = DISK_DIR / "base_config.json"
SEEDS = range(4)
OBJECTIVE_FLAGS = {
    "energy_only": (False, False),
    "energy_response_trust": (False, True),
    "energy_stationarity": (True, False),
    "energy_response_trust_stationarity": (True, True),
}


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def _normalized_experiment(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    dataset = normalized["dataset"]
    assert isinstance(dataset, dict)
    coupling_source = dataset["coupling_source"]
    assert isinstance(coupling_source, dict)
    indexed_gp = coupling_source["indexed_gp"]
    assert isinstance(indexed_gp, dict)
    indexed_gp["seed"] = 0

    training = normalized["coupling_training"]
    assert isinstance(training, dict)
    training["seed"] = 0
    stationarity = training["post_line_search_stationarity"]
    response_trust = training["response_trust"]
    assert isinstance(stationarity, dict)
    assert isinstance(response_trust, dict)
    stationarity["enabled"] = True
    response_trust["enabled"] = True
    return normalized


def test_disk_coefficient_is_asymmetric_uniformly_elliptic_diffusion() -> None:
    coefficient = load_coefficient_functions(DISK_DIR / "coefficient.py")
    x = torch.tensor([0.25, 0.25, 0.0], dtype=torch.float64)
    y = torch.tensor([0.125, -0.125, 0.0], dtype=torch.float64)

    diffusion = coefficient.a_fun(x, y)

    assert torch.allclose(
        diffusion,
        torch.tensor([1.5, 0.5, 1.0], dtype=torch.float64),
    )
    assert not torch.allclose(
        coefficient.a_fun(x, y),
        coefficient.a_fun(y, x),
    )
    assert torch.allclose(coefficient.bx_fun(x, y), torch.zeros_like(x))
    assert torch.allclose(coefficient.by_fun(x, y), torch.zeros_like(x))
    assert torch.allclose(coefficient.c_fun(x, y), torch.zeros_like(x))


def test_disk_coefficient_derivatives_match_autograd_and_green_checkpoint() -> None:
    coefficient = load_coefficient_functions(DISK_DIR / "coefficient.py")
    green_coefficient = load_coefficient_functions(
        ROOT / "coefficients" / "Sinusoidal_Diffusion_Only.py"
    )
    x = torch.tensor([-0.31, -0.08, 0.17, 0.39], dtype=torch.float64)
    y = torch.tensor([0.11, -0.27, 0.29, -0.06], dtype=torch.float64)
    x_grad = x.clone().requires_grad_(True)
    y_grad = y.clone().requires_grad_(True)

    grad_x, grad_y = torch.autograd.grad(
        coefficient.a_fun(x_grad, y_grad).sum(),
        (x_grad, y_grad),
    )

    assert torch.allclose(coefficient.apx_fun(x, y), grad_x)
    assert torch.allclose(coefficient.apy_fun(x, y), grad_y)
    for name in ("a_fun", "apx_fun", "apy_fun", "bx_fun", "by_fun", "c_fun"):
        actual = getattr(coefficient, name)(x, y)
        expected = getattr(green_coefficient, name)(x, y)
        assert torch.allclose(actual, expected)


def test_disk_objective_matrix_has_four_paired_seeds_and_no_other_differences() -> None:
    base = _load_json(BASE_CONFIG_PATH)
    expected_paths = {
        DISK_DIR / f"disk_{objective}_seed{seed}.json"
        for objective in OBJECTIVE_FLAGS
        for seed in SEEDS
    }
    actual_paths = set(DISK_DIR.glob("disk_*.json"))

    assert actual_paths == expected_paths
    for objective, (
        stationarity_enabled,
        response_trust_enabled,
    ) in OBJECTIVE_FLAGS.items():
        for seed in SEEDS:
            path = DISK_DIR / f"disk_{objective}_seed{seed}.json"
            payload = _load_json(path)
            dataset = payload["dataset"]
            training = payload["coupling_training"]
            assert isinstance(dataset, dict)
            assert isinstance(training, dict)
            coupling_source = dataset["coupling_source"]
            assert isinstance(coupling_source, dict)
            indexed_gp = coupling_source["indexed_gp"]
            assert isinstance(indexed_gp, dict)
            stationarity = training["post_line_search_stationarity"]
            response_trust = training["response_trust"]
            assert isinstance(stationarity, dict)
            assert isinstance(response_trust, dict)

            assert indexed_gp["seed"] == seed
            assert training["seed"] == seed
            assert stationarity["enabled"] is stationarity_enabled
            assert response_trust["enabled"] is response_trust_enabled
            assert _normalized_experiment(payload) == base


def test_disk_configs_use_fixed_paper_protocol_and_strictly_parse() -> None:
    config_paths = [BASE_CONFIG_PATH, *sorted(DISK_DIR.glob("disk_*.json"))]

    for path in config_paths:
        payload = _load_json(path)
        dataset = payload["dataset"]
        model = payload["coupling_model"]
        training = payload["coupling_training"]
        pipeline = payload["pipeline"]
        assert isinstance(dataset, dict)
        assert isinstance(model, dict)
        assert isinstance(training, dict)
        assert isinstance(pipeline, dict)
        coupling_source = dataset["coupling_source"]
        assert isinstance(coupling_source, dict)
        indexed_gp = coupling_source["indexed_gp"]
        assert isinstance(indexed_gp, dict)

        assert indexed_gp["num_train"] == 4800
        assert indexed_gp["num_valid"] == 300
        assert dataset["geometry_path"] == ("data/geometry/disk_radius_05_1_128.npz")
        assert dataset["reference_diagnostics"] == {
            "training": False,
            "validation": False,
        }
        assert model["coefficient_terms"] == {
            "diffusion": True,
            "convection": False,
            "reaction": False,
        }
        assert model["branch_fusion"] == {"mode": "product_fuser"}
        projection = model["balance_projection"]
        assert isinstance(projection, dict)
        tangent = projection["symmetric_tangent_green_response"]
        assert isinstance(tangent, dict)
        assert tangent["subspace_dimension"] == 4
        assert training["epochs"] == 100
        assert training["batch_size"] == 200
        assert training["warmup_steps"] == 240
        assert training["validation_every_steps"] == 24
        assert training["canonical_energy"] == {"boundary_weight": 0.0}
        assert pipeline["green_pretrained_path"] == (
            "checkpoints/numerical_examples/disk/green/model.safetensors"
        )

        TrainCLI()._build_configs(path)

    assert (ROOT / "data" / "geometry" / "disk_radius_05_1_128.npz").is_file()
    test_path = (
        ROOT
        / "data"
        / "complex_samples"
        / "circle_radius_05_1_128_sinusoidal_diffusion"
        / "test"
    )
    assert len(tuple(test_path.glob("*.npz"))) == 50
    assert (
        ROOT
        / "checkpoints"
        / "numerical_examples"
        / "disk"
        / "green"
        / "model.safetensors"
    ).is_file()
