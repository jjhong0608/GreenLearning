from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from cli.train import TrainCLI


ROOT = Path(__file__).resolve().parents[1]
PENTAGRAM_DIR = ROOT / "numerical_examples" / "pentagram"
K_VALUES = (0, 1, 2, 3, 4, 5, 9, 10)
MACHINE_SPECS = {
    "nvidia_a40": {"device": "cuda:1", "seeds": (0, 2)},
    "mac_studio": {"device": "cpu", "seeds": (1, 3)},
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def _expected_paths() -> set[Path]:
    return {
        PENTAGRAM_DIR
        / machine
        / f"seed{seed}"
        / f"pentagram_k{k_value}_seed{seed}.json"
        for machine, spec in MACHINE_SPECS.items()
        for seed in spec["seeds"]
        for k_value in K_VALUES
    }


def _remove_experiment_axes(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    source = normalized["dataset"]["coupling_source"]["indexed_gp"]
    training = normalized["coupling_training"]
    tangent = normalized["coupling_model"]["balance_projection"][
        "symmetric_tangent_green_response"
    ]
    source["seed"] = "paired"
    training["seed"] = "paired"
    training["device"] = "paired"
    training["tangent_context_checkpoint"]["enabled"] = "paired"
    projection = normalized["coupling_model"]["balance_projection"]
    projection["mode"] = "paired"
    for key in (
        "subspace_dimension",
        "max_subspace_dimension",
        "eta",
        "eta_cap_enabled",
        "eta_strategy",
    ):
        tangent[key] = "paired"
    return normalized


def test_pentagram_paper_config_matrix_is_complete_and_strictly_parseable() -> None:
    actual_paths = set(PENTAGRAM_DIR.glob("*/*/*.json"))
    assert actual_paths == _expected_paths()
    assert len(actual_paths) == 32

    cli = TrainCLI()
    for path in sorted(actual_paths):
        cli._build_configs(path)


def test_pentagram_paper_configs_preserve_the_paired_training_contract() -> None:
    normalized_payloads: list[dict[str, Any]] = []
    for machine, spec in MACHINE_SPECS.items():
        for seed in spec["seeds"]:
            for k_value in K_VALUES:
                path = (
                    PENTAGRAM_DIR
                    / machine
                    / f"seed{seed}"
                    / f"pentagram_k{k_value}_seed{seed}.json"
                )
                payload = _load(path)
                dataset = payload["dataset"]
                model = payload["coupling_model"]
                training = payload["coupling_training"]
                artifacts = payload["coupling_artifacts"]
                indexed_gp = dataset["coupling_source"]["indexed_gp"]
                projection = model["balance_projection"]
                tangent = projection["symmetric_tangent_green_response"]

                assert indexed_gp["num_train"] == 4800
                assert indexed_gp["num_valid"] == 300
                assert indexed_gp["seed"] == seed
                assert training["seed"] == seed
                assert training["device"] == spec["device"]
                assert training["epochs"] == 100
                assert training["batch_size"] == 200
                assert (
                    training["epochs"]
                    * (indexed_gp["num_train"] // training["batch_size"])
                    == 2400
                )
                assert "warmup_epochs" not in training
                assert training["warmup_steps"] == 240
                assert training["validation_every_steps"] == 24
                assert training["deterministic_algorithms"] is True
                assert training["optimizer"]["name"] == "soap"
                assert artifacts["checkpoint"] == "best_energy"
                assert artifacts["plot_workers"] == 1

                assert model["branch_fusion"]["mode"] == "concat_fuser"
                assert model["geometry_branch"]["enabled"] is False
                axis = model["axis_1d_trunk"]
                assert axis["fixed_line_transverse_branch"]["enabled"] is False
                assert axis["transverse_trunk"] == {
                    "enabled": True,
                    "fusion": "concat_fuser",
                    "length_context": True,
                }

                context_checkpoint = training["tangent_context_checkpoint"]
                if k_value == 0:
                    assert projection["mode"] == "physical_symmetric"
                    assert tangent["subspace_dimension"] == 1
                    assert tangent["max_subspace_dimension"] == 1
                    assert tangent["eta"] == 0.0
                    assert tangent["eta_cap_enabled"] is True
                    assert tangent["eta_strategy"] == "fixed"
                    assert context_checkpoint["enabled"] is False
                else:
                    assert projection["mode"] == "symmetric_tangent_green_response"
                    assert tangent["subspace_dimension"] == k_value
                    assert tangent["max_subspace_dimension"] == k_value
                    assert tangent["eta_cap_enabled"] is False
                    assert tangent["eta_strategy"] == "closed_loop_exact_line_search"
                    assert context_checkpoint["enabled"] is True

                normalized_payloads.append(_remove_experiment_axes(payload))

    reference = normalized_payloads[0]
    assert all(payload == reference for payload in normalized_payloads[1:])
