from __future__ import annotations

import copy
import json
from pathlib import Path

from cli.train import TrainCLI


ROOT = Path(__file__).resolve().parents[1]
PENTAGRAM_DIR = ROOT / "numerical_examples" / "pentagram"
FUSIONS = ("product_fuser", "concat_fuser")
SEEDS = range(4)


def _load(path: Path) -> dict[str, object]:
    with path.open() as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def _expected(*, fusion: str, seed: int) -> dict[str, object]:
    payload = copy.deepcopy(_load(PENTAGRAM_DIR / "pentagram_base.json"))
    dataset = payload["dataset"]
    model = payload["coupling_model"]
    training = payload["coupling_training"]
    assert isinstance(dataset, dict)
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    source = dataset["coupling_source"]
    projection = model["balance_projection"]
    axis = model["axis_1d_trunk"]
    assert isinstance(source, dict)
    assert isinstance(projection, dict)
    assert isinstance(axis, dict)
    indexed_gp = source["indexed_gp"]
    tangent = projection["symmetric_tangent_green_response"]
    transverse = axis["transverse_trunk"]
    assert isinstance(indexed_gp, dict)
    assert isinstance(tangent, dict)
    assert isinstance(transverse, dict)
    indexed_gp["seed"] = seed
    training["seed"] = seed
    transverse["fusion"] = fusion
    tangent["preconditioner_variant"] = "separable"
    tangent["cross_axis_relative_eps"] = 1.0e-12
    return payload


def test_pentagram_trunk_fuser_configs_are_strict_paired_derivatives() -> None:
    for fusion in FUSIONS:
        for seed in SEEDS:
            path = PENTAGRAM_DIR / f"pentagram_trunk_{fusion}_seed{seed}.json"
            assert _load(path) == _expected(fusion=fusion, seed=seed)
            TrainCLI()._build_configs(path)


def test_pentagram_trunk_fuser_pairs_differ_only_in_trunk_fusion() -> None:
    for seed in SEEDS:
        product = _load(
            PENTAGRAM_DIR / f"pentagram_trunk_product_fuser_seed{seed}.json"
        )
        concat = _load(PENTAGRAM_DIR / f"pentagram_trunk_concat_fuser_seed{seed}.json")
        product_axis = product["coupling_model"]["axis_1d_trunk"]  # type: ignore[index]
        concat_axis = concat["coupling_model"]["axis_1d_trunk"]  # type: ignore[index]
        product_transverse = product_axis["transverse_trunk"]  # type: ignore[index]
        concat_transverse = concat_axis["transverse_trunk"]  # type: ignore[index]
        assert product_transverse["fusion"] == "product_fuser"  # type: ignore[index]
        assert concat_transverse["fusion"] == "concat_fuser"  # type: ignore[index]
        product_transverse["fusion"] = "paired"
        concat_transverse["fusion"] = "paired"
        assert product == concat
