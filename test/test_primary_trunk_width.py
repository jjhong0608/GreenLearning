from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from cli.eval_coupling import EvalCouplingCLI
from cli.train import TrainCLI
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.config import CouplingModelConfig
from greenonet.coupling_artifacts import load_coupling_artifact_configs
from greenonet.coupling_model import CouplingNet, MLP
from greenonet.io import load_state_dict_auto, save_state_dict_safetensors
from test.test_complex_coupling_model import _build_item, _forward, _model

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "numerical_examples/unit_square/nvidia_a40/seed0"


@pytest.mark.parametrize("explicit_null", [False, True])
def test_default_width_loaders(tmp_path, explicit_null) -> None:
    raw = json.loads((BASE / "unit_square_trunk_off_seed0.json").read_text())
    if explicit_null:
        raw["coupling_model"]["primary_trunk_hidden_dim"] = None
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    assert TrainCLI()._build_configs(path)[3].primary_trunk_hidden_dim is None
    assert (
        load_coupling_artifact_configs(path).coupling_model.primary_trunk_hidden_dim
        is None
    )


def test_legacy_positional_config_order() -> None:
    cfg = CouplingModelConfig(4, 2, 8, 3, "tanh")
    assert cfg.hidden_dim == 8 and cfg.depth == 3
    assert cfg.activation == "tanh" and cfg.primary_trunk_hidden_dim is None


@pytest.mark.parametrize(
    "bad", [True, False, 0, -1, 428.0, "428", float("nan"), float("inf")]
)
def test_invalid_width(bad) -> None:
    with pytest.raises(ValueError, match="primary_trunk_hidden_dim"):
        CouplingModelConfig(primary_trunk_hidden_dim=bad)


def test_legacy_square_rejects_width() -> None:
    with pytest.raises(ValueError, match="only for ComplexCouplingNet"):
        CouplingNet(CouplingModelConfig(primary_trunk_hidden_dim=8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mlp_separate_output_dimension(dtype) -> None:
    model = MLP(1, 13, 4, "tanh", True, 0, True, output_dim=8).to(dtype)
    y = model(torch.ones(3, 1, dtype=dtype))
    assert y.shape == (3, 8) and y.dtype == dtype
    y.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("bad", [0, -1, True, 4.0, "4"])
def test_invalid_mlp_output_dimension(bad) -> None:
    with pytest.raises(ValueError, match="output_dim"):
        MLP(1, 8, 1, "tanh", True, 0, output_dim=bad)


def test_mlp_matches_historical_initialization_and_output() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(37)
    layers = []
    for i in range(4):
        layers.extend(
            [nn.Linear(1 if i == 0 else 8, 8), MLP.build_activation("rational")]
        )
    layers.extend([nn.Linear(8, 8), MLP.build_activation("rational")])
    legacy = nn.Sequential(*layers)
    for output_dim in (None, 8):
        torch.manual_seed(37)
        model = MLP(1, 8, 4, "rational", True, 0, True, output_dim=output_dim)
        assert set(model.net.state_dict()) == set(legacy.state_dict())
        for key, value in legacy.state_dict().items():
            assert torch.equal(value, model.net.state_dict()[key])
        x = torch.linspace(0, 1, 5).unsqueeze(-1)
        assert torch.equal(legacy(x), model(x))


def test_default_explicit_same_width_and_strict_checkpoint(tmp_path) -> None:
    original = _model()
    torch.manual_seed(0)
    explicit = ComplexCouplingNet(replace(original.config, primary_trunk_hidden_dim=8))
    assert original.state_dict().keys() == explicit.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    path = tmp_path / "legacy.safetensors"
    save_state_dict_safetensors(original.state_dict(), path)
    load_state_dict_auto(explicit, path)
    geometry, item = _build_item(tmp_path)
    assert torch.equal(
        _forward(original, geometry, item), _forward(explicit, geometry, item)
    )


@pytest.mark.parametrize("fusion", ["product", "product_fuser", "concat_fuser"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("width", [None, 13])
def test_synthetic_forward_backward(tmp_path, fusion, enabled, width) -> None:
    geometry, item = _build_item(tmp_path)
    template = _model(transverse_trunk_enabled=enabled, transverse_trunk_fusion=fusion)
    model = ComplexCouplingNet(replace(template.config, primary_trunk_hidden_dim=width))
    output = _forward(model, geometry, item)
    assert output.shape == (1, 2, geometry.num_points)
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.trunk.parameters()
    )
    assert torch.isfinite(output).all()
    info = model.architecture_provenance()
    assert info["primary_trunk_hidden_dim_resolved"] == (8 if width is None else width)
    assert info["primary_trunk_output_dim"] == 8
    saved = tmp_path / "model.safetensors"
    save_state_dict_safetensors(model.state_dict(), saved)
    clone = ComplexCouplingNet(model.config)
    load_state_dict_auto(clone, saved)
    assert torch.equal(output, _forward(clone, geometry, item))
    if width is not None:
        with pytest.raises(RuntimeError, match="size mismatch"):
            model.load_state_dict(template.state_dict(), strict=True)


def test_parameter_counts_and_source_preservation() -> None:
    old = TrainCLI()._build_configs(BASE / "unit_square_trunk_off_seed0.json")[3]
    on = TrainCLI()._build_configs(BASE / "unit_square_trunk_on_seed0.json")[3]
    shapes = None
    for cfg, expected in (
        (old, 560182),
        (on, 955994),
        (replace(old, primary_trunk_hidden_dim=428), 958018),
    ):
        torch.manual_seed(7)
        model = ComplexCouplingNet(cfg)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
        source = model.branch_source.state_dict()
        if shapes is None:
            shapes = source
        for name in source:
            assert torch.equal(source[name], shapes[name])
        if cfg.primary_trunk_hidden_dim == 428:
            assert sum(p.numel() for p in model.trunk.parameters()) == 661546
            assert model.trunk.net[-2].out_features == 256


@pytest.mark.parametrize("seed", range(4))
def test_configs_loaders_and_config_used(tmp_path, monkeypatch, seed) -> None:
    hardware = "nvidia_a40" if seed in (0, 2) else "mac_studio"
    folder = ROOT / f"numerical_examples/unit_square/{hardware}/seed{seed}"
    path = folder / f"unit_square_primary_w428_trunk_off_seed{seed}.json"
    raw = json.loads(path.read_text())
    expected = json.loads(
        (folder / f"unit_square_trunk_off_seed{seed}.json").read_text()
    )
    expected["coupling_model"]["primary_trunk_hidden_dim"] = 428
    assert raw == expected
    dataset, _, training, model, coupling_training, pipeline, _ = (
        TrainCLI()._build_configs(path)
    )
    assert model.primary_trunk_hidden_dim == 428
    artifacts = load_coupling_artifact_configs(path)
    assert artifacts.coupling_model.primary_trunk_hidden_dim == 428
    TrainCLI._write_config_used(
        config_path=path,
        work_dir=tmp_path,
        dataset_cfg=dataset,
        training_cfg=training,
        coupling_training_cfg=coupling_training,
        coupling_model_cfg=model,
        pipeline_cfg=pipeline,
    )
    saved = tmp_path / "config_used.json"
    assert TrainCLI()._build_configs(saved)[3].primary_trunk_hidden_dim == 428
    assert (
        load_coupling_artifact_configs(saved).coupling_model.primary_trunk_hidden_dim
        == 428
    )
    cli = EvalCouplingCLI()
    observed = []
    monkeypatch.setattr(
        cli,
        "_run_complex_evaluation",
        lambda **kw: observed.append(kw["coupling_model_cfg"]),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval",
            "--config",
            str(saved),
            "--work-dir",
            str(tmp_path),
            "--coupling-checkpoint",
            "unused.safetensors",
            "--green-checkpoint",
            "unused.safetensors",
        ],
    )
    cli.run()
    assert observed[0].primary_trunk_hidden_dim == 428
