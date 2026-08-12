from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from cli.train import TrainCLI
from greenonet.model import GreenONetModel
from test.complex_fixtures import write_geometry_npz


def _write_complex_green_config(path, *, geometry_path=None):
    payload = {
        "dataset": {
            "geometry_mode": "complex",
            "geometry_path": None if geometry_path is None else str(geometry_path),
            "samples_per_line": 1,
            "validation_samples_per_line": 0,
            "sampler_mode": "forward",
            "scale_length": 0.1,
            "deterministic": True,
            "dtype": "float64",
        },
        "model": {
            "hidden_dim": 4,
            "depth": 1,
            "branch_input_dim": 5,
            "dtype": "float64",
        },
        "training": {
            "seed": 0,
            "epochs": 1,
            "batch_size": 1,
            "device": "cpu",
            "compile": {"enabled": False},
        },
        "pipeline": {
            "run_green": True,
            "run_coupling": False,
            "green_pretrained_path": None,
            "coupling_pretrained_path": None,
        },
    }
    path.write_text(json.dumps(payload))
    return payload


def test_cli_train_complex_green_dispatches_to_complex_runner(tmp_path, monkeypatch):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    config_path = tmp_path / "config.json"
    _write_complex_green_config(config_path, geometry_path=geometry_path)
    work_dir = tmp_path / "work"
    captured = {}

    def fake_run_complex_green_o_net(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=GreenONetModel(kwargs["model_cfg"]))

    def fail_unit_runner(*_args, **_kwargs):
        raise AssertionError("unit-square runner must not be used for complex mode.")

    monkeypatch.setattr(
        "cli.train.run_complex_green_o_net", fake_run_complex_green_o_net
    )
    monkeypatch.setattr("cli.train.run_green_o_net", fail_unit_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--work-dir", str(work_dir)],
    )

    TrainCLI().run()

    assert captured["geometry_path"] == geometry_path
    assert captured["ndata"] == 1
    assert captured["sampler_mode"] == "forward"
    assert captured["seed"] == 0
    assert captured["model_cfg"].branch_input_dim == 5


def test_cli_train_complex_green_requires_geometry_path(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    _write_complex_green_config(config_path, geometry_path=None)
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--work-dir", str(work_dir)],
    )

    with pytest.raises(ValueError, match="dataset.geometry_path"):
        TrainCLI().run()
