from __future__ import annotations

import csv
import json
import math

import pytest
import torch
from safetensors.torch import load_file

from greenonet.compile_utils import model_state_dict_for_save
from greenonet.config import GreenOptimizerConfig, ModelConfig, TrainingConfig
from greenonet.green_optimizer import GreenOptimizerFactory
from greenonet.model import GreenONetModel
from greenonet.optimizers import SOAP
from greenonet.trainer import Trainer


def test_green_default_optimizer_is_adamw_without_weight_decay():
    config = TrainingConfig(learning_rate=2.0e-3)
    factory = GreenOptimizerFactory(config)
    parameter = torch.nn.Parameter(torch.ones(2))

    optimizer = factory.build([parameter])
    provenance = factory.provenance()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-8)
    assert provenance.name == "adamw"
    assert provenance.soap is None


def test_green_soap_config_round_trip_and_factory_mapping():
    config = TrainingConfig(
        learning_rate=3.0e-3,
        weight_decay=0.02,
        optimizer={
            "name": "soap",
            "betas": [0.95, 0.95],
            "eps": 2.0e-8,
            "profile_step_time": True,
            "soap": {
                "shampoo_beta": 0.9,
                "precondition_frequency": 7,
                "max_precondition_dim": 64,
                "merge_dims": True,
                "precondition_1d": True,
                "normalize_grads": True,
                "correct_bias": False,
            },
        },
    )
    factory = GreenOptimizerFactory(config)
    parameter = torch.nn.Parameter(torch.ones((2, 2)))

    optimizer = factory.build([parameter])
    provenance = factory.provenance()

    assert isinstance(optimizer, SOAP)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-3)
    assert optimizer.param_groups[0]["precondition_frequency"] == 7
    assert optimizer.param_groups[0]["max_precond_dim"] == 64
    assert provenance.name == "soap"
    assert provenance.soap is not None
    assert provenance.upstream_commit == ("a1e553530fde97d0e6b307d7c82ac6d38b072340")
    assert factory.resolved_config()["betas"] == (0.95, 0.95)


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    (
        ({"name": "adam"}, ValueError, "Adam has been removed"),
        ({"name": "sgd"}, ValueError, "optimizer.name"),
        ({"betas": [0.9]}, TypeError, "optimizer.betas"),
        ({"betas": [0.9, math.inf]}, ValueError, "optimizer.betas"),
        ({"eps": 0.0}, ValueError, "optimizer.eps"),
        ({"profile_step_time": 1}, TypeError, "profile_step_time"),
        ({"unknown": 1}, TypeError, "unknown keys"),
        (
            {"soap": {"precondition_frequency": 0}},
            ValueError,
            "precondition_frequency",
        ),
    ),
)
def test_green_optimizer_config_rejects_invalid_values(raw, error_type, message):
    with pytest.raises(error_type, match=message):
        GreenOptimizerConfig.from_raw(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 0.0),
        ("learning_rate", math.inf),
        ("weight_decay", -1.0),
        ("weight_decay", math.nan),
    ),
)
def test_green_optimizer_factory_rejects_invalid_shared_values(field, value):
    config = TrainingConfig()
    setattr(config, field, value)

    with pytest.raises(ValueError, match=field):
        GreenOptimizerFactory(config)


def test_green_soap_first_step_is_noop_and_second_step_updates():
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    factory = GreenOptimizerFactory(
        TrainingConfig(
            learning_rate=1.0e-2,
            optimizer={
                "name": "soap",
                "soap": {
                    "precondition_frequency": 2,
                    "max_precondition_dim": 16,
                },
            },
        )
    )
    optimizer = factory.build([parameter])
    initial = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    assert torch.equal(parameter, initial)

    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    assert not torch.equal(parameter, initial)
    assert torch.isfinite(parameter).all()


def test_unit_square_green_trainer_soap_schedule_and_metrics(tmp_path):
    torch.manual_seed(0)
    m_points = 3
    coords = torch.zeros((2, 1, m_points, 2), dtype=torch.float64)
    solution = torch.ones((2, 1, m_points), dtype=torch.float64)
    source = torch.ones_like(solution)
    coefficients = torch.ones_like(solution)
    zeros = torch.zeros_like(solution)
    dataset = [
        (
            coords,
            solution,
            source,
            coefficients,
            zeros,
            zeros,
            zeros,
        )
    ]
    model_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        activation="tanh",
        use_green=False,
        branch_input_dim=m_points,
        dtype=torch.float64,
    )
    model = GreenONetModel(model_cfg)
    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            epochs=3,
            batch_size=1,
            log_interval=1,
            device="cpu",
            use_lr_schedule=True,
            warmup_epochs=2,
            min_lr=1.0e-4,
            optimizer={
                "name": "soap",
                "profile_step_time": True,
                "soap": {
                    "precondition_frequency": 2,
                    "max_precondition_dim": 16,
                },
            },
            lbfgs_max_iter=0,
        ),
        work_dir=tmp_path,
        model_cfg=model_cfg,
    )

    trainer.train(dataset)

    rows = list(csv.DictReader((tmp_path / "green_training_metrics.csv").open()))
    assert [row["phase"] for row in rows] == ["soap", "soap", "soap"]
    assert [float(row["learning_rate"]) for row in rows] == pytest.approx(
        [5.0e-4, 1.0e-3, 1.0e-4]
    )
    assert all(float(row["optimizer_step_count"]) == 1.0 for row in rows)
    provenance = json.loads((tmp_path / "green_optimizer_provenance.json").read_text())
    assert provenance["optimizer"]["name"] == "soap"
    assert provenance["learning_rate_schedule"]["enabled"] is True
    assert provenance["learning_rate_schedule"]["steps_per_epoch"] == 1
    assert provenance["learning_rate_schedule"]["total_optimizer_steps"] == 3
    assert provenance["validation_schedule"] == {"active": False}
    assert provenance["lbfgs_scheduler"] == "disabled"
    expected_model_keys = set(model_state_dict_for_save(model))
    for checkpoint_name in ("model_pre_lbfgs.safetensors", "model.safetensors"):
        checkpoint = tmp_path / checkpoint_name
        assert checkpoint.exists()
        assert set(load_file(checkpoint)) == expected_model_keys

    training_log = (tmp_path / "training.log").read_text()
    assert "Green optimizer name=soap" in training_log
    assert "Green learning-rate schedule enabled=True" in training_log
    assert "learning_rate=5.000000e-04" in training_log
