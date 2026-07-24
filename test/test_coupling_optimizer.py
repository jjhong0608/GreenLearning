from __future__ import annotations

import math

import pytest
import torch

from greenonet.config import (
    CouplingOptimizerConfig,
    CouplingTrainingConfig,
    SoapOptimizerConfig,
    validate_unit_square_coupling_training_config,
)
from greenonet.coupling_optimizer import (
    ComplexCouplingOptimizerFactory,
    OptimizerStepProfiler,
    SOAP_UPSTREAM_COMMIT,
)
from greenonet.optimizers import SOAP


@pytest.fixture
def float64_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_default_optimizer_config_preserves_adamw_defaults():
    training = CouplingTrainingConfig(learning_rate=2.0e-3, weight_decay=0.05)
    factory = ComplexCouplingOptimizerFactory(training)
    parameter = torch.nn.Parameter(torch.ones(2))

    optimizer = factory.build([parameter])
    provenance = factory.provenance()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-8)
    assert provenance.name == "adamw"
    assert provenance.soap is None
    assert provenance.upstream_commit is None


def test_soap_config_round_trip_and_factory_mapping():
    training = CouplingTrainingConfig(
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
    factory = ComplexCouplingOptimizerFactory(training)
    parameter = torch.nn.Parameter(torch.ones((2, 2)))

    optimizer = factory.build([parameter])
    provenance = factory.provenance()

    assert isinstance(optimizer, SOAP)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert optimizer.param_groups[0]["betas"] == (0.95, 0.95)
    assert optimizer.param_groups[0]["eps"] == pytest.approx(2.0e-8)
    assert optimizer.param_groups[0]["precondition_frequency"] == 7
    assert optimizer.param_groups[0]["max_precond_dim"] == 64
    assert optimizer.param_groups[0]["merge_dims"] is True
    assert optimizer.param_groups[0]["precondition_1d"] is True
    assert optimizer.param_groups[0]["normalize_grads"] is True
    assert optimizer.param_groups[0]["correct_bias"] is False
    assert provenance.name == "soap"
    assert provenance.profile_step_time is True
    assert provenance.upstream_commit == SOAP_UPSTREAM_COMMIT
    assert provenance.soap is not None
    assert provenance.soap["max_precondition_dim"] == 64


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    (
        ({"name": "sgd"}, ValueError, "optimizer.name"),
        ({"betas": [0.9]}, TypeError, "optimizer.betas"),
        ({"betas": [0.9, 1.0]}, ValueError, "optimizer.betas"),
        ({"betas": [0.9, math.inf]}, ValueError, "optimizer.betas"),
        ({"eps": 0.0}, ValueError, "optimizer.eps"),
        ({"profile_step_time": 1}, TypeError, "profile_step_time"),
        ({"unknown": 1}, TypeError, "unknown keys"),
        (
            {"soap": {"precondition_frequency": 0}},
            ValueError,
            "precondition_frequency",
        ),
        (
            {"soap": {"max_precondition_dim": 0}},
            ValueError,
            "max_precondition_dim",
        ),
        ({"soap": {"shampoo_beta": 1.0}}, ValueError, "shampoo_beta"),
        ({"soap": {"merge_dims": 1}}, TypeError, "merge_dims"),
        ({"soap": {"unknown": 1}}, TypeError, "unknown keys"),
    ),
)
def test_optimizer_config_rejects_invalid_values(raw, error_type, message):
    with pytest.raises(error_type, match=message):
        CouplingOptimizerConfig.from_raw(raw)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("learning_rate", 0.0, ValueError),
        ("learning_rate", math.inf, ValueError),
        ("weight_decay", -1.0, ValueError),
        ("weight_decay", math.nan, ValueError),
    ),
)
def test_optimizer_factory_rejects_invalid_shared_values(field, value, error_type):
    config = CouplingTrainingConfig()
    setattr(config, field, value)
    with pytest.raises(error_type, match=field):
        ComplexCouplingOptimizerFactory(config)


def test_unit_square_rejects_soap_optimizer():
    config = CouplingTrainingConfig(optimizer={"name": "soap"})

    with pytest.raises(ValueError, match="only for ComplexCouplingTrainer"):
        validate_unit_square_coupling_training_config(config)


def test_soap_matches_pinned_upstream_numerical_fixture(float64_default_dtype):
    del float64_default_dtype
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    )
    target = torch.tensor([[0.25, -0.5], [1.0, -1.0]], dtype=torch.float32)
    optimizer = SOAP(
        [parameter],
        lr=1.0e-2,
        betas=(0.9, 0.95),
        shampoo_beta=-1.0,
        eps=1.0e-8,
        weight_decay=0.01,
        precondition_frequency=2,
        max_precond_dim=16,
        merge_dims=False,
        precondition_1d=False,
        normalize_grads=False,
        correct_bias=True,
    )
    initial = parameter.detach().clone()

    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        (parameter - target).square().sum().backward()
        optimizer.step()

    expected = torch.tensor(
        [
            [0.970324695110321, -1.99346923828125],
            [0.49387887120246887, 2.9696877002716064],
        ],
        dtype=torch.float32,
    )
    assert not torch.equal(parameter, initial)
    assert torch.allclose(parameter, expected, rtol=0.0, atol=1.0e-7)
    assert optimizer.step_call_count == 4
    assert optimizer.basis_refresh_count == 1


def test_soap_first_step_is_noop_and_float64_updates_are_finite(
    float64_default_dtype,
):
    del float64_default_dtype
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    optimizer = SOAP(
        [parameter],
        lr=1.0e-2,
        precondition_frequency=2,
        max_precond_dim=16,
    )
    initial = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    assert torch.equal(parameter, initial)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        optimizer.step()

    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    assert parameter.square().sum() < initial.square().sum()
    assert state["exp_avg"].dtype == torch.float64
    assert state["GG"][0].dtype == torch.float32
    assert state["Q"][0].dtype == torch.float32


def test_soap_state_dict_round_trip_preserves_parameter_state():
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    optimizer = SOAP([parameter], precondition_frequency=2, max_precond_dim=16)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        optimizer.step()

    state_dict = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = SOAP(
        [restored_parameter],
        precondition_frequency=2,
        max_precond_dim=16,
    )
    restored.load_state_dict(state_dict)
    source_state = optimizer.state[parameter]
    restored_state = restored.state[restored_parameter]

    assert restored_state["step"] == source_state["step"]
    assert torch.equal(restored_state["exp_avg"], source_state["exp_avg"])
    assert torch.equal(restored_state["exp_avg_sq"], source_state["exp_avg_sq"])
    for restored_factor, source_factor in zip(
        restored_state["GG"],
        source_state["GG"],
        strict=True,
    ):
        assert torch.equal(restored_factor, source_factor)


def test_soap_keeps_1d_parameters_in_original_adam_basis():
    matrix = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    vector = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    optimizer = SOAP(
        [matrix, vector],
        precondition_1d=False,
        max_precond_dim=16,
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (matrix.square().sum() + vector.square().sum()).backward()
        optimizer.step()

    vector_state = optimizer.state[vector]
    assert vector_state["exp_avg"].shape == vector.shape
    assert vector_state["exp_avg_sq"].shape == vector.shape
    assert vector_state["GG"] == [[]]
    assert vector_state["Q"] == [[]]
    assert torch.isfinite(vector).all()


def test_soap_allows_none_gradients_and_rejects_sparse_gradients():
    parameter = torch.nn.Parameter(torch.ones((2, 2)))
    optimizer = SOAP([parameter], max_precond_dim=16)
    optimizer.step()
    assert parameter not in optimizer.state

    with torch.sparse.check_sparse_tensor_invariants():
        parameter.grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [1]]),
            values=torch.tensor([1.0]),
            size=(2, 2),
        )
    with pytest.raises(RuntimeError, match="sparse gradients"):
        optimizer.step()


def test_optimizer_step_profiler_reports_step_and_refresh_telemetry():
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    optimizer = SOAP(
        [parameter],
        lr=1.0e-2,
        precondition_frequency=2,
        max_precond_dim=16,
    )
    profiler = OptimizerStepProfiler(
        optimizer=optimizer,
        enabled=True,
        device=torch.device("cpu"),
    )
    profiler.begin_epoch()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        profiler.step()

    metrics = profiler.finish_epoch()

    assert metrics["optimizer_step_count"] == 3.0
    assert metrics["optimizer_basis_refresh_count"] == 1.0
    assert metrics["optimizer_step_time_mean_ms"] > 0.0
    assert (
        metrics["optimizer_step_time_mean_ms"] <= metrics["optimizer_step_time_max_ms"]
    )
    assert metrics["optimizer_step_time_p95_ms"] > 0.0
    assert metrics["optimizer_peak_memory_mib"] == 0.0


def test_disabled_optimizer_profiler_adds_no_metrics():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    profiler = OptimizerStepProfiler(
        optimizer=optimizer,
        enabled=False,
        device=torch.device("cpu"),
    )
    profiler.begin_epoch()
    parameter.square().sum().backward()
    profiler.step()

    assert profiler.finish_epoch() == {}


def test_enabled_adamw_profiler_reports_steps_without_basis_refreshes():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    profiler = OptimizerStepProfiler(
        optimizer=optimizer,
        enabled=True,
        device=torch.device("cpu"),
    )
    profiler.begin_epoch()
    parameter.square().sum().backward()
    profiler.step()

    metrics = profiler.finish_epoch()

    assert metrics["optimizer_step_count"] == 1.0
    assert metrics["optimizer_basis_refresh_count"] == 0.0
    assert metrics["optimizer_step_time_mean_ms"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_cuda_optimizer_profiler_records_peak_memory():
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda"))
    optimizer = torch.optim.AdamW([parameter])
    profiler = OptimizerStepProfiler(
        optimizer=optimizer,
        enabled=True,
        device=torch.device("cuda"),
    )
    profiler.begin_epoch()
    parameter.square().sum().backward()
    profiler.step()

    metrics = profiler.finish_epoch()

    assert metrics["optimizer_step_count"] == 1.0
    assert metrics["optimizer_step_time_mean_ms"] > 0.0
    assert metrics["optimizer_peak_memory_mib"] > 0.0


def test_soap_dataclass_defaults_are_stable():
    assert SoapOptimizerConfig() == SoapOptimizerConfig.from_raw(None)
