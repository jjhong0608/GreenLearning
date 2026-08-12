from __future__ import annotations

import random
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_sources import derive_indexed_seed
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingBranchFusionConfig,
    CouplingModelConfig,
    CouplingTrainingConfig,
    PipelineConfig,
    TrainingConfig,
    TransverseTrunkConfig,
    validate_active_training_seeds,
)
from greenonet.reproducibility import (
    MAX_TRAINING_SEED,
    TrainingSeedContext,
    derive_training_subseed,
)


@pytest.fixture(autouse=True)
def _restore_torch_determinism() -> None:
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _small_coupling_config(fusion_mode: str) -> CouplingModelConfig:
    return CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=8,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        branch_fusion=CouplingBranchFusionConfig(mode=fusion_mode),
        axis_1d_trunk=Axis1DTrunkConfig(
            enabled=True,
            num_frequencies=2,
            max_frequency=2.0,
            transverse_trunk=TransverseTrunkConfig(
                enabled=True,
                fusion="product",
                length_context=True,
            ),
        ),
    )


def _one_step(seed: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    context = TrainingSeedContext(
        stage="coupling",
        base_seed=seed,
        deterministic_algorithms=True,
        device="cpu",
    )
    context.configure_process()
    context.apply("model")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 6),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(6, 1),
    ).double()
    context.apply("runtime")
    indices = torch.arange(12)
    values = torch.stack((indices.double(), (indices.double() + 1.0) / 3.0), dim=1)
    loader = DataLoader(
        TensorDataset(indices, values),
        batch_size=4,
        shuffle=True,
        generator=context.make_generator("loader_train"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_indices, batch_values = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    model(batch_values).square().mean().backward()
    optimizer.step()
    return batch_indices.tolist(), {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }


@pytest.mark.parametrize("seed", [0, MAX_TRAINING_SEED])
def test_training_seed_boundary_values_are_valid(seed: int) -> None:
    assert TrainingConfig(seed=seed).seed == seed
    assert CouplingTrainingConfig(seed=seed).seed == seed


@pytest.mark.parametrize("seed", [-1, MAX_TRAINING_SEED + 1])
def test_training_seed_out_of_range_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        TrainingConfig(seed=seed)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_training_seed_non_integer_is_rejected(seed: object) -> None:
    with pytest.raises(TypeError, match="integer or null"):
        CouplingTrainingConfig(seed=seed)  # type: ignore[arg-type]


def test_deterministic_algorithms_requires_strict_boolean() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        TrainingConfig(deterministic_algorithms=1)  # type: ignore[arg-type]


def test_active_training_stages_require_their_own_seed() -> None:
    with pytest.raises(ValueError, match="training.seed"):
        validate_active_training_seeds(
            training=TrainingConfig(),
            coupling_training=CouplingTrainingConfig(seed=2),
            pipeline=PipelineConfig(run_green=True, run_coupling=False),
        )
    with pytest.raises(ValueError, match="coupling_training.seed"):
        validate_active_training_seeds(
            training=TrainingConfig(seed=1),
            coupling_training=CouplingTrainingConfig(),
            pipeline=PipelineConfig(run_green=False, run_coupling=True),
        )
    validate_active_training_seeds(
        training=TrainingConfig(),
        coupling_training=CouplingTrainingConfig(),
        pipeline=PipelineConfig(run_green=False, run_coupling=False),
    )


def test_subseed_derivation_is_stable_and_namespaced() -> None:
    first = derive_training_subseed(17, stage="green", namespace="model")
    assert first == derive_training_subseed(17, stage="green", namespace="model")
    assert first != derive_training_subseed(17, stage="green", namespace="runtime")
    assert first != derive_training_subseed(17, stage="coupling", namespace="model")
    with pytest.raises(ValueError, match="Unsupported green seed namespace"):
        derive_training_subseed(17, stage="green", namespace="unknown")


def test_context_repeats_python_numpy_and_torch_rng() -> None:
    context = TrainingSeedContext(
        stage="green",
        base_seed=23,
        deterministic_algorithms=True,
        device="cpu",
    )
    context.apply("runtime")
    first = (random.random(), np.random.random(), torch.rand(3))
    context.apply("runtime")
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)


def test_strict_cpu_mode_enables_deterministic_algorithms() -> None:
    context = TrainingSeedContext(
        stage="green",
        base_seed=29,
        deterministic_algorithms=True,
        device="cpu",
    )
    context.configure_process()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_strict_cuda_mode_configures_cublas_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    context = TrainingSeedContext(
        stage="coupling",
        base_seed=29,
        deterministic_algorithms=True,
        device="cuda:0",
    )
    context.configure_process()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_strict_cuda_mode_rejects_late_workspace_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    context = TrainingSeedContext(
        stage="coupling",
        base_seed=29,
        deterministic_algorithms=True,
        device="cuda:0",
    )
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        context.configure_process()


def test_same_seed_repeats_shuffle_and_one_step_update() -> None:
    first_order, first_state = _one_step(31)
    second_order, second_state = _one_step(31)
    different_order, different_state = _one_step(32)

    assert first_order == second_order
    for key in first_state:
        torch.testing.assert_close(
            first_state[key], second_state[key], rtol=0.0, atol=0.0
        )
    assert first_order != different_order or any(
        not torch.equal(first_state[key], different_state[key]) for key in first_state
    )


def test_green_data_count_does_not_shift_validation_or_model_streams() -> None:
    def snapshot(train_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = TrainingSeedContext(
            stage="green",
            base_seed=41,
            deterministic_algorithms=True,
            device="cpu",
        )
        context.apply("data_train")
        _ = torch.rand(train_count)
        context.apply("data_valid")
        validation = torch.rand(8)
        context.apply("model")
        model_weight = torch.nn.Linear(3, 2).weight.detach().clone()
        return validation, model_weight

    short_validation, short_model = snapshot(4)
    long_validation, long_model = snapshot(400)
    torch.testing.assert_close(short_validation, long_validation, rtol=0.0, atol=0.0)
    torch.testing.assert_close(short_model, long_model, rtol=0.0, atol=0.0)


def test_green_rng_consumption_does_not_shift_coupling_initialization() -> None:
    def coupling_weight(*, run_green_first: bool) -> torch.Tensor:
        if run_green_first:
            green = TrainingSeedContext(
                stage="green",
                base_seed=5,
                deterministic_algorithms=True,
                device="cpu",
            )
            green.apply("data_train")
            _ = torch.rand(1000)
            green.apply("runtime")
            _ = torch.rand(1000)
        coupling = TrainingSeedContext(
            stage="coupling",
            base_seed=7,
            deterministic_algorithms=True,
            device="cpu",
        )
        coupling.apply("model")
        return torch.nn.Linear(4, 3).weight.detach().clone()

    torch.testing.assert_close(
        coupling_weight(run_green_first=False),
        coupling_weight(run_green_first=True),
        rtol=0.0,
        atol=0.0,
    )


def test_indexed_gp_identity_seed_is_independent_of_training_seed() -> None:
    expected = derive_indexed_seed(13, "train", 27)
    for training_seed in (0, 99):
        context = TrainingSeedContext(
            stage="coupling",
            base_seed=training_seed,
            deterministic_algorithms=True,
            device="cpu",
        )
        context.apply("runtime")
        assert derive_indexed_seed(13, "train", 27) == expected


def test_product_and_product_fuser_share_identical_common_parameters() -> None:
    context = TrainingSeedContext(
        stage="coupling",
        base_seed=53,
        deterministic_algorithms=True,
        device="cpu",
    )
    context.apply("model")
    product = ComplexCouplingNet(_small_coupling_config("product"))
    context.apply("model")
    product_fuser = ComplexCouplingNet(_small_coupling_config("product_fuser"))

    product_state = product.state_dict()
    fuser_state = product_fuser.state_dict()
    common_keys = sorted(set(product_state) & set(fuser_state))
    assert common_keys
    assert set(fuser_state) - set(product_state)
    for key in common_keys:
        torch.testing.assert_close(
            product_state[key], fuser_state[key], rtol=0.0, atol=0.0
        )
