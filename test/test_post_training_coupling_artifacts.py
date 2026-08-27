from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from greenonet.config import (
    ComplexCouplingSourceConfig,
    ComplexReferenceDiagnosticsConfig,
    CouplingArtifactsConfig,
    CouplingTrainingConfig,
    DatasetConfig,
    IndexedGpSourceConfig,
    PipelineConfig,
    validate_coupling_artifacts_config,
)
from greenonet.coupling_artifact_runtime import export_coupling_artifact_request
from greenonet.coupling_artifacts import CouplingArtifactRequest
from greenonet.post_training_coupling_artifacts import (
    FullReferenceTestDatasetValidator,
    PostTrainingCouplingArtifactRunner,
)


def _write_full_reference(path: Path, *, aliases: bool = False) -> None:
    values = np.zeros((2, 2), dtype=np.float64)
    fields = {"rhs": values, "sol": values}
    if aliases:
        fields.update({"uxx": values, "uyy": values})
    else:
        fields.update({"phi": values, "psi": values})
    np.savez(path, **fields)


def _valid_cross_config(
    tmp_path: Path,
) -> tuple[
    CouplingArtifactsConfig,
    DatasetConfig,
    CouplingTrainingConfig,
    PipelineConfig,
]:
    geometry = tmp_path / "geometry.npz"
    geometry.touch()
    test_path = tmp_path / "test"
    test_path.mkdir()
    _write_full_reference(test_path / "sample.npz")
    green = tmp_path / "green.safetensors"
    green.touch()
    dataset = DatasetConfig(
        geometry_mode="complex",
        geometry_path=geometry,
        test_path=test_path,
        coupling_source=ComplexCouplingSourceConfig(
            mode="indexed_gp",
            indexed_gp=IndexedGpSourceConfig(num_train=2, num_valid=1),
        ),
        reference_diagnostics=ComplexReferenceDiagnosticsConfig(
            training=False,
            validation=False,
        ),
    )
    training = CouplingTrainingConfig(
        best_energy_checkpoint={"enabled": True},
    )
    pipeline = PipelineConfig(
        run_green=False,
        run_coupling=True,
        green_pretrained_path=green,
    )
    return CouplingArtifactsConfig(enabled=True), dataset, training, pipeline


def test_full_reference_validator_accepts_both_directional_schemas(
    tmp_path: Path,
) -> None:
    _write_full_reference(tmp_path / "preferred.npz")
    _write_full_reference(tmp_path / "legacy.npz", aliases=True)

    files = FullReferenceTestDatasetValidator.validate(tmp_path)

    assert [path.name for path in files] == ["legacy.npz", "preferred.npz"]


@pytest.mark.parametrize(
    "fields",
    [
        {"rhs": np.zeros((2, 2)), "phi": np.zeros((2, 2)), "psi": np.zeros((2, 2))},
        {"rhs": np.zeros((2, 2)), "sol": np.zeros((2, 2))},
    ],
)
def test_full_reference_validator_rejects_incomplete_sample(
    tmp_path: Path,
    fields: dict[str, np.ndarray],
) -> None:
    np.savez(tmp_path / "sample.npz", **fields)

    with pytest.raises(KeyError):
        FullReferenceTestDatasetValidator.validate(tmp_path)


def test_coupling_artifact_cross_config_validation_is_opt_in(tmp_path: Path) -> None:
    validate_coupling_artifacts_config(
        artifacts=CouplingArtifactsConfig(),
        dataset=DatasetConfig(
            geometry_mode="complex",
            geometry_path=tmp_path / "missing_geometry.npz",
        ),
        coupling_training=CouplingTrainingConfig(),
        pipeline=PipelineConfig(run_green=False, run_coupling=False),
    )

    artifacts, dataset, training, pipeline = _valid_cross_config(tmp_path)
    validate_coupling_artifacts_config(
        artifacts=artifacts,
        dataset=dataset,
        coupling_training=training,
        pipeline=pipeline,
    )


def test_coupling_artifact_cross_config_requires_npz_validation_files(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "geometry.npz"
    geometry.touch()
    test_path = tmp_path / "test"
    test_path.mkdir()
    _write_full_reference(test_path / "sample.npz")
    green = tmp_path / "green.safetensors"
    green.touch()
    dataset = DatasetConfig(
        geometry_mode="complex",
        geometry_path=geometry,
        training_path=tmp_path / "train",
        validation_path=tmp_path / "missing_validation",
        test_path=test_path,
    )

    with pytest.raises(FileNotFoundError, match="missing_validation"):
        validate_coupling_artifacts_config(
            artifacts=CouplingArtifactsConfig(enabled=True),
            dataset=dataset,
            coupling_training=CouplingTrainingConfig(
                best_energy_checkpoint={"enabled": True}
            ),
            pipeline=PipelineConfig(
                run_green=False,
                run_coupling=True,
                green_pretrained_path=green,
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("run_coupling", "run_coupling"),
        ("unit_square", "complex CouplingNet"),
        ("best_energy", "best_energy_checkpoint"),
        ("green", "green_pretrained_path"),
    ],
)
def test_coupling_artifact_cross_config_rejects_incompatible_run(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    artifacts, dataset, training, pipeline = _valid_cross_config(tmp_path)
    if mutation == "run_coupling":
        pipeline = PipelineConfig(
            run_green=False,
            run_coupling=False,
            green_pretrained_path=pipeline.green_pretrained_path,
        )
    elif mutation == "unit_square":
        dataset.geometry_mode = "unit_square"
    elif mutation == "best_energy":
        training = CouplingTrainingConfig(best_energy_checkpoint={"enabled": False})
    elif mutation == "green":
        pipeline = PipelineConfig(run_green=False, run_coupling=True)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        validate_coupling_artifacts_config(
            artifacts=artifacts,
            dataset=dataset,
            coupling_training=training,
            pipeline=pipeline,
        )


def test_post_training_runner_resolves_fixed_best_energy_paths(tmp_path: Path) -> None:
    artifacts, dataset, training, pipeline = _valid_cross_config(tmp_path)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    (work_dir / "config_used.json").write_text("{}")
    (work_dir / "complex_coupling_model_best_energy.safetensors").touch()

    runner = PostTrainingCouplingArtifactRunner(
        config=artifacts,
        dataset=dataset,
        coupling_training=training,
        pipeline=pipeline,
        work_dir=work_dir,
    )
    request = runner.build_request()

    assert request.config == (work_dir / "config_used.json").resolve()
    assert (
        request.coupling_checkpoint
        == (work_dir / "complex_coupling_model_best_energy.safetensors").resolve()
    )
    assert request.green_checkpoint == Path(pipeline.green_pretrained_path).resolve()
    assert request.outdir == (work_dir / "artifacts_best_energy").resolve()
    assert request.generation_trigger == "post_training"
    assert request.checkpoint_selector == "best_energy"


def test_post_training_runner_uses_run_green_checkpoint(tmp_path: Path) -> None:
    artifacts, dataset, training, _pipeline = _valid_cross_config(tmp_path)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    for name in (
        "config_used.json",
        "complex_coupling_model_best_energy.safetensors",
        "model.safetensors",
    ):
        (work_dir / name).touch()
    pipeline = PipelineConfig(run_green=True, run_coupling=True)

    request = PostTrainingCouplingArtifactRunner(
        config=artifacts,
        dataset=dataset,
        coupling_training=training,
        pipeline=pipeline,
        work_dir=work_dir,
    ).build_request()

    assert request.green_checkpoint == (work_dir / "model.safetensors").resolve()


def test_post_training_runner_rejects_nonempty_output(tmp_path: Path) -> None:
    artifacts, dataset, training, pipeline = _valid_cross_config(tmp_path)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    (work_dir / "config_used.json").write_text("{}")
    (work_dir / "complex_coupling_model_best_energy.safetensors").touch()
    outdir = work_dir / "artifacts_best_energy"
    outdir.mkdir()
    sentinel = outdir / "keep.txt"
    sentinel.write_text("keep")

    runner = PostTrainingCouplingArtifactRunner(
        config=artifacts,
        dataset=dataset,
        coupling_training=training,
        pipeline=pipeline,
        work_dir=work_dir,
    )

    with pytest.raises(FileExistsError, match="not empty"):
        runner.build_request()
    assert sentinel.read_text() == "keep"


def test_shared_runtime_dispatches_complex_request(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": {"geometry_mode": "complex"}}))
    request = CouplingArtifactRequest(
        config=config,
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
    )
    calls: list[str] = []

    def _fake_complex(_request, *, logger):
        assert logger.name == "test"
        calls.append("complex")
        return {"selected_samples": [1]}

    monkeypatch.setattr(
        "greenonet.coupling_artifact_runtime.export_complex_coupling_artifacts",
        _fake_complex,
    )

    summary = export_coupling_artifact_request(
        request,
        logger=logging.getLogger("test"),
    )

    assert calls == ["complex"]
    assert summary["selected_samples"] == [1]
