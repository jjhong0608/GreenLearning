import logging
from pathlib import Path

import torch

from greenonet.axial import make_square_axial_lines
from greenonet.backward_sampler import BackwardSampler
from greenonet.config import CompileConfig, TrainingConfig
from greenonet.runner import run_green_o_net
from greenonet.sampler import ForwardSampler, TrainingData


def a_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2.0 + torch.cos(torch.pi * x) * torch.cos(torch.pi * y)


def apx_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -torch.cos(torch.pi * y) * torch.pi * torch.sin(torch.pi * x)


def apy_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -torch.cos(torch.pi * x) * torch.pi * torch.sin(torch.pi * y)


def zero_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


class TestAxialLines:
    def test_square_layout(self) -> None:
        lines = make_square_axial_lines(step_size=0.5)
        # With interior-only lines, only one interior line remains per axis for step_size=0.5
        assert len(lines.xaxial_lines) == 1
        assert len(lines.yaxial_lines) == 1
        for line in lines.xaxial_lines:
            assert line.x_coordinates.numel() == 3
            assert torch.allclose(
                line.coordinates[:, 1],
                torch.full(
                    (3,),
                    line.y_coordinate,
                    dtype=line.coordinates.dtype,
                ),
            )
        for line in lines.yaxial_lines:
            assert line.y_coordinates.numel() == 3
            assert torch.allclose(
                line.coordinates[:, 0],
                torch.full(
                    (3,),
                    line.x_coordinate,
                    dtype=line.coordinates.dtype,
                ),
            )


class TestForwardSampler:
    def test_generate_dataset_shapes(self) -> None:
        lines = make_square_axial_lines(step_size=0.5)
        sampler = ForwardSampler(
            axial_lines=lines,
            data_size_per_each_line=2,
            scale_length=0.1,
            deterministic=True,
        )
        data = sampler.generate_dataset(
            a_fun=a_fun,
            ap_fun=apx_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
        )
        point_count = lines.xaxial_lines[0].x_coordinates.numel()
        n_lines = len(lines.xaxial_lines)
        assert data.U.shape == (2, 2, n_lines, point_count)
        assert data.F.shape == (2, 2, n_lines, point_count)
        assert torch.isfinite(data.U).all()
        assert torch.isfinite(data.F).all()
        energy = sampler._simpson_integral(
            lines.xaxial_lines[0].x_coordinates, data.U[0, 0, 0] ** 2
        )
        assert energy > 0.0


class TestRunGreenONet:
    def test_passes_b_c_functions(self, tmp_path: Path, monkeypatch) -> None:
        captured = {}

        def b_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return 0.25 + 0.0 * x + 0.0 * y

        def c_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return 0.5 + 0.0 * x + 0.0 * y

        def fake_generate_dataset(self, **kwargs):
            captured["b_fun"] = kwargs["b_fun"]
            captured["c_fun"] = kwargs["c_fun"]

            n_lines = len(self.axial_lines.xaxial_lines)
            x_coords = self.axial_lines.xaxial_lines[0].x_coordinates
            y_coords = self.axial_lines.yaxial_lines[0].y_coordinates
            m_points = x_coords.numel()

            coords_x = torch.stack(
                [line.coordinates for line in self.axial_lines.xaxial_lines], dim=0
            )
            coords_y = torch.stack(
                [line.coordinates for line in self.axial_lines.yaxial_lines], dim=0
            )
            coords = torch.stack((coords_x, coords_y), dim=0)

            zeros = torch.zeros((1, 2, n_lines, m_points), dtype=torch.float64)
            return TrainingData(
                X=x_coords,
                Y=y_coords,
                U=zeros,
                F=zeros,
                A=zeros,
                AP=zeros,
                B=zeros,
                C=zeros,
                COORDS=coords,
            )

        def fake_train(self, dataset, validation_dataset=None):
            del validation_dataset
            return None

        monkeypatch.setattr(ForwardSampler, "generate_dataset", fake_generate_dataset)
        monkeypatch.setattr("greenonet.trainer.Trainer.train", fake_train)

        run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=b_fun,
            c_fun=c_fun,
            activation="tanh",
            work_dir=tmp_path / "run_b_c",
            ndata=1,
            seed=0,
            scale_length=0.1,
            use_operator_learning=True,
            deterministic=True,
        )

        assert captured["b_fun"] is b_fun
        assert captured["c_fun"] is c_fun
        logging.getLogger("GreenONetRunner").handlers.clear()
        logging.getLogger("Trainer").handlers.clear()

    def test_compile_enabled_calls_torch_compile(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        compiled = {"count": 0}

        def fake_compile(model):
            compiled["count"] += 1
            return model

        def fake_train(self, dataset, validation_dataset=None):
            del dataset, validation_dataset
            return None

        monkeypatch.setattr(torch, "compile", fake_compile)
        monkeypatch.setattr("greenonet.trainer.Trainer.train", fake_train)

        run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
            activation="tanh",
            work_dir=tmp_path / "run_compile",
            ndata=1,
            seed=0,
            scale_length=0.1,
            use_operator_learning=True,
            deterministic=True,
            training_cfg=TrainingConfig(
                epochs=1,
                batch_size=4,
                compile=CompileConfig(enabled=True),
            ),
        )

        assert compiled["count"] == 1

    def test_end_to_end_run(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        work_dir = tmp_path / "run"
        result = run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
            activation="tanh",
            work_dir=work_dir,
            ndata=2,
            seed=0,
            scale_length=0.1,
            use_operator_learning=True,
            deterministic=True,
        )
        assert work_dir.exists()
        assert (work_dir / "training.log").exists()
        assert (work_dir / "loss_curve.html").exists()
        assert (work_dir / "model.safetensors").exists()
        assert result.loss_history, "Trainer should record loss values"

    def test_sampler_mode_backward_routes_sampler(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        calls = {"forward": 0, "backward": 0}

        def _fake_training_data() -> TrainingData:
            lines = make_square_axial_lines(step_size=0.5, n_points_per_line=5)
            n_lines = len(lines.xaxial_lines)
            m_points = lines.xaxial_lines[0].x_coordinates.numel()
            coords_x = torch.stack(
                [line.coordinates for line in lines.xaxial_lines], dim=0
            )
            coords_y = torch.stack(
                [line.coordinates for line in lines.yaxial_lines], dim=0
            )
            coords = torch.stack((coords_x, coords_y), dim=0)
            zeros = torch.zeros((1, 2, n_lines, m_points), dtype=torch.float64)
            return TrainingData(
                X=lines.xaxial_lines[0].x_coordinates,
                Y=lines.yaxial_lines[0].y_coordinates,
                U=zeros,
                F=zeros,
                A=zeros,
                AP=zeros,
                B=zeros,
                C=zeros,
                COORDS=coords,
            )

        def fake_forward_generate_dataset(self, **kwargs):
            calls["forward"] += 1
            return _fake_training_data()

        def fake_backward_generate_dataset(self, **kwargs):
            calls["backward"] += 1
            return _fake_training_data()

        def fake_train(self, dataset, validation_dataset=None):
            del validation_dataset
            return None

        monkeypatch.setattr(
            ForwardSampler, "generate_dataset", fake_forward_generate_dataset
        )
        monkeypatch.setattr(
            BackwardSampler, "generate_dataset", fake_backward_generate_dataset
        )
        monkeypatch.setattr("greenonet.trainer.Trainer.train", fake_train)

        run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
            activation="tanh",
            work_dir=tmp_path / "run_backward_sampler",
            ndata=1,
            seed=0,
            scale_length=0.1,
            use_operator_learning=True,
            deterministic=True,
            sampler_mode="backward",
        )
        assert calls["backward"] == 1
        assert calls["forward"] == 0

    def test_green_validation_dataset_uses_separate_count_and_scale(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        sampler_calls: list[tuple[int, float | tuple[float, float]]] = []
        train_capture: dict[str, int] = {}

        def _fake_training_data(sample_count: int) -> TrainingData:
            lines = make_square_axial_lines(step_size=0.5, n_points_per_line=5)
            n_lines = len(lines.xaxial_lines)
            m_points = lines.xaxial_lines[0].x_coordinates.numel()
            coords_x = torch.stack(
                [line.coordinates for line in lines.xaxial_lines], dim=0
            )
            coords_y = torch.stack(
                [line.coordinates for line in lines.yaxial_lines], dim=0
            )
            coords = torch.stack((coords_x, coords_y), dim=0)
            zeros = torch.zeros(
                (sample_count, 2, n_lines, m_points), dtype=torch.float64
            )
            return TrainingData(
                X=lines.xaxial_lines[0].x_coordinates,
                Y=lines.yaxial_lines[0].y_coordinates,
                U=zeros,
                F=zeros,
                A=zeros,
                AP=zeros,
                B=zeros,
                C=zeros,
                COORDS=coords,
            )

        def fake_generate_dataset(self, **kwargs):
            del kwargs
            sampler_calls.append((self.data_size_per_each_line, self.scale_length))
            return _fake_training_data(self.data_size_per_each_line)

        def fake_train(self, dataset, validation_dataset=None):
            train_capture["train_len"] = len(dataset)
            train_capture["validation_len"] = (
                len(validation_dataset) if validation_dataset is not None else -1
            )
            return None

        monkeypatch.setattr(ForwardSampler, "generate_dataset", fake_generate_dataset)
        monkeypatch.setattr("greenonet.trainer.Trainer.train", fake_train)

        run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
            activation="tanh",
            work_dir=tmp_path / "run_green_validation",
            ndata=3,
            validation_ndata=2,
            seed=0,
            scale_length=(0.05, 0.25),
            validation_scale_length=(0.10, 0.20),
            use_operator_learning=True,
            deterministic=True,
            training_cfg=TrainingConfig(compute_validation_rel_sol=True),
        )

        assert sampler_calls == [
            (3, (0.05, 0.25)),
            (2, (0.10, 0.20)),
        ]
        assert train_capture["train_len"] == 3
        assert train_capture["validation_len"] == 2

    def test_green_validation_dataset_can_use_different_sampler_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        calls = {"forward": 0, "backward": 0}
        train_capture: dict[str, int] = {}

        def _fake_training_data(sample_count: int) -> TrainingData:
            lines = make_square_axial_lines(step_size=0.5, n_points_per_line=5)
            n_lines = len(lines.xaxial_lines)
            m_points = lines.xaxial_lines[0].x_coordinates.numel()
            coords_x = torch.stack(
                [line.coordinates for line in lines.xaxial_lines], dim=0
            )
            coords_y = torch.stack(
                [line.coordinates for line in lines.yaxial_lines], dim=0
            )
            coords = torch.stack((coords_x, coords_y), dim=0)
            zeros = torch.zeros(
                (sample_count, 2, n_lines, m_points), dtype=torch.float64
            )
            return TrainingData(
                X=lines.xaxial_lines[0].x_coordinates,
                Y=lines.yaxial_lines[0].y_coordinates,
                U=zeros,
                F=zeros,
                A=zeros,
                AP=zeros,
                B=zeros,
                C=zeros,
                COORDS=coords,
            )

        def fake_forward_generate_dataset(self, **kwargs):
            del kwargs
            calls["forward"] += 1
            return _fake_training_data(self.data_size_per_each_line)

        def fake_backward_generate_dataset(self, **kwargs):
            del kwargs
            calls["backward"] += 1
            return _fake_training_data(self.data_size_per_each_line)

        def fake_train(self, dataset, validation_dataset=None):
            train_capture["train_len"] = len(dataset)
            train_capture["validation_len"] = (
                len(validation_dataset) if validation_dataset is not None else -1
            )
            return None

        monkeypatch.setattr(
            ForwardSampler, "generate_dataset", fake_forward_generate_dataset
        )
        monkeypatch.setattr(
            BackwardSampler, "generate_dataset", fake_backward_generate_dataset
        )
        monkeypatch.setattr("greenonet.trainer.Trainer.train", fake_train)

        run_green_o_net(
            a_fun=a_fun,
            apx_fun=apx_fun,
            apy_fun=apy_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
            activation="tanh",
            work_dir=tmp_path / "run_green_validation_mode_split",
            ndata=3,
            validation_ndata=2,
            seed=0,
            scale_length=0.1,
            validation_scale_length=0.2,
            use_operator_learning=True,
            deterministic=True,
            sampler_mode="forward",
            validation_sampler_mode="backward",
            training_cfg=TrainingConfig(compute_validation_rel_sol=True),
        )

        assert calls["forward"] == 1
        assert calls["backward"] == 1
        assert train_capture["train_len"] == 3
        assert train_capture["validation_len"] == 2
