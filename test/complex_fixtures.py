from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def write_geometry_npz(path: Path, **overrides: np.ndarray) -> Path:
    payload: dict[str, np.ndarray | float] = {
        "coords_valid": np.array(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]],
            dtype=np.float64,
        ),
        "valid_grid_y_index": np.array([1, 1, 3], dtype=np.int64),
        "valid_grid_x_index": np.array([1, 3, 1], dtype=np.int64),
        "x_segment_id": np.array([0, 0, 1], dtype=np.int64),
        "y_segment_id": np.array([0, 1, 0], dtype=np.int64),
        "x_local_t": np.array([0.25, 0.75, 0.5], dtype=np.float64),
        "y_local_t": np.array([0.25, 0.25, 0.75], dtype=np.float64),
        "x_segment_left": np.array([0.0, 0.0], dtype=np.float64),
        "x_segment_right": np.array([1.0, 0.5], dtype=np.float64),
        "x_segment_y": np.array([0.25, 0.75], dtype=np.float64),
        "x_segment_length": np.array([1.0, 0.5], dtype=np.float64),
        "y_segment_bottom": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "y_segment_top": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "y_segment_x": np.array([0.25, 0.75, 0.5], dtype=np.float64),
        "y_segment_length": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "x_recon_ptr": np.array([0, 4, 7], dtype=np.int64),
        "x_recon_t": np.array([0.0, 0.25, 0.75, 1.0, 0.0, 0.5, 1.0]),
        "x_recon_weight": np.array([0.125, 0.375, 0.375, 0.125, 0.25, 0.5, 0.25]),
        "x_recon_valid_index": np.array([-1, 0, 1, -1, -1, 2, -1]),
        "y_recon_ptr": np.array([0, 4, 7, 9], dtype=np.int64),
        "y_recon_t": np.array([0.0, 0.25, 0.75, 1.0, 0.0, 0.25, 1.0, 0.0, 1.0]),
        "y_recon_weight": np.array(
            [0.125, 0.375, 0.375, 0.125, 0.125, 0.5, 0.375, 0.5, 0.5]
        ),
        "y_recon_valid_index": np.array([-1, 0, 2, -1, -1, 1, -1, -1, -1]),
        "x_edges": np.array([[0, 1]], dtype=np.int64),
        "y_edges": np.array([[0, 2]], dtype=np.int64),
        "hx": np.array(0.5, dtype=np.float64),
        "hy": np.array(0.5, dtype=np.float64),
        "grid_x": np.linspace(0.0, 1.0, 5, dtype=np.float64),
        "grid_y": np.linspace(0.0, 1.0, 5, dtype=np.float64),
    }
    payload.update(overrides)
    np.savez(path, **payload)
    return path


def write_sample_npz(
    data_dir: Path,
    *,
    name: str = "sample_0000.npz",
    include_flux: bool = True,
    legacy_flux: bool = False,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    grid = np.arange(25, dtype=np.float64).reshape(5, 5)
    rhs = 1.0 + grid
    sol = 2.0 + grid
    payload: dict[str, np.ndarray] = {"rhs": rhs, "sol": sol}
    if include_flux:
        if legacy_flux:
            payload["uxx"] = 3.0 + grid
            payload["uyy"] = 4.0 + grid
        else:
            payload["phi"] = 5.0 + grid
            payload["psi"] = 6.0 + grid
    path = data_dir / name
    np.savez(path, **payload)
    return path


def write_coefficients(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return 1.0 + 0.0 * x + 0.0 * y",
                "def apx_fun(x, y): return 2.0 + 0.0 * x + 0.0 * y",
                "def apy_fun(x, y): return 3.0 + 0.0 * x + 0.0 * y",
                "def bx_fun(x, y): return 4.0 + 0.0 * x + 0.0 * y",
                "def by_fun(x, y): return 5.0 + 0.0 * x + 0.0 * y",
                "def c_fun(x, y): return 6.0 + 0.0 * x + 0.0 * y",
            ]
        )
    )
    return path


def write_complex_config(
    path: Path,
    *,
    geometry_path: Path,
    train_path: Path | None,
    test_path: Path | None,
    coefficient_path: Path,
    coupling_checkpoint: Path | None = None,
) -> Path:
    payload = {
        "dataset": {
            "geometry_mode": "complex",
            "geometry_path": str(geometry_path),
            "training_path": None if train_path is None else str(train_path),
            "test_path": None if test_path is None else str(test_path),
            "coefficient_functions_path": str(coefficient_path),
            "dtype": "float64",
        },
        "model": {
            "hidden_dim": 4,
            "depth": 1,
            "branch_input_dim": 4,
            "use_green": False,
            "dtype": "float64",
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "device": "cpu",
            "compile": {"enabled": False},
        },
        "coupling_model": {
            "branch_input_dim": 4,
            "hidden_dim": 4,
            "depth": 1,
            "activation": "tanh",
            "dropout": 0.0,
            "dtype": "float64",
            "balance_projection": {
                "enabled": True,
                "mode": "response_space",
            },
            "axis_1d_trunk": {
                "enabled": True,
                "num_frequencies": 2,
                "max_frequency": 2.0,
                "transverse_trunk": {
                    "enabled": True,
                    "fusion": "product",
                    "length_context": True,
                },
            },
        },
        "coupling_training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "log_interval": 1,
            "device": "cpu",
            "integration_rule": "trapezoid",
            "compile": {"enabled": False},
            "length_jump_balance": {
                "enabled": True,
                "log_sigma_jump_threshold": 0.6931471805599453,
                "transition_fraction": 0.5,
                "eps": 1e-12,
            },
            "best_energy_checkpoint": {"enabled": True},
            "best_rel_sol_checkpoint": {"enabled": False},
            "losses": {
                "cross_consistency": {"enabled": True, "weight": 99.0},
                "balance_loss": {"enabled": True, "weight": 99.0},
            },
        },
        "pipeline": {
            "run_green": False,
            "run_coupling": True,
            "green_pretrained_path": "green.safetensors",
            "coupling_pretrained_path": (
                None if coupling_checkpoint is None else str(coupling_checkpoint)
            ),
        },
    }
    path.write_text(json.dumps(payload))
    return path


class ConstantGreen(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value

    def forward_pairs(
        self,
        trunk_coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> torch.Tensor:
        del a_vals, ap_vals, b_vals, c_vals
        return torch.full(
            (1, *trunk_coords.shape[:-1]),
            self.value,
            dtype=trunk_coords.dtype,
            device=trunk_coords.device,
        )
