import math
from pathlib import Path

import plotly.io as pio

from plot_green_logs import make_fig, parse_green_log


def test_parse_green_log_current_adam_and_lbfgs_metrics(tmp_path: Path) -> None:
    path = tmp_path / "training.log"
    path.write_text(
        "\n".join(
            [
                "maybe_compile_model - Compiling GreenONetModel with torch.compile",
                (
                    "train - Epoch 1: loss=3.8542e-04 | train_rel_sol=1.0351e-02 "
                    "| val_rel_sol=1.0187e-02 | rel_green=1.1128e-02"
                ),
                (
                    "train - Epoch 2: loss=1.1291e-04 | train_rel_sol=3.2557e-03 "
                    "| val_rel_sol=3.2120e-03 | rel_green=3.4329e-03"
                ),
                "train - Starting LBFGS fine-tuning (epochs=100, max_iter=1000, lr=1.0)",
                (
                    "train - LBFGS epoch 1 last loss: 4.5453e-09 "
                    "| train_rel_sol=3.4077e-05 | val_rel_sol=3.5096e-05 "
                    "| rel_green=3.5651e-04"
                ),
            ]
        )
    )

    metrics = parse_green_log(path)

    assert metrics["epoch"] == [1, 2, 3]
    assert metrics["loss"] == [3.8542e-04, 1.1291e-04, 4.5453e-09]
    assert metrics["train_rel_sol"] == [1.0351e-02, 3.2557e-03, 3.4077e-05]
    assert metrics["val_rel_sol"] == [1.0187e-02, 3.2120e-03, 3.5096e-05]
    assert metrics["rel_green"] == [1.1128e-02, 3.4329e-03, 3.5651e-04]


def test_parse_green_log_legacy_rel_sol_maps_to_train_rel_sol(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.log"
    path.write_text(
        "\n".join(
            [
                "train - Epoch 1: loss=1.0000e-02 | rel_sol=2.0000e-01 | rel_green=3.0000e-01",
                "train - LBFGS epoch 1 last loss: 4.0000e-03 | rel_sol=5.0000e-02 | rel_green=6.0000e-02",
            ]
        )
    )

    metrics = parse_green_log(path)

    assert metrics["epoch"] == [1, 2]
    assert metrics["loss"] == [1.0000e-02, 4.0000e-03]
    assert metrics["train_rel_sol"] == [2.0000e-01, 5.0000e-02]
    assert math.isnan(metrics["val_rel_sol"][0])
    assert math.isnan(metrics["val_rel_sol"][1])
    assert metrics["rel_green"] == [3.0000e-01, 6.0000e-02]


def test_make_fig_applies_plotly_theme() -> None:
    fig = make_fig(
        "loss",
        "Training Loss",
        {"run": {"epoch": [1, 2], "loss": [1.0, 0.5]}},
        {"family": "Times New Roman", "size": 14},
        "plotly_dark",
    )

    assert fig.layout.template == pio.templates["plotly_dark"]
    assert fig.layout.yaxis.type == "log"
