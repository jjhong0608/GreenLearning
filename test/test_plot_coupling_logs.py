from pathlib import Path

import plotly.graph_objs as go
import plotly.io as pio

from plot_coupling_logs import make_fig_loss, make_fig_metric, parse_log, save_fig


def test_parse_log_epochs_and_metrics(tmp_path: Path) -> None:
    log_text = "\n".join(
        [
            "Epoch 1 | train loss=1.0 cons=1.0 rel_flux=2.0 rel_sol=3.0 | val cons=1.5 rel_flux=2.5 rel_sol=3.5",
            "Epoch 2 | train loss=0.5 cons=0.5 rel_flux=1.5 rel_sol=2.5 | val cons=1.0 rel_flux=2.0 rel_sol=3.0",
            "Coupling LBFGS epoch 1 last loss: 0.4 | train cons=0.4 rel_flux=1.4 rel_sol=2.4 | val cons=0.9 rel_flux=1.9 rel_sol=2.9",
        ]
    )
    path = tmp_path / "training.log"
    path.write_text(log_text)

    metrics = parse_log(path)
    assert metrics["epoch"] == [1.0, 2.0, 3.0]
    assert metrics["loss_train"] == [1.0, 0.5, 0.4]
    assert metrics["loss_val"][0] != metrics["loss_val"][0]
    assert all(value != value for value in metrics["l2_cons_train"])
    assert metrics["energy_cons_train"] == [1.0, 0.5, 0.4]
    assert metrics["rel_flux_val"] == [2.5, 2.0, 1.9]


def test_parse_log_current_coupling_format_with_train_and_val_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.log"
    path.write_text(
        "\n".join(
            [
                "maybe_compile_model - Compiling CouplingNet with torch.compile",
                (
                    "_run_training_phase - epoch 1 | train loss=2.6421e-02 "
                    "l2_cons=3.4491e-04 energy_cons=2.6421e-02 "
                    "cross_cons=4.0042e+00 balance_loss=1.0000e-03 "
                    "symmetric_boundary_loss=3.0000e-03 rel_flux=2.8326e-01 "
                    "rel_sol=3.8143e-01 | w_l2=1.0000e+00 on_l2=False "
                    "w_energy=1.0000e+00 on_energy=True w_cross=1.0000e+00 "
                    "on_cross=False | lr=6.6667e-04 smooth_mask_diff_power=1.0000e+00 "
                    "| val loss=2.7298e-02 l2_cons=3.5841e-04 "
                    "energy_cons=2.7298e-02 cross_cons=4.1362e+00 "
                    "balance_loss=2.0000e-03 "
                    "symmetric_boundary_loss=4.0000e-03 rel_flux=2.8763e-01 "
                    "rel_sol=3.8722e-01"
                ),
                (
                    "_run_training_phase - epoch 2 | train loss=2.6358e-02 "
                    "l2_cons=3.4396e-04 energy_cons=2.6358e-02 "
                    "cross_cons=3.9972e+00 rel_flux=2.8305e-01 "
                    "rel_sol=3.8113e-01 | w_l2=1.0000e+00 on_l2=False "
                    "w_energy=1.0000e+00 on_energy=True w_cross=1.0000e+00 "
                    "on_cross=False | lr=1.3333e-03 smooth_mask_diff_power=1.0000e+00 "
                    "| val loss=2.7019e-02 l2_cons=3.5399e-04 "
                    "energy_cons=2.7019e-02 cross_cons=4.1079e+00 "
                    "rel_flux=2.8609e-01 rel_sol=3.8492e-01"
                ),
            ]
        )
    )

    metrics = parse_log(path)

    assert metrics["epoch"] == [1.0, 2.0]
    assert metrics["loss_train"] == [2.6421e-02, 2.6358e-02]
    assert metrics["loss_val"] == [2.7298e-02, 2.7019e-02]
    assert metrics["l2_cons_train"] == [3.4491e-04, 3.4396e-04]
    assert metrics["l2_cons_val"] == [3.5841e-04, 3.5399e-04]
    assert metrics["energy_cons_train"] == [2.6421e-02, 2.6358e-02]
    assert metrics["energy_cons_val"] == [2.7298e-02, 2.7019e-02]
    assert metrics["rel_flux_train"] == [2.8326e-01, 2.8305e-01]
    assert metrics["rel_sol_val"] == [3.8722e-01, 3.8492e-01]


def test_coupling_figures_apply_plotly_theme() -> None:
    metrics = {
        "epoch": [1.0, 2.0],
        "loss_train": [1.2, 0.7],
        "loss_val": [1.5, 0.9],
        "l2_cons_train": [0.8, 0.4],
        "l2_cons_val": [0.9, 0.45],
        "energy_cons_train": [1.0, 0.5],
        "energy_cons_val": [1.5, 0.8],
        "rel_sol_train": [0.3, 0.2],
        "rel_sol_val": [0.4, 0.25],
    }
    series = [("run", metrics)]
    font = {"family": "Times New Roman", "size": 14}

    fig_loss = make_fig_loss(series, font, "plotly_dark")
    fig_error = make_fig_metric(
        series,
        metric_key="rel_sol",
        title="Solution Error",
        yaxis_title="Error",
        log_scale=True,
        font=font,
        theme="plotly_dark",
    )

    assert fig_loss.layout.template == pio.templates["plotly_dark"]
    assert fig_error.layout.template == pio.templates["plotly_dark"]
    assert [trace.name for trace in fig_loss.data] == [
        "run Loss (train)",
        "run Loss (val)",
    ]
    assert len(fig_loss.layout.annotations) == 0


def test_coupling_figures_optionally_annotate_last_and_min_values() -> None:
    metrics = {
        "epoch": [1.0, 2.0, 3.0],
        "rel_sol_train": [0.5, 0.2, 0.3],
        "rel_sol_val": [0.7, 0.4, 0.1],
    }
    series = [("run", metrics)]
    font = {"family": "Times New Roman", "size": 14}

    fig = make_fig_metric(
        series,
        metric_key="rel_sol",
        title="Solution Error",
        yaxis_title="Error",
        log_scale=True,
        font=font,
        theme="plotly_white",
        show_annotations=True,
    )

    annotation_texts = [annotation.text for annotation in fig.layout.annotations]
    assert any("run (train)<br>last 3.000e-01" == text for text in annotation_texts)
    assert any("run (train)<br>min 2.000e-01" == text for text in annotation_texts)
    assert any("run (val)<br>last/min 1.000e-01" == text for text in annotation_texts)


def test_save_fig_writes_json_when_static_export_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_write_image(self: go.Figure, path: str) -> None:
        del self, path
        raise RuntimeError("no static backend")

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)
    fig = go.Figure(data=[go.Scatter(x=[1.0, 2.0], y=[3.0, 4.0])])

    save_fig(fig, tmp_path / "loss")

    assert (tmp_path / "loss.html").exists()
    assert (tmp_path / "loss.json").exists()
    assert not (tmp_path / "loss.png").exists()
    assert not (tmp_path / "loss.pdf").exists()
