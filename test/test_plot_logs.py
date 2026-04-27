from pathlib import Path

from plot_logs import make_fig_errors, make_fig_losses, parse_log


def test_parse_log_current_coupling_format_and_ignore_noise(tmp_path: Path) -> None:
    log_text = "\n".join(
        [
            "maybe_compile_model - Compiling CouplingNet with torch.compile",
            "save_model_with_config - Saved model+config to run/coupling_model_adam_best_rel_sol.safetensors",
            "_run_training_phase - epoch 1 | train loss=9.4756e-03 l2_cons=6.0632e-03 energy_cons=3.4125e-01 cross_cons=7.5000e-02 rel_flux=1.0129e+00 rel_sol=1.5867e+00 | w_l2=1.0000e+00 on_l2=True w_energy=1.0000e-02 on_energy=True w_cross=5.0000e-01 on_cross=False | lr=1.0000e-03 | val loss=1.0300e-02 l2_cons=6.7485e-03 energy_cons=3.5588e-01 cross_cons=8.2000e-02 rel_flux=1.0224e+00 rel_sol=1.5545e+00",
            "_run_training_phase - epoch 1 | train loss=8.0000e-03 l2_cons=5.0000e-03 energy_cons=3.0000e-01 cross_cons=4.0000e-02 rel_flux=9.0000e-01 rel_sol=1.4000e+00 | w_l2=1.0000e+00 on_l2=True w_energy=2.0000e-02 on_energy=False w_cross=2.5000e-01 on_cross=True | lr=9.0000e-04 | val loss=4.0000e-03 l2_cons=4.0000e-03 energy_cons=2.0000e-01 cross_cons=1.0000e-01 rel_flux=8.0000e-01 rel_sol=1.2000e+00",
        ]
    )
    path = tmp_path / "training.log"
    path.write_text(log_text)

    metrics = parse_log(path)

    assert metrics["epoch"] == [1.0, 2.0]
    assert metrics["loss_train"] == [9.4756e-03, 8.0000e-03]
    assert metrics["l2_cons_train"] == [6.0632e-03, 5.0000e-03]
    assert metrics["energy_cons_train"] == [3.4125e-01, 3.0000e-01]
    assert metrics["cross_cons_train"] == [7.5000e-02, 4.0000e-02]
    assert metrics["rel_flux_val"] == [1.0224e00, 8.0000e-01]
    assert metrics["lr"] == [1.0000e-03, 9.0000e-04]
    assert metrics["w_energy"] == [1.0000e-02, 2.0000e-02]
    assert metrics["on_energy"] == [1.0, 0.0]
    assert metrics["w_cross"] == [5.0000e-01, 2.5000e-01]
    assert metrics["on_cross"] == [0.0, 1.0]
    assert metrics["loss_val"] == [1.0300e-02, 4.0000e-03]


def test_plot_logs_figures_accept_current_metrics(tmp_path: Path) -> None:
    path = tmp_path / "training.log"
    path.write_text(
        "_run_training_phase - epoch 1 | train loss=1.0e-02 l2_cons=6.0e-03 energy_cons=4.0e-01 cross_cons=2.0e-01 rel_flux=1.0e+00 rel_sol=1.5e+00 | w_l2=1.0e+00 on_l2=True w_energy=1.0e-02 on_energy=True w_cross=1.0e+00 on_cross=True | lr=1.0e-03 | val loss=1.3e-02 l2_cons=7.0e-03 energy_cons=3.0e-01 cross_cons=3.0e-01 rel_flux=1.1e+00 rel_sol=1.6e+00\n"
    )

    metrics = parse_log(path)
    fig_losses = make_fig_losses(
        {path.name: metrics}, {"family": "Times New Roman", "size": 14}
    )
    fig_errors = make_fig_errors(
        {path.name: metrics}, {"family": "Times New Roman", "size": 14}
    )

    assert len(fig_losses.data) == 8
    assert len(fig_errors.data) == 4
