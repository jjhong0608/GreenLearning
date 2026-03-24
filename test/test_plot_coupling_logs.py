from pathlib import Path

from plot_coupling_logs import parse_log


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
    assert metrics["cons_train"] == [1.0, 0.5, 0.4]
    assert metrics["rel_flux_val"] == [2.5, 2.0, 1.9]
