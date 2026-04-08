from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import plotly.graph_objs as go


VALUE_RE = r"(?:nan|inf|-inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def _parse_float(value: str | None, default: float = float("nan")) -> float:
    if value is None:
        return default
    return float(value)


def _parse_bool(value: str | None) -> float:
    if value is None:
        return float("nan")
    return 1.0 if value.lower() == "true" else 0.0


def parse_log(path: Path) -> Dict[str, List[float]]:
    """Parse current CouplingNet training logs into epoch-aligned metric series."""
    pattern_epoch = re.compile(
        rf"epoch\s+(?P<epoch>\d+).*?\|\s*train\s+loss=(?P<loss_tr>{VALUE_RE})"
        rf"\s+l2_cons=(?P<l2_cons_tr>{VALUE_RE})"
        rf"\s+flux_cons=(?P<flux_cons_tr>{VALUE_RE})"
        rf"\s+cross_cons=(?P<cross_cons_tr>{VALUE_RE})"
        rf"\s+rel_flux=(?P<rflux_tr>{VALUE_RE})"
        rf"\s+rel_sol=(?P<rsol_tr>{VALUE_RE})"
        rf"\s*\|\s*w_l2=(?P<w_l2>{VALUE_RE})\s+on_l2=(?P<on_l2>True|False)"
        rf"\s+w_flux=(?P<w_flux>{VALUE_RE})\s+on_flux=(?P<on_flux>True|False)"
        rf"\s+w_cross=(?P<w_cross>{VALUE_RE})\s+on_cross=(?P<on_cross>True|False)"
        rf"(?:\s*\|\s*lr=(?P<lr>{VALUE_RE}))?"
        rf"(?:\s*\|\s*val\s+loss=(?P<loss_val>{VALUE_RE})"
        rf"\s+l2_cons=(?P<l2_cons_val>{VALUE_RE})"
        rf"\s+flux_cons=(?P<flux_cons_val>{VALUE_RE})"
        rf"\s+cross_cons=(?P<cross_cons_val>{VALUE_RE})"
        rf"\s+rel_flux=(?P<rflux_val>{VALUE_RE})"
        rf"\s+rel_sol=(?P<rsol_val>{VALUE_RE}))?",
        re.IGNORECASE,
    )

    entries: List[Dict[str, float]] = []
    for line in path.read_text().splitlines():
        match = pattern_epoch.search(line)
        if match is None:
            continue

        entries.append(
            {
                "raw_epoch": float(match.group("epoch")),
                "loss_train": _parse_float(match.group("loss_tr")),
                "l2_cons_train": _parse_float(match.group("l2_cons_tr")),
                "flux_cons_train": _parse_float(match.group("flux_cons_tr")),
                "cross_cons_train": _parse_float(match.group("cross_cons_tr")),
                "rel_flux_train": _parse_float(match.group("rflux_tr")),
                "rel_sol_train": _parse_float(match.group("rsol_tr")),
                "loss_val": _parse_float(match.group("loss_val")),
                "l2_cons_val": _parse_float(match.group("l2_cons_val")),
                "flux_cons_val": _parse_float(match.group("flux_cons_val")),
                "cross_cons_val": _parse_float(match.group("cross_cons_val")),
                "rel_flux_val": _parse_float(match.group("rflux_val")),
                "rel_sol_val": _parse_float(match.group("rsol_val")),
                "lr": _parse_float(match.group("lr")),
                "w_l2": _parse_float(match.group("w_l2")),
                "on_l2": _parse_bool(match.group("on_l2")),
                "w_flux": _parse_float(match.group("w_flux")),
                "on_flux": _parse_bool(match.group("on_flux")),
                "w_cross": _parse_float(match.group("w_cross")),
                "on_cross": _parse_bool(match.group("on_cross")),
            }
        )

    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "loss_train": [],
        "l2_cons_train": [],
        "flux_cons_train": [],
        "cross_cons_train": [],
        "rel_flux_train": [],
        "rel_sol_train": [],
        "loss_val": [],
        "l2_cons_val": [],
        "flux_cons_val": [],
        "cross_cons_val": [],
        "rel_flux_val": [],
        "rel_sol_val": [],
        "lr": [],
        "w_l2": [],
        "on_l2": [],
        "w_flux": [],
        "on_flux": [],
        "w_cross": [],
        "on_cross": [],
    }

    offset = 0.0
    last_raw = None
    last_effective = 0.0
    for entry in entries:
        raw_epoch = entry["raw_epoch"]
        if last_raw is not None and raw_epoch <= last_raw:
            offset = last_effective
        effective_epoch = raw_epoch + offset
        last_raw = raw_epoch
        last_effective = effective_epoch

        metrics["epoch"].append(effective_epoch)
        for key in metrics:
            if key == "epoch":
                continue
            metrics[key].append(entry[key])
    return metrics


def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
    output: List[float | None] = []
    for value in values:
        if value != value:
            output.append(None)
        else:
            output.append(max(value, floor))
    return output


def _has_visible_values(values: List[float | None]) -> bool:
    return any(value is not None for value in values)


def make_fig_losses(data_by_log: Dict[str, Dict[str, List[float]]], font: Dict) -> go.Figure:
    fig = go.Figure()
    colors = {
        "loss": "#1f77b4",
        "l2_cons": "#d62728",
        "flux_cons": "#2ca02c",
        "cross_cons": "#ff7f0e",
    }
    labels = {
        "loss": "Total Loss",
        "l2_cons": "L2 Consistency",
        "flux_cons": "Flux Consistency",
        "cross_cons": "Cross Consistency",
    }

    for log_name, metrics in data_by_log.items():
        epochs = metrics["epoch"]
        for key in ("loss", "l2_cons", "flux_cons", "cross_cons"):
            for split, dash in (("train", "solid"), ("val", "dot")):
                metric_key = f"{key}_{split}"
                if metric_key not in metrics:
                    continue
                y_vals = _mask_nan(metrics[metric_key])
                if not _has_visible_values(y_vals):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=y_vals,
                        mode="lines",
                        name=f"{labels[key]} ({split}) [{log_name}]",
                        line=dict(color=colors[key], dash=dash),
                        connectgaps=True,
                    )
                )
    fig.update_layout(
        title="Training vs Validation Losses",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        yaxis=dict(exponentformat="power"),
        template="plotly_white",
        font=font,
        legend=dict(
            x=1.0,
            y=0.9,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


def make_fig_errors(data_by_log: Dict[str, Dict[str, List[float]]], font: Dict) -> go.Figure:
    fig = go.Figure()
    colors = {
        "rel_flux": "#9467bd",
        "rel_sol": "#8c564b",
    }
    labels = {
        "rel_flux": "Flux-Divergence Error",
        "rel_sol": "Solution Error",
    }

    for log_name, metrics in data_by_log.items():
        epochs = metrics["epoch"]
        for key in ("rel_flux", "rel_sol"):
            for split, dash in (("train", "solid"), ("val", "dot")):
                metric_key = f"{key}_{split}"
                if metric_key not in metrics:
                    continue
                y_vals = _mask_nan(metrics[metric_key])
                if not _has_visible_values(y_vals):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=y_vals,
                        mode="lines",
                        name=f"{labels[key]} ({split}) [{log_name}]",
                        line=dict(color=colors[key], dash=dash),
                        connectgaps=True,
                    )
                )
    fig.update_layout(
        title="Training vs Validation Errors",
        xaxis_title="Epoch",
        yaxis_title="Error",
        yaxis_type="log",
        yaxis=dict(exponentformat="power"),
        template="plotly_white",
        font=font,
        legend=dict(
            x=1.0,
            y=0.8,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


_warned_static = False


def save_fig(fig: go.Figure, base_path: Path) -> None:
    global _warned_static
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        if not _warned_static:
            print("Static export skipped (requires kaleido + Chrome); HTML saved instead.")
            _warned_static = True
    fig.write_html(str(base_path.with_suffix(".html")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/validation metrics from logs.")
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to training.log files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        default="Times New Roman",
        help="Font family for the plots.",
    )
    args = parser.parse_args()

    data_by_log: Dict[str, Dict[str, List[float]]] = {}
    for log_path in args.logs:
        metrics = parse_log(log_path)
        if not metrics.get("epoch"):
            print(f"Warning: no metrics parsed from {log_path}")
            continue
        data_by_log[log_path.name] = metrics

    if not data_by_log:
        raise RuntimeError("No valid log data parsed.")

    font = {"family": args.font_family, "size": 14}

    fig_losses = make_fig_losses(data_by_log, font)
    fig_errors = make_fig_errors(data_by_log, font)

    save_fig(fig_losses, args.outdir / "losses")
    save_fig(fig_errors, args.outdir / "errors")


if __name__ == "__main__":
    main()
