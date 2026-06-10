from __future__ import annotations

import argparse
import re
import math
from pathlib import Path
from typing import Dict, List

import plotly.graph_objs as go


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOKEN_PATTERN = re.compile(
    rf"\|\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>{NUMBER_PATTERN})"
)
LOG_Y_FLOOR = 1e-16


def _parse_metric_tokens(text: str) -> Dict[str, float]:
    return {m.group("key"): float(m.group("value")) for m in TOKEN_PATTERN.finditer(text)}


def parse_green_log(path: Path) -> Dict[str, List[float]]:
    """Parse GreenONet training log for loss and Green reconstruction metrics."""
    pattern_epoch = re.compile(
        rf"Epoch\s+(?P<epoch>\d+):\s*loss=(?P<loss>{NUMBER_PATTERN})(?P<rest>.*)",
        re.IGNORECASE,
    )
    pattern_lbfgs = re.compile(
        rf"LBFGS\s+epoch\s+(?P<epoch>\d+)\s+last\s+loss:\s*"
        rf"(?P<loss>{NUMBER_PATTERN})(?P<rest>.*)",
        re.IGNORECASE,
    )
    entries: List[Dict[str, float]] = []
    last_adam = 0

    for line in path.read_text().splitlines():
        m = pattern_epoch.search(line)
        if m:
            epoch = int(m.group("epoch"))
            last_adam = max(last_adam, epoch)
            tokens = _parse_metric_tokens(m.group("rest"))
            train_rel_sol = tokens.get("train_rel_sol", tokens.get("rel_sol", float("nan")))
            entries.append(
                {
                    "epoch": epoch,
                    "loss": float(m.group("loss")),
                    "train_rel_sol": train_rel_sol,
                    "val_rel_sol": tokens.get("val_rel_sol", float("nan")),
                    "rel_green": tokens.get("rel_green", float("nan")),
                }
            )
            continue
        m2 = pattern_lbfgs.search(line)
        if m2:
            epoch = last_adam + int(m2.group("epoch"))
            tokens = _parse_metric_tokens(m2.group("rest"))
            train_rel_sol = tokens.get("train_rel_sol", tokens.get("rel_sol", float("nan")))
            entries.append(
                {
                    "epoch": epoch,
                    "loss": float(m2.group("loss")),
                    "train_rel_sol": train_rel_sol,
                    "val_rel_sol": tokens.get("val_rel_sol", float("nan")),
                    "rel_green": tokens.get("rel_green", float("nan")),
                }
            )

    entries = sorted(entries, key=lambda e: e["epoch"])
    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "loss": [],
        "train_rel_sol": [],
        "val_rel_sol": [],
        "rel_green": [],
    }
    for e in entries:
        metrics["epoch"].append(e["epoch"])
        metrics["loss"].append(e["loss"])
        metrics["train_rel_sol"].append(e["train_rel_sol"])
        metrics["val_rel_sol"].append(e["val_rel_sol"])
        metrics["rel_green"].append(e["rel_green"])
    return metrics


def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        if v != v:
            out.append(None)
        else:
            out.append(max(v, floor))
    return out


def _finite_points(
    epochs: List[float],
    values: List[float],
) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for idx, (epoch, value) in enumerate(zip(epochs, values)):
        if math.isfinite(epoch) and math.isfinite(value):
            points.append((idx, epoch, value))
    return points


def _format_annotation_value(value: float) -> str:
    return f"{value:.3e}"


def _annotation_y(value: float, *, log_scale: bool) -> float:
    del log_scale
    return max(value, LOG_Y_FLOOR)


def _annotation_axis_y(value: float, *, log_scale: bool) -> float:
    value = max(value, LOG_Y_FLOOR)
    return math.log10(value) if log_scale else value


def _annotation_offsets(split: str) -> tuple[int, int, str]:
    is_val = split == "val"
    return (46, 30 if is_val else -30, "left")


def _add_point_annotation(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    split: str,
) -> None:
    ax, ay, xanchor = _annotation_offsets(split)
    fig.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1,
        arrowcolor=color,
        ax=ax,
        ay=ay,
        xanchor=xanchor,
        yanchor="middle",
        align="center",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=10, color=color),
    )


def _add_last_annotation(
    fig: go.Figure,
    *,
    epochs: List[float],
    values: List[float],
    color: str,
    marker_label: str,
    annotation_label: str,
    split: str,
    log_scale: bool,
) -> None:
    points = _finite_points(epochs, values)
    if not points:
        return

    _, last_epoch, last_value = points[-1]
    y_values = [_annotation_y(last_value, log_scale=log_scale)]
    fig.add_trace(
        go.Scatter(
            x=[last_epoch],
            y=y_values,
            mode="markers",
            marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2)),
            name=f"{marker_label} marker",
            text=[f"{marker_label}<br>last {_format_annotation_value(last_value)}<br>epoch {last_epoch:g}"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    _add_point_annotation(
        fig,
        x=last_epoch,
        y=_annotation_axis_y(last_value, log_scale=log_scale),
        text=(
            f"{annotation_label} last<br>"
            f"{_format_annotation_value(last_value)}<br>"
            f"ep {last_epoch:g}"
        ),
        color=color,
        split=split,
    )


def _xaxis_config(
    series: List[tuple[str, Dict[str, List[float]]]],
    show_annotations: bool,
) -> dict[str, object]:
    config: dict[str, object] = {"title": "Epoch"}
    if not show_annotations:
        return config

    epochs = [
        epoch
        for _, metrics in series
        for epoch in metrics.get("epoch", [])
        if math.isfinite(epoch)
    ]
    if not epochs:
        return config
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    span = max(max_epoch - min_epoch, 1.0)
    config["range"] = [min_epoch, max_epoch + 0.12 * span]
    return config


def make_fig(
    metric_key: str,
    label: str,
    data_by_log: Dict[str, Dict[str, List[float]]],
    font: Dict[str, object],
    theme: str,
    show_annotations: bool = False,
) -> go.Figure:
    fig = go.Figure()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for idx, (log_name, metrics) in enumerate(data_by_log.items()):
        epochs = metrics["epoch"]
        y_vals = _mask_nan(metrics[metric_key])
        split = "val" if metric_key.startswith("val_") else "train"
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=y_vals,
                mode="lines",
                name=f"{log_name}",
                line=dict(color=colors[idx % len(colors)]),
                connectgaps=True,
            )
        )
        if show_annotations:
            annotation_label = (
                metric_key if len(data_by_log) == 1 else f"{log_name} {metric_key}"
            )
            _add_last_annotation(
                fig,
                epochs=epochs,
                values=metrics[metric_key],
                color=colors[idx % len(colors)],
                marker_label=log_name,
                annotation_label=annotation_label,
                split=split,
                log_scale=True,
            )
    fig.update_layout(
        title=label,
        xaxis_title="Epoch",
        yaxis_title=label,
        yaxis_type="log",
        xaxis=_xaxis_config(list(data_by_log.items()), show_annotations),
        yaxis=dict(exponentformat="power"),
        template=theme,
        font=font,
        legend=dict(
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


_warned_static = False


def save_fig(fig: go.Figure, base_path: Path) -> None:
    global _warned_static
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(base_path.with_suffix(".html")))
    fig.write_json(str(base_path.with_suffix(".json")), pretty=True)
    try:
        fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        if not _warned_static:
            print(
                "Static export skipped (requires kaleido + Chrome); "
                "HTML/JSON saved instead."
            )
            _warned_static = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GreenONet training metrics.")
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="Paths to training.log files.")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        help="Optional labels for each log (same order as --logs).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("plots_green"), help="Output directory for figures.")
    parser.add_argument(
        "--font-family",
        type=str,
        default="Times New Roman",
        help="Font family for the plots.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="plotly_white",
        help="Plotly template name for the plots.",
    )
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        help="Annotate each trace with its last value.",
    )
    args = parser.parse_args()

    data_by_log: Dict[str, Dict[str, List[float]]] = {}
    for idx, log_path in enumerate(args.logs):
        metrics = parse_green_log(log_path)
        if not metrics.get("epoch"):
            print(f"Warning: no metrics parsed from {log_path}")
            continue
        if args.labels and idx < len(args.labels):
            key = args.labels[idx]
        else:
            key = f"{log_path.parent.name}/{log_path.name}"
        if key in data_by_log:
            key = f"{key}#{idx}"
        data_by_log[key] = metrics
    if not data_by_log:
        raise RuntimeError("No valid log data parsed.")

    font = {"family": args.font_family, "size": 20}

    fig_loss = make_fig(
        "loss",
        "Training Loss",
        data_by_log,
        font,
        args.theme,
        args.show_annotations,
    )
    fig_train_rel_sol = make_fig(
        "train_rel_sol",
        "Training relative error of represented solution",
        data_by_log,
        font,
        args.theme,
        args.show_annotations,
    )
    fig_val_rel_sol = make_fig(
        "val_rel_sol",
        "Validation relative error of represented solution",
        data_by_log,
        font,
        args.theme,
        args.show_annotations,
    )
    fig_rel_green = make_fig(
        "rel_green",
        "Relative error of Green's function",
        data_by_log,
        font,
        args.theme,
        args.show_annotations,
    )

    save_fig(fig_loss, args.outdir / "loss")
    save_fig(fig_train_rel_sol, args.outdir / "train_rel_sol")
    save_fig(fig_val_rel_sol, args.outdir / "val_rel_sol")
    save_fig(fig_rel_green, args.outdir / "rel_green")


if __name__ == "__main__":
    main()
