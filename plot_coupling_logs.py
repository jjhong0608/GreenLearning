from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import plotly.graph_objs as go

from greenonet.plotly_io import save_plotly_figure


VALUE_RE = r"(?:nan|inf|-inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
LOG_Y_FLOOR = 1e-16


def _parse_float(value: str | None, default: float = float("nan")) -> float:
    if value is None:
        return default
    return float(value)


def _parse_entries(lines: Iterable[str]) -> List[Dict[str, float]]:
    pattern_current = re.compile(
        rf"epoch\s+(?P<epoch>\d+).*?\|\s*train\s+loss=(?P<loss_tr>{VALUE_RE})"
        rf"\s+l2_cons=(?P<l2_cons_tr>{VALUE_RE})"
        rf"\s+energy_cons=(?P<energy_cons_tr>{VALUE_RE})"
        rf"\s+cross_cons=(?P<cross_cons_tr>{VALUE_RE})"
        rf"(?:\s+balance_loss=(?P<balance_loss_tr>{VALUE_RE}))?"
        rf"(?:\s+symmetric_boundary_loss=(?P<symmetric_boundary_loss_tr>{VALUE_RE}))?"
        rf"\s+rel_flux=(?P<rflux_tr>{VALUE_RE})"
        rf"\s+rel_sol=(?P<rsol_tr>{VALUE_RE})"
        rf"(?:.*?\|\s*val\s+loss=(?P<loss_val>{VALUE_RE})"
        rf"\s+l2_cons=(?P<l2_cons_val>{VALUE_RE})"
        rf"\s+energy_cons=(?P<energy_cons_val>{VALUE_RE})"
        rf"\s+cross_cons=(?P<cross_cons_val>{VALUE_RE})"
        rf"(?:\s+balance_loss=(?P<balance_loss_val>{VALUE_RE}))?"
        rf"(?:\s+symmetric_boundary_loss=(?P<symmetric_boundary_loss_val>{VALUE_RE}))?"
        rf"\s+rel_flux=(?P<rflux_val>{VALUE_RE})"
        rf"\s+rel_sol=(?P<rsol_val>{VALUE_RE}))?",
        re.IGNORECASE,
    )
    pattern_epoch = re.compile(
        r"Epoch\s+(?P<epoch>\d+).*?"
        rf"train[^|]*?loss=(?P<loss_tr>{VALUE_RE})"
        rf"[^|]*?cons=(?P<cons_tr>{VALUE_RE})"
        rf"[^|]*?rel_flux=(?P<rflux_tr>{VALUE_RE})"
        rf"[^|]*?rel_sol=(?P<rsol_tr>{VALUE_RE})"
        rf"(?:.*?\|\s*val(?:[^|]*?loss=(?P<loss_val>{VALUE_RE}))?"
        rf"[^|]*?cons=(?P<cons_val>{VALUE_RE})"
        rf"[^|]*?rel_flux=(?P<rflux_val>{VALUE_RE})"
        rf"[^|]*?rel_sol=(?P<rsol_val>{VALUE_RE}))?",
        re.IGNORECASE,
    )
    pattern_lbfgs = re.compile(
        r"(?:Coupling\s+)?LBFGS\s+epoch\s+(?P<epoch>\d+).*?"
        rf"last\s+loss:\s*(?P<loss_tr>{VALUE_RE})"
        rf".*?cons=(?P<cons_tr>{VALUE_RE})"
        rf".*?rel_flux=(?P<rflux_tr>{VALUE_RE})"
        rf".*?rel_sol=(?P<rsol_tr>{VALUE_RE})"
        rf"(?:.*?\|\s*val(?:[^|]*?loss=(?P<loss_val>{VALUE_RE}))?"
        rf"[^|]*?cons=(?P<cons_val>{VALUE_RE})"
        rf"[^|]*?rel_flux=(?P<rflux_val>{VALUE_RE})"
        rf"[^|]*?rel_sol=(?P<rsol_val>{VALUE_RE}))?",
        re.IGNORECASE,
    )
    entries: List[Dict[str, float]] = []

    for line in lines:
        match = pattern_current.search(line)
        if match:
            entries.append(
                {
                    "raw_epoch": _parse_float(match.group("epoch")),
                    "loss_train": _parse_float(match.group("loss_tr")),
                    "loss_val": _parse_float(match.group("loss_val")),
                    "l2_cons_train": _parse_float(match.group("l2_cons_tr")),
                    "energy_cons_train": _parse_float(match.group("energy_cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "l2_cons_val": _parse_float(match.group("l2_cons_val")),
                    "energy_cons_val": _parse_float(match.group("energy_cons_val")),
                    "rel_flux_val": _parse_float(match.group("rflux_val")),
                    "rel_sol_val": _parse_float(match.group("rsol_val")),
                }
            )
            continue
        match = pattern_epoch.search(line)
        if match:
            entries.append(
                {
                    "raw_epoch": _parse_float(match.group("epoch")),
                    "loss_train": _parse_float(match.group("loss_tr")),
                    "loss_val": _parse_float(match.group("loss_val")),
                    "l2_cons_train": float("nan"),
                    "energy_cons_train": _parse_float(match.group("cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "l2_cons_val": float("nan"),
                    "energy_cons_val": _parse_float(match.group("cons_val")),
                    "rel_flux_val": _parse_float(match.group("rflux_val")),
                    "rel_sol_val": _parse_float(match.group("rsol_val")),
                }
            )
            continue
        match = pattern_lbfgs.search(line)
        if match:
            entries.append(
                {
                    "raw_epoch": _parse_float(match.group("epoch")),
                    "loss_train": _parse_float(match.group("loss_tr")),
                    "loss_val": _parse_float(match.group("loss_val")),
                    "l2_cons_train": float("nan"),
                    "energy_cons_train": _parse_float(match.group("cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "l2_cons_val": float("nan"),
                    "energy_cons_val": _parse_float(match.group("cons_val")),
                    "rel_flux_val": _parse_float(match.group("rflux_val")),
                    "rel_sol_val": _parse_float(match.group("rsol_val")),
                }
            )
    return entries


def parse_log(path: Path) -> Dict[str, List[float]]:
    """Parse CouplingNet training.log and return epoch-aligned metrics."""
    entries = _parse_entries(path.read_text().splitlines())
    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "loss_train": [],
        "loss_val": [],
        "l2_cons_train": [],
        "energy_cons_train": [],
        "rel_flux_train": [],
        "rel_sol_train": [],
        "l2_cons_val": [],
        "energy_cons_val": [],
        "rel_flux_val": [],
        "rel_sol_val": [],
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
        metrics["loss_train"].append(entry["loss_train"])
        metrics["loss_val"].append(entry["loss_val"])
        metrics["l2_cons_train"].append(entry["l2_cons_train"])
        metrics["energy_cons_train"].append(entry["energy_cons_train"])
        metrics["rel_flux_train"].append(entry["rel_flux_train"])
        metrics["rel_sol_train"].append(entry["rel_sol_train"])
        metrics["l2_cons_val"].append(entry["l2_cons_val"])
        metrics["energy_cons_val"].append(entry["energy_cons_val"])
        metrics["rel_flux_val"].append(entry["rel_flux_val"])
        metrics["rel_sol_val"].append(entry["rel_sol_val"])
    return metrics


def _log_plot_y(value: float) -> float:
    return max(value, LOG_Y_FLOOR)


def _mask_nan(values: List[float]) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        if v != v:
            out.append(None)
        else:
            out.append(_log_plot_y(v))
    return out


def _has_visible_values(values: List[float | None]) -> bool:
    return any(value is not None for value in values)


def _format_annotation_value(value: float) -> str:
    return f"{value:.3e}"


def _annotation_y(value: float, *, log_scale: bool) -> float:
    return _log_plot_y(value) if log_scale else value


def _annotation_axis_y(value: float, *, log_scale: bool) -> float:
    plot_value = _log_plot_y(value) if log_scale else value
    return math.log10(plot_value) if log_scale else plot_value


def _finite_points(
    epochs: List[float],
    values: List[float],
) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for idx, (epoch, value) in enumerate(zip(epochs, values)):
        if math.isfinite(epoch) and math.isfinite(value):
            points.append((idx, epoch, value))
    return points


def _annotation_offsets(kind: str, split: str) -> tuple[int, int, str]:
    is_val = split == "val"
    if kind == "last":
        return (46, 30 if is_val else -30, "left")
    return (0, -46, "center")


def _xaxis_config(
    series: List[Tuple[str, Dict[str, List[float]]]],
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


def _add_point_annotation(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    kind: str,
    split: str,
) -> None:
    ax, ay, xanchor = _annotation_offsets(kind, split)
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


def _add_last_min_annotations(
    fig: go.Figure,
    *,
    epochs: List[float],
    values: List[float],
    marker_label: str,
    annotation_label: str,
    split: str,
    color: str,
    log_scale: bool,
) -> None:
    points = _finite_points(epochs, values)
    if not points:
        return

    last_idx, last_epoch, last_value = points[-1]
    min_idx, min_epoch, min_value = min(points, key=lambda item: (item[2], -item[0]))
    marker_x = [last_epoch]
    last_y = _annotation_y(last_value, log_scale=log_scale)
    min_y = _annotation_y(min_value, log_scale=log_scale)
    last_annotation_y = _annotation_axis_y(last_value, log_scale=log_scale)
    min_annotation_y = _annotation_axis_y(min_value, log_scale=log_scale)
    marker_y = [last_y]
    marker_text = [
        f"{marker_label}<br>last {_format_annotation_value(last_value)}<br>"
        f"epoch {last_epoch:g}"
    ]
    if min_idx != last_idx:
        marker_x.append(min_epoch)
        marker_y.append(min_y)
        marker_text.append(
            f"{marker_label}<br>min {_format_annotation_value(min_value)}<br>"
            f"epoch {min_epoch:g}"
        )

    fig.add_trace(
        go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2)),
            name=f"{marker_label} markers",
            text=marker_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    if min_idx == last_idx:
        _add_point_annotation(
            fig,
            x=last_epoch,
            y=last_annotation_y,
            text=(
                f"{annotation_label} last/min<br>"
                f"{_format_annotation_value(last_value)}<br>"
                f"ep {last_epoch:g}"
            ),
            color=color,
            kind="last",
            split=split,
        )
        return

    _add_point_annotation(
        fig,
        x=last_epoch,
        y=last_annotation_y,
        text=(
            f"{annotation_label} last<br>"
            f"{_format_annotation_value(last_value)}<br>"
            f"ep {last_epoch:g}"
        ),
        color=color,
        kind="last",
        split=split,
    )
    _add_point_annotation(
        fig,
        x=min_epoch,
        y=min_annotation_y,
        text=(
            f"{annotation_label} min<br>"
            f"{_format_annotation_value(min_value)}<br>"
            f"ep {min_epoch:g}"
        ),
        color=color,
        kind="min",
        split=split,
    )


def _color_cycle() -> List[str]:
    return [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def make_fig_loss(
    series: List[Tuple[str, Dict[str, List[float]]]],
    font: Dict[str, object],
    theme: str,
    show_annotations: bool = False,
) -> go.Figure:
    fig = go.Figure()
    colors = _color_cycle()
    for idx, (label, metrics) in enumerate(series):
        epochs = metrics["epoch"]
        color = colors[idx % len(colors)]
        for split, dash in (("train", "solid"), ("val", "dot")):
            key = f"loss_{split}"
            if key not in metrics:
                continue
            y_vals = _mask_nan(metrics[key])
            if not _has_visible_values(y_vals):
                continue
            trace_name = f"{label} Loss ({split})"
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=y_vals,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
            if show_annotations:
                annotation_label = split if len(series) == 1 else f"{label} {split}"
                _add_last_min_annotations(
                    fig,
                    epochs=epochs,
                    values=metrics[key],
                    marker_label=trace_name,
                    annotation_label=annotation_label,
                    split=split,
                    color=color,
                    log_scale=True,
                )
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis=_xaxis_config(series, show_annotations),
        yaxis_title="Loss",
        yaxis_type="log",
        yaxis=dict(exponentformat="power"),
        template=theme,
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


def make_fig_metric(
    series: List[Tuple[str, Dict[str, List[float]]]],
    metric_key: str,
    title: str,
    yaxis_title: str,
    log_scale: bool,
    font: Dict[str, object],
    theme: str,
    show_annotations: bool = False,
) -> go.Figure:
    fig = go.Figure()
    colors = _color_cycle()
    for idx, (label, metrics) in enumerate(series):
        epochs = metrics["epoch"]
        color = colors[idx % len(colors)]
        for split, dash in (("train", "solid"), ("val", "dot")):
            key = f"{metric_key}_{split}"
            if key not in metrics:
                continue
            y_vals = _mask_nan(metrics[key])
            if not _has_visible_values(y_vals):
                continue
            trace_name = f"{label} ({split})"
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=y_vals,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
            if show_annotations:
                annotation_label = split if len(series) == 1 else f"{label} {split}"
                _add_last_min_annotations(
                    fig,
                    epochs=epochs,
                    values=metrics[key],
                    marker_label=trace_name,
                    annotation_label=annotation_label,
                    split=split,
                    color=color,
                    log_scale=log_scale,
                )
    fig.update_layout(
        title=title,
        xaxis=_xaxis_config(series, show_annotations),
        yaxis_title=yaxis_title,
        yaxis_type="log" if log_scale else "linear",
        yaxis=dict(exponentformat="power"),
        template=theme,
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


def save_fig(fig: go.Figure, base_path: Path) -> None:
    save_plotly_figure(fig, base_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CouplingNet training logs with separate error figures."
    )
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to training.log files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels for each log (must match number of logs).",
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
    parser.add_argument(
        "--theme",
        type=str,
        default="plotly_white",
        help="Plotly template name for the plots.",
    )
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        help="Annotate each trace with its last value and minimum value.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        raise ValueError("Number of labels must match number of logs.")

    series: List[Tuple[str, Dict[str, List[float]]]] = []
    for idx, log_path in enumerate(args.logs):
        metrics = parse_log(log_path)
        if not metrics.get("epoch"):
            print(f"Warning: no metrics parsed from {log_path}")
            continue
        label = args.labels[idx] if args.labels else log_path.stem.replace("_", " ")
        series.append((label, metrics))

    if not series:
        raise RuntimeError("No valid log data parsed.")

    font = {"family": args.font_family, "size": 14}

    fig_loss = make_fig_loss(series, font, args.theme, args.show_annotations)
    fig_l2 = make_fig_metric(
        series,
        metric_key="l2_cons",
        title="L2 Consistency",
        yaxis_title="L2 Consistency",
        log_scale=True,
        font=font,
        theme=args.theme,
        show_annotations=args.show_annotations,
    )
    fig_energy = make_fig_metric(
        series,
        metric_key="energy_cons",
        title="Energy Consistency",
        yaxis_title="Energy Consistency",
        log_scale=True,
        font=font,
        theme=args.theme,
        show_annotations=args.show_annotations,
    )
    fig_flux = make_fig_metric(
        series,
        metric_key="rel_flux",
        title="Flux-Divergence Relative Error",
        yaxis_title="Relative Error",
        log_scale=True,
        font=font,
        theme=args.theme,
        show_annotations=args.show_annotations,
    )
    fig_sol = make_fig_metric(
        series,
        metric_key="rel_sol",
        title="Solution Relative Error",
        yaxis_title="Relative Error",
        log_scale=True,
        font=font,
        theme=args.theme,
        show_annotations=args.show_annotations,
    )

    save_fig(fig_loss, args.outdir / "loss")
    save_fig(fig_l2, args.outdir / "l2_consistency")
    save_fig(fig_energy, args.outdir / "energy_consistency")
    save_fig(fig_flux, args.outdir / "rel_flux")
    save_fig(fig_sol, args.outdir / "rel_sol")


if __name__ == "__main__":
    main()
