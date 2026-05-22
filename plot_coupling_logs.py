from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import plotly.graph_objs as go


VALUE_RE = r"(?:nan|inf|-inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


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
                    "cons_train": _parse_float(match.group("energy_cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "cons_val": _parse_float(match.group("energy_cons_val")),
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
                    "cons_train": _parse_float(match.group("cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "cons_val": _parse_float(match.group("cons_val")),
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
                    "cons_train": _parse_float(match.group("cons_tr")),
                    "rel_flux_train": _parse_float(match.group("rflux_tr")),
                    "rel_sol_train": _parse_float(match.group("rsol_tr")),
                    "cons_val": _parse_float(match.group("cons_val")),
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
        "cons_train": [],
        "rel_flux_train": [],
        "rel_sol_train": [],
        "cons_val": [],
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
        metrics["cons_train"].append(entry["cons_train"])
        metrics["rel_flux_train"].append(entry["rel_flux_train"])
        metrics["rel_sol_train"].append(entry["rel_sol_train"])
        metrics["cons_val"].append(entry["cons_val"])
        metrics["rel_flux_val"].append(entry["rel_flux_val"])
        metrics["rel_sol_val"].append(entry["rel_sol_val"])
    return metrics


def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        if v != v:
            out.append(None)
        else:
            out.append(max(v, floor))
    return out


def _has_visible_values(values: List[float | None]) -> bool:
    return any(value is not None for value in values)


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
    series: List[Tuple[str, Dict[str, List[float]]]], font: Dict, theme: str
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
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=y_vals,
                    mode="lines",
                    name=f"{label} Loss ({split})",
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
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


def make_fig_error(
    series: List[Tuple[str, Dict[str, List[float]]]],
    metric_key: str,
    title: str,
    font: Dict,
    theme: str,
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
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=_mask_nan(metrics[key]),
                    mode="lines",
                    name=f"{label} ({split})",
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Error",
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


_warned_static = False


def save_fig(fig: go.Figure, base_path: Path) -> None:
    global _warned_static
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        if not _warned_static:
            print(
                "Static export skipped (requires kaleido + Chrome); HTML saved instead."
            )
            _warned_static = True
    fig.write_html(str(base_path.with_suffix(".html")))


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

    fig_loss = make_fig_loss(series, font, args.theme)
    fig_sol = make_fig_error(
        series,
        metric_key="rel_sol",
        title="Solution Error",
        font=font,
        theme=args.theme,
    )
    fig_flux = make_fig_error(
        series,
        metric_key="rel_flux",
        title="Flux-Divergence Error",
        font=font,
        theme=args.theme,
    )

    save_fig(fig_loss, args.outdir / "losses")
    save_fig(fig_sol, args.outdir / "errors_solution")
    save_fig(fig_flux, args.outdir / "errors_flux")


if __name__ == "__main__":
    main()
