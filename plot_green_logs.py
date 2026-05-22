from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import plotly.graph_objs as go


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOKEN_PATTERN = re.compile(
    rf"\|\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>{NUMBER_PATTERN})"
)


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


def make_fig(
    metric_key: str,
    label: str,
    data_by_log: Dict[str, Dict[str, List[float]]],
    font: Dict,
    theme: str,
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
    fig.update_layout(
        title=label,
        xaxis_title="Epoch",
        yaxis_title=label,
        yaxis_type="log",
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
    try:
        fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        if not _warned_static:
            print("Static export skipped (requires kaleido + Chrome); HTML saved instead.")
            _warned_static = True
    fig.write_html(str(base_path.with_suffix(".html")))


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

    fig_loss = make_fig("loss", "Training Loss", data_by_log, font, args.theme)
    fig_train_rel_sol = make_fig(
        "train_rel_sol",
        "Training relative error of represented solution",
        data_by_log,
        font,
        args.theme,
    )
    fig_val_rel_sol = make_fig(
        "val_rel_sol",
        "Validation relative error of represented solution",
        data_by_log,
        font,
        args.theme,
    )
    fig_rel_green = make_fig(
        "rel_green",
        "Relative error of Green's function",
        data_by_log,
        font,
        args.theme,
    )

    save_fig(fig_loss, args.outdir / "loss")
    save_fig(fig_train_rel_sol, args.outdir / "train_rel_sol")
    save_fig(fig_val_rel_sol, args.outdir / "val_rel_sol")
    save_fig(fig_rel_green, args.outdir / "rel_green")


if __name__ == "__main__":
    main()
