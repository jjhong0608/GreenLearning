from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import plotly.graph_objs as go


METRIC_MAP = {
    "eq": "Cross-Integral",
    "fdb": "Flux-Divergence Balance",
    "cons": "Consistency",
    "rel_flux": "Flux-Divergence Error",
    "rel_sol": "Solution Error",
}


def parse_log(path: Path) -> Dict[str, List[float]]:
    """Parse training.log extracting train/val metrics per epoch."""
    pattern_full = re.compile(
        r"Epoch\s+(?P<epoch>\d+).*?"
        r"train[^|]*?eq=(?P<eq_tr>[-+eE0-9\.]+)"
        r"[^|]*?fdb=(?P<fdb_tr>[-+eE0-9\.]+)"
        r"[^|]*?cons=(?P<cons_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_tr>[-+eE0-9\.]+)"
        r".*?val[^|]*?eq=(?P<eq_val>[-+eE0-9\.]+)"
        r"[^|]*?fdb=(?P<fdb_val>[-+eE0-9\.]+)"
        r"[^|]*?cons=(?P<cons_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_val>[-+eE0-9\.]+)",
        re.IGNORECASE,
    )
    pattern_train_only = re.compile(
        r"Epoch\s+(?P<epoch>\d+).*?"
        r"loss_eq=(?P<eq_tr>[-+eE0-9\.]+)"
        r".*?rel_flux=(?P<rflux_tr>[-+eE0-9\.]+)"
        r".*?rel_sol=(?P<rsol_tr>[-+eE0-9\.]+)",
        re.IGNORECASE,
    )
    pattern_lbfgs = re.compile(
        r"LBFGS\s+epoch\s+(?P<epoch>\d+)[^|]*?\|\s*train[^|]*?eq=(?P<eq_tr>[-+eE0-9\.]+)"
        r"[^|]*?fdb=(?P<fdb_tr>[-+eE0-9\.]+)"
        r"[^|]*?cons=(?P<cons_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_tr>[-+eE0-9\.]+)"
        r"(?:[^|]*?\|\s*val[^|]*?eq=(?P<eq_val>[-+eE0-9\.]+)"
        r"[^|]*?fdb=(?P<fdb_val>[-+eE0-9\.]+)"
        r"[^|]*?cons=(?P<cons_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_val>[-+eE0-9\.]+))?",
        re.IGNORECASE,
    )
    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "eq_train": [],
        "fdb_train": [],
        "cons_train": [],
        "rel_flux_train": [],
        "rel_sol_train": [],
        "eq_val": [],
        "fdb_val": [],
        "cons_val": [],
        "rel_flux_val": [],
        "rel_sol_val": [],
    }
    entries: List[Dict[str, float]] = []

    for line in path.read_text().splitlines():
        m = pattern_full.search(line)
        if m:
            epoch = int(m.group("epoch"))
            entries.append(
                {
                    "raw_epoch": epoch,
                    "stage": "adam",
                    "eq_train": float(m.group("eq_tr")),
                    "fdb_train": float(m.group("fdb_tr")),
                    "cons_train": float(m.group("cons_tr")),
                    "rel_flux_train": float(m.group("rflux_tr")),
                    "rel_sol_train": float(m.group("rsol_tr")),
                    "eq_val": float(m.group("eq_val")),
                    "fdb_val": float(m.group("fdb_val")),
                    "cons_val": float(m.group("cons_val")),
                    "rel_flux_val": float(m.group("rflux_val")),
                    "rel_sol_val": float(m.group("rsol_val")),
                }
            )
            continue
        m2 = pattern_train_only.search(line)
        if m2:
            epoch = int(m2.group("epoch"))
            entries.append(
                {
                    "raw_epoch": epoch,
                    "stage": "adam",
                    "eq_train": float(m2.group("eq_tr")),
                    "fdb_train": float("nan"),
                    "cons_train": float("nan"),
                    "rel_flux_train": float(m2.group("rflux_tr")),
                    "rel_sol_train": float(m2.group("rsol_tr")),
                    "eq_val": float("nan"),
                    "fdb_val": float("nan"),
                    "cons_val": float("nan"),
                    "rel_flux_val": float("nan"),
                    "rel_sol_val": float("nan"),
                }
            )
            continue
        m3 = pattern_lbfgs.search(line)
        if m3:
            lb_epoch = int(m3.group("epoch"))
            entries.append(
                {
                    "raw_epoch": lb_epoch,
                    "stage": "lbfgs",
                    "eq_train": float(m3.group("eq_tr")),
                    "fdb_train": float(m3.group("fdb_tr")),
                    "cons_train": float(m3.group("cons_tr")),
                    "rel_flux_train": float(m3.group("rflux_tr")),
                    "rel_sol_train": float(m3.group("rsol_tr")),
                    "eq_val": float(m3.group("eq_val")) if m3.group("eq_val") else float("nan"),
                    "fdb_val": float(m3.group("fdb_val")) if m3.group("fdb_val") else float("nan"),
                    "cons_val": float(m3.group("cons_val")) if m3.group("cons_val") else float("nan"),
                    "rel_flux_val": float(m3.group("rflux_val")) if m3.group("rflux_val") else float("nan"),
                    "rel_sol_val": float(m3.group("rsol_val")) if m3.group("rsol_val") else float("nan"),
                }
            )
            continue

    # Compute cumulative epochs in file order to avoid overlapping epochs (multiple runs/stages)
    cumulative_entries: List[Dict[str, float]] = []
    offset = 0
    last_raw = None
    last_effective = 0
    for e in entries:
        raw_epoch = e["raw_epoch"]
        if last_raw is not None and raw_epoch <= last_raw:
            offset += last_effective
        effective_epoch = raw_epoch + offset
        last_raw = raw_epoch
        last_effective = effective_epoch
        e["epoch"] = effective_epoch
        cumulative_entries.append(e)

    for e in cumulative_entries:
        metrics["epoch"].append(e["epoch"])
        metrics["eq_train"].append(e["eq_train"])
        metrics["fdb_train"].append(e["fdb_train"])
        metrics["cons_train"].append(e["cons_train"])
        metrics["rel_flux_train"].append(e["rel_flux_train"])
        metrics["rel_sol_train"].append(e["rel_sol_train"])
        metrics["eq_val"].append(e["eq_val"])
        metrics["fdb_val"].append(e["fdb_val"])
        metrics["cons_val"].append(e["cons_val"])
        metrics["rel_flux_val"].append(e["rel_flux_val"])
        metrics["rel_sol_val"].append(e["rel_sol_val"])
    return metrics


def make_fig_losses(data_by_log: Dict[str, Dict[str, List[float]]], font: Dict) -> go.Figure:
    fig = go.Figure()
    colors = {
        "eq": "#1f77b4",
        "fdb": "#2ca02c",
        "cons": "#d62728",
    }
    def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
        out: List[float | None] = []
        for v in values:
            if v != v:
                out.append(None)
            else:
                out.append(max(v, floor))
        return out  # NaN check via v==v

    for log_name, metrics in data_by_log.items():
        epochs = metrics["epoch"]
        for key, label in (("eq", "Cross-Integral"), ("cons", "Consistency")):
        # for key, label in (("eq", "Cross-Integral"), ("fdb", "Flux-Divergence Balance"), ("cons", "Consistency")):
            for split, dash in (("train", "solid"), ("val", "dash")):
                metric_key = f"{key}_{split}"
                if metric_key not in metrics:
                    continue
                y_vals = _mask_nan(metrics[metric_key])
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=y_vals,
                        mode="lines",
                        name=f"{label} ({split}) [{log_name}]",
                        line=dict(color=colors[key], dash=dash),
                        connectgaps=True,
                    )
                )
    fig.update_layout(
        title="Training vs Validation Loss Components",
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
    def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
        out: List[float | None] = []
        for v in values:
            if v != v:
                out.append(None)
            else:
                out.append(max(v, floor))
        return out

    for log_name, metrics in data_by_log.items():
        epochs = metrics["epoch"]
        for key, label in (("rel_flux", "Flux-Divergence Error"), ("rel_sol", "Solution Error")):
            for split, dash in (("train", "solid"), ("val", "dash")):
                metric_key = f"{key}_{split}"
                if metric_key not in metrics:
                    continue
                y_vals = _mask_nan(metrics[metric_key])
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=y_vals,
                        mode="lines",
                        name=f"{label} ({split}) [{log_name}]",
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
