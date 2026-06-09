from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go


_WARNED_STATIC_EXPORT = False


def save_plotly_figure(
    fig: go.Figure,
    base_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Save a Plotly figure in editable and publication-oriented formats."""
    global _WARNED_STATIC_EXPORT

    base_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = base_path.with_suffix(".html")
    json_path = base_path.with_suffix(".json")
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_json(str(json_path), pretty=True)

    try:
        fig.write_image(str(png_path))
        fig.write_image(str(pdf_path))
    except Exception as exc:  # pragma: no cover - depends on local kaleido/Chrome
        if not _WARNED_STATIC_EXPORT:
            message = (
                "Static Plotly export skipped; HTML and JSON were saved. "
                f"Reason: {exc}"
            )
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
            _WARNED_STATIC_EXPORT = True
