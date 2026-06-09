from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

import greenonet.plotly_io as plotly_io
from greenonet.plotly_io import save_plotly_figure


def test_save_plotly_figure_writes_all_formats_when_static_export_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    fig = go.Figure(data=go.Scatter(x=[0, 1], y=[1, 0]))
    base_path = tmp_path / "figure"

    save_plotly_figure(fig, base_path)

    assert base_path.with_suffix(".html").exists()
    assert base_path.with_suffix(".json").exists()
    assert base_path.with_suffix(".png").exists()
    assert base_path.with_suffix(".pdf").exists()


def test_save_plotly_figure_keeps_html_json_when_static_export_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self, path
        raise RuntimeError("kaleido unavailable")

    plotly_io._WARNED_STATIC_EXPORT = False
    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    fig = go.Figure(data=go.Scatter(x=[0, 1], y=[1, 0]))
    base_path = tmp_path / "figure"

    save_plotly_figure(fig, base_path)

    assert base_path.with_suffix(".html").exists()
    assert base_path.with_suffix(".json").exists()
    assert not base_path.with_suffix(".png").exists()
    assert not base_path.with_suffix(".pdf").exists()
