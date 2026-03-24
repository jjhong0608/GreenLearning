from plotly import graph_objects as go
import numpy as np

MIN_PT = 0.0
MAX_PT = 1.0
NUM_PT = 101
NUM_LN = 15

if __name__ == "__main__":
    line = np.linspace(MIN_PT, MAX_PT, NUM_PT)
    line_position = np.linspace(MIN_PT, MAX_PT, NUM_LN + 2)[1:-1]

    layout = go.Layout(
        template="simple_white",
        # template="plotly_white",
        width=900,
        height=900,
        font=go.layout.Font(family="Times New Roman", weight="bold", size=24),
        xaxis=go.layout.XAxis(
            visible=False,
        ),
        yaxis=go.layout.YAxis(
            visible=False,
        ),
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0,
        ),
    )
    fig = go.Figure(layout=layout)
    for y in line_position:
        fig.add_trace(
            go.Scatter(
                x=line,
                y=np.full_like(line, y),
                mode="lines",
                line=go.scatter.Line(color="#17becf"),
                showlegend=False,
            )
        )
    for x in line_position:
        fig.add_trace(
            go.Scatter(
                y=line,
                x=np.full_like(line, x),
                mode="lines",
                line=go.scatter.Line(color="#ff9896"),
                showlegend=False,
            )
        )

    from pathlib import Path

    target = Path.cwd() / "axial_line_plot" / "axial_lines.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"{target} is saved")
    fig.write_image(target.with_suffix(".png"))
    fig.write_image(target.with_suffix(".pdf"))
