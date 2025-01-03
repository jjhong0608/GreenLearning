from pathlib import Path
from typing import Union
from plotly import graph_objects as go
import numpy as np


def plot_results(path: Union[str, Path]) -> None:
    path = Path(path) if not isinstance(path, Path) else path
    results = np.load(path)
    x = results["x"]
    y = results["y"]
    z = results["green_output"].reshape(x.shape[0], y.shape[0])
    data = [
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            showscale=False,
        )
    ]
    layout = go.Layout(
        template="plotly_white",
        width=1500,
        height=1500,
        font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    fig.write_image(path.with_suffix(".png"))


if __name__ == "__main__":
    plot_results(
        "/Users/jjhong0608/Documents/AGM/GreenLearning/work_dir/attmp1/green_function_output.npz"
    )
