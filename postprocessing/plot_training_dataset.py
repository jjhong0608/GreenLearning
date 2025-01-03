from pathlib import Path
from typing import Union

from plotly import graph_objects as go
import numpy as np


def plot_training_dataset(path: Union[str, Path]) -> None:
    path = Path(path) if not isinstance(path, Path) else path
    data = np.load(path)
    x = data["X"]
    y = data["Y"]
    u = data["U"]
    f = data["F"]
    data = list()
    for i in range(u.shape[1]):
        data.append(
            go.Scatter(
                x=x,
                y=u[:, i],
                mode="lines",
                name=f"u{i}",
            )
        )

    layout = go.Layout(
        template="plotly_white",
        width=1500,
        height=1500,
        font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
        showlegend=False,
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_image(path.with_suffix(".png"))


if __name__ == "__main__":
    plot_training_dataset(
        "/Users/jjhong0608/Documents/AGM/GreenLearning/Examples/burgers.npz"
    )
