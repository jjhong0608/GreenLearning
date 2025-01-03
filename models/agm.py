import numpy as np
import torch
from plotly import graph_objects as go
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from torch import nn

from test_pinn import exact_solution


class AGM:
    def __init__(self, model: nn.Module):
        self.model = model.cpu().double()
        self.model = self.model.eval()
        x = np.linspace(0, 1, 101)[1:-1]
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        self.points = np.stack((X, Y), axis=1)

        self.data = list()
        self.row = list()
        self.col = list()

        basis = np.zeros(shape=(99, 201))
        for i in range(basis.shape[0]):
            basis[i, 2 * (i + 1) - 1] = 0.5
            basis[i, 2 * (i + 1)] = 1
            basis[i, 2 * (i + 1) + 1] = 0.5
        self.basis = basis

        self.axial_line = np.linspace(0, 1, 101)[1:-1].reshape(-1, 1)
        self.x_index = np.arange(0, x.shape[0] * y.shape[0])
        self.y_index = self.x_index.copy().reshape(x.shape[0], y.shape[0]).T.flatten()

        k = 15
        self.f = np.array(
            list(
                map(
                    lambda x: 2
                    * (k**2 - np.pi**2)
                    * np.sin(np.pi * x[0])
                    * np.sin(np.pi * x[1]),
                    self.points,
                )
            )
        )
        print(f"points: {self.points.shape}\n{self.points}")
        print(f"f: {self.f.shape}\n{self.f}")

    def load_data(self) -> "AGM":
        data = np.load("Examples/helmholtz_basis.npz")

        self.u = data["U"][1:-1, ::2]
        print(f"u: {self.u.shape}\n{self.u}")

        return self

    def construct_matrix(self) -> "AGM":
        output = self.model(
            torch.tensor(self.axial_line).double(), torch.tensor(self.basis).double()
        )  # (99, 99)
        # output = torch.tensor(self.u)
        n = self.axial_line.shape[0]
        data_along_x = output.repeat(n, 1).detach().numpy().flatten()
        row_along_x = (
            torch.arange(n**2).reshape(-1, 1).repeat(1, n).detach().numpy().flatten()
        )
        col_along_x = (
            torch.arange(n**2).reshape(n, n).repeat(1, n).detach().numpy().flatten()
        )

        data_along_y = output.repeat(1, n).detach().numpy().flatten()
        row_along_y = row_along_x.copy()
        col_along_y = (
            torch.arange(n**2).reshape(n, n).T.repeat(n, 1).detach().numpy().flatten()
        )

        self.data = np.concatenate((data_along_x, data_along_y))
        self.row = np.concatenate((row_along_x, row_along_y))
        self.col = np.concatenate((col_along_x, col_along_y))

        rhs = np.zeros(shape=(n**2,))
        for data, row, col in zip(data_along_y, row_along_y, col_along_y):
            rhs[row] += data * self.f[col]
        self.rhs = rhs
        print(f"rhs: {self.rhs.shape}\n{self.rhs}")
        # self.rhs = np.sum(
        #     np.array(np.split(data_along_y * self.f[col_along_y], n**2)), axis=1
        # )
        # print(f"rhs: {self.rhs.shape}\n{self.rhs}")

        self.data_along_x = data_along_x
        self.row_along_x = row_along_x
        self.col_along_x = col_along_x
        self.data_along_y = data_along_y
        self.row_along_y = row_along_y
        self.col_along_y = col_along_y

        return self

    def make_matrix(self) -> "AGM":
        coo = coo_matrix((self.data, (self.row, self.col)))
        self.matrix = coo.tocsr()
        print(f"matrix: {self.matrix.shape}\n{self.matrix}")

        return self

    def calculate_matrix(self) -> "AGM":
        x = spsolve(self.matrix, self.rhs)
        self.sol = x
        print(f"sol: {self.sol.shape}\n{self.sol}")
        return self

    def calculate_solution_along_x(self) -> "AGM":
        n = self.axial_line.shape[0]
        sol_along_x = np.zeros(shape=(n**2,))
        for data, row, col in zip(
            self.data_along_x, self.row_along_x, self.col_along_x
        ):
            sol_along_x[row] += data * self.sol[col]
        self.sol_along_x = sol_along_x

        return self

    def calculate_solution_along_y(self) -> "AGM":
        n = self.axial_line.shape[0]
        sol_along_y = np.zeros(shape=(n**2,))
        for data, row, col in zip(
            self.data_along_y, self.row_along_y, self.col_along_y
        ):

            sol_along_y[row] -= data * (self.sol[col] - self.f[col])
        self.sol_along_y = sol_along_y
        return self

    def calculate_solution(self) -> "AGM":
        self.calculate_solution_along_x()
        self.calculate_solution_along_y()
        self.solution = (self.sol_along_x + self.sol_along_y) * 0.5
        print(f"solution: {self.solution.shape}\n{self.solution}")
        print(f"solution_along_x: {self.sol_along_x.shape}\n{self.sol_along_x}")
        print(f"solution_along_y: {self.sol_along_y.shape}\n{self.sol_along_y}")

        return self

    def plot(self) -> "AGM":
        # output = self.model(
        #     torch.tensor(self.axial_line).double(), torch.tensor(self.basis).double()
        # )
        output = torch.tensor(self.u)

        data = list()
        for i in range(output.shape[1]):
            data.append(
                go.Scatter(
                    x=np.linspace(0, 1, 101),
                    # x=self.axial_line.flatten(),
                    y=output[:, i].detach().numpy().flatten(),
                    mode="lines",
                    name=f"u{i}",
                    showlegend=False,
                )
            )

        layout = go.Layout(
            template="plotly_white",
            width=1500,
            height=1500,
            font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
            showlegend=False,
        )

        figure = go.Figure(data=[data[21]], layout=layout)
        figure.show()

        return self

    def plot_solution(self) -> "AGM":
        solution = self.solution.reshape(
            self.axial_line.shape[0], self.axial_line.shape[0]
        )
        solution = np.pad(solution, ((1, 1), (1, 1)), "constant")
        axial_line = np.linspace(0, 1, 101)

        exact_solution = np.array(
            list(
                map(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]), self.points)
            )
        ).reshape(99, 99)
        exact_solution = np.pad(exact_solution, ((1, 1), (1, 1)), "constant")

        error = np.abs(solution - exact_solution)

        print(f"axial_line: {axial_line.shape}\n{axial_line}")
        print(f"solution: {solution.shape}\n{solution}")
        data = [
            go.Heatmap(
                x=axial_line,
                y=axial_line,
                # z=solution,
                # z=error,
                z=exact_solution,
                # showscale=False,
            )
        ]

        layout = go.Layout(
            template="plotly_white",
            width=1700,
            height=1500,
            font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
            showlegend=False,
            # title="Predicted Solution",
            title="Exact Solution",
        )

        figure = go.Figure(data=data, layout=layout)
        figure.show()
        figure.write_image("Exact_solution.png")

        return self
