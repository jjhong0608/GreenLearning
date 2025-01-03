import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn
from safetensors.torch import save_model, load_model
from torch import optim

from .MLP import MLP
from .modified_loss import Loss
from .oper_learn_arch import DeepOperatorLearningArchitecture
from .lars import LARS
from .scheduler import WarmupCosineAnnealingScheduler


class GreenLearning:
    def __init__(
        self,
        green_layers: List[int],
        activation: str = "rational",
        adam_epochs: int = 1000,
        lbfgs_epochs: int = 5000,
        work_dir: Union[str, Path] = "work_dir",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        # Training parameters
        self.adam_epochs = adam_epochs
        self.lbfgs_epochs = lbfgs_epochs
        self.device = device

        # Create working directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = Path(work_dir) / timestamp
        self.work_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = log_dir / f"{timestamp}.txt"
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
            datefmt="%Y/%m/%d_%H:%M:%S",
        )
        logging.info(
            f"GreenLearning initialized at {datetime.now()} - Log file: {self.log_file}"
        )

    def load_data(self, path: Union[str, Path]) -> "GreenLearning":
        # Ensure `path` is a Path object
        if not isinstance(path, Path):
            path = Path(path)

        # Validate file format
        if path.suffix != ".npz":
            raise ValueError(f"Data file {path} must be a .npz file.")

        # Log data loading
        logging.info(f"Loading data from {path}")

        # Load data from the .npz file
        data = np.load(path)

        # Convert loaded data into PyTorch tensors and move to the specified device
        self.x = torch.tensor(data["X"], device=self.device)  # Spatial domain points
        self.y = torch.tensor(data["Y"], device=self.device)  # Singular points
        self.u = torch.tensor(data["U"], device=self.device)  # Ground truth solution
        self.f = torch.tensor(data["F"], device=self.device)  # Forcing terms (RHS)
        self.hom_u = torch.tensor(
            data["HOM_U"], device=self.device
        )  # Homogeneous solution
        self.green_network = DeepOperatorLearningArchitecture(
            rhs_size=self.f.shape[0], num_layers=4, hidden_size=50
        ).to(self.device)

        self.training_input = self.x.reshape(-1, 1)

        self._initialize_optimizers()

        return self

    def load_valid_data(self, path: Union[str, Path]) -> "GreenLearning":
        # Ensure `path` is a Path object
        if not isinstance(path, Path):
            path = Path(path)

        # Validate file format
        if path.suffix != ".npz":
            raise ValueError(f"Data file {path} must be a .npz file.")

        # Log data loading
        logging.info(f"Loading data from {path}")

        # Load data from the .npz file
        data = np.load(path)

        # Convert loaded data into PyTorch tensors and move to the specified device
        self.valid_x = torch.tensor(
            data["X"], device=self.device
        )  # Spatial domain points
        self.valid_y = torch.tensor(data["Y"], device=self.device)  # Singular points
        self.valid_u = torch.tensor(
            data["U"], device=self.device
        )  # Ground truth solution
        self.valid_f = torch.tensor(
            data["F"], device=self.device
        )  # Forcing terms (RHS)
        self.valid_training_input = self.x.reshape(-1, 1)

        return self

    def _initialize_optimizers(self) -> "GreenLearning":
        # Initialize the LARS optimizer
        self.lars = LARS(
            self.green_network.parameters(), lr=4.8, momentum=0.9, weight_decay=0.0001
        )
        self.scheduler = WarmupCosineAnnealingScheduler(
            self.lars, warmup_iters=10, T_max=self.adam_epochs
        )

        # Initialize the L-BFGS optimizer with specified hyperparameters
        self.lbfgs = optim.LBFGS(
            self.green_network.parameters(),
            lr=1.0,
            max_iter=self.lbfgs_epochs,
            max_eval=10**5,
            tolerance_grad=1.0 * np.finfo(float).eps,  # Tolerance for gradient norm
            tolerance_change=1e-20,  # Minimum change in loss for stopping
            history_size=100,  # Number of past updates to store for Hessian approximation
            line_search_fn="strong_wolfe",  # Line search strategy for step size determination
        )

        return self

    def export_green_function(
        self, path: Optional[Union[str, Path]] = None
    ) -> "GreenLearning":
        if path is None:
            path = self.work_dir

        # Ensure `path` is a Path object for consistency.
        if not isinstance(path, Path):
            path = Path(path)

        # Check if the provided path is a directory.
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        # Create the directory if it does not exist.
        path.mkdir(parents=True, exist_ok=True)

        # Generate a finely discretized grid over the spatial domain.
        # x: Fine discretization of the first spatial dimension.
        # y: Fine discretization of the second spatial dimension.
        x = torch.linspace(
            self.x[0], self.x[-1], 1001
        )  # 1001 points for high resolution.
        y = torch.linspace(self.y[0], self.y[-1], 1001)

        # Create a mesh grid of the spatial domain points.
        # The resulting `_input` tensor contains all combinations of the (x, y) points.
        _input = (
            torch.stack(
                torch.meshgrid(x, y, indexing="ij"), dim=2
            )  # Create a 2D grid of (x, y) pairs.
            .reshape(-1, 2)  # Flatten the grid into a list of 2D points.
            .to(
                self.device
            )  # Move the tensor to the specified device (e.g., CPU or GPU).
        )

        # Pass the input points through the Green's function network to get the output.
        green_output = self.green_network(_input)

        # Detach the output tensor from the computation graph and move it to the CPU.
        # Convert the tensor to a NumPy array for saving.
        green_output = green_output.detach().cpu().numpy()

        # Save the Green's function output to a `.npz` file in the specified directory.
        # The file includes the spatial domain points (x, y) and the computed output.
        np.savez(
            path / "green_function_output.npz", x=x, y=y, green_output=green_output
        )

        # Return the instance for method chaining.
        return self

    def dataloader(self) -> torch.utils.data.DataLoader:
        class dataset(torch.utils.data.Dataset):
            def __init__(self, u: torch.Tensor, f: torch.Tensor) -> None:
                self.u = u.T
                self.f = f.T

            def __len__(self) -> int:
                return len(self.f)

            def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.u[index], self.f[index]

        return torch.utils.data.DataLoader(
            dataset(self.u, self.f), batch_size=100, shuffle=True
        )

    def train(self) -> "GreenLearning":
        # Initialize the custom loss function with necessary tensors.
        # The loss function uses domain points, singular points, RHS terms, and ground truth solutions.
        criterion = Loss(
            spatial_dom=self.x,
            singular_points=self.y,
            rhs=self.f,
            sol=self.u,
            hom_sol=self.hom_u,
        )

        dataloader = self.dataloader()
        print(f"len(dataloader) = {len(dataloader)}")
        # Set up a progress bar using the `rich` library.
        # The progress bar provides real-time feedback on the training process.
        with Progress(
            "[progress.description]{task.description}",  # Description of the current task.
            TimeElapsedColumn(),  # Column showing the elapsed time.
            TimeRemainingColumn(),  # Column showing the estimated remaining time.
            "[progress.percentage]{task.percentage:>3.0f}%",  # Column showing percentage completed.
        ) as progress:
            # Add a task to the progress bar, specifying the total number of epochs.
            task = progress.add_task(
                "[green]Training Green function...",
                total=self.lbfgs_epochs * len(dataloader),
            )

            # Phase 1: Training using the Adam optimizer.
            for i in range(1, self.adam_epochs + 1):
                running_loss = 0.0
                for u, f in dataloader:
                    self.lars.zero_grad()  # Reset gradients from the previous step.

                    # Compute outputs from the Green's function and homogeneous solution networks.
                    green_output = self.green_network(self.training_input, f)
                    # green_output = self.green_network(self.training_input, self.f.T)

                    # Calculate the loss between predicted and actual solutions.
                    loss = criterion(green_output, u.T)

                    # Perform backpropagation to compute gradients.
                    loss.backward()
                    running_loss += loss.item() / len(dataloader)

                    # Update the weights of the networks using the Adam optimizer.
                    self.lars.step()
                    self.scheduler.step(i)

                with torch.no_grad():
                    green_output = self.green_network(
                        self.valid_training_input, self.valid_f.T
                    )
                    valid_loss = criterion(green_output, self.valid_u)

                # Log and update the progress bar for the current epoch.
                description = f"Epoch {i:>6}/{self.adam_epochs} - Loss: {running_loss:.4e} - Validation Loss: {valid_loss:.4e}"
                logging.info(description, extra={"file": __file__, "function": "train"})
                progress.update(task, advance=1, description=f"[green]{description}")

            # Set the starting epoch for L-BFGS optimization.
            self.lbfgs_step = self.adam_epochs + 1

            while self.lbfgs_step < self.lbfgs_epochs * len(dataloader):
                for u, f in dataloader:

                    def closure():
                        """
                        Closure function for the L-BFGS optimizer.

                        This function recomputes the forward and backward pass, including
                        the loss calculation and gradient updates.

                        Returns
                        -------
                        torch.Tensor
                            The current loss value.
                        """
                        self.lbfgs.zero_grad()  # Reset gradients for the L-BFGS optimizer.

                        # Compute outputs from the Green's function and homogeneous solution networks.
                        green_output = self.green_network(self.training_input, f)

                        # Calculate the loss between predicted and actual solutions.
                        loss = criterion(green_output, u.T)

                        # Perform backpropagation to compute gradients.
                        loss.backward()
                        with torch.no_grad():
                            green_output = self.green_network(
                                self.valid_training_input, self.valid_f.T
                            )
                            valid_loss = criterion(green_output, self.valid_u)

                        # Update the description and progress bar for the current epoch.
                        description = f"Epoch {self.lbfgs_step:>6}/{self.lbfgs_epochs * len(dataloader)} - Loss: {loss:.4e} - Validation Loss: {valid_loss:.4e}"
                        logging.info(
                            description, extra={"file": __file__, "function": "train"}
                        )
                        self.lbfgs_step += 1
                        progress.update(
                            task, advance=1, description=f"[green]{description}"
                        )

                        return loss

                    self.lbfgs.step(closure)

        # Return the class instance to allow method chaining.
        return self

    def save_model(self, path: Optional[Union[str, Path]] = None):
        if path is None:
            path = self.work_dir

        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        path.mkdir(parents=True, exist_ok=True)

        save_model(
            self.green_network,
            path / "green_network.safetensors",
        )

    def load_model(self, path: Union[str, Path]) -> "GreenLearning":
        if not isinstance(path, Path):
            path = Path(path)
        load_model(
            self.green_network,
            path / "green_network.safetensors",
        )
        print(f"Model loaded from {path}")
        print(f"self.x: {self.x.shape}, [{self.x[0]}, {self.x[-1]}]")
        print(f"self.y: {self.y.shape}, [{self.y[0]}, {self.y[-1]}]")
        return self

    def evaluate(self) -> "GreenLearning":
        idx = 41

        f = torch.zeros_like(self.f.T[0:1, :])
        f[0, 49] = 0.5
        f[0, 50] = 1.0
        f[0, 51] = 0.5

        # f = self.f.T[idx : idx + 1, :]

        k = 15
        k = torch.as_tensor(k)
        exact_green = (k * torch.sin(k)) ** (-1) * torch.sin(k * self.x).unsqueeze(
            1
        ) * torch.sin(k * (self.y - 1).unsqueeze(0)) * (
            self.x.unsqueeze(1) <= self.y.unsqueeze(0)
        ) + (
            k * torch.sin(k)
        ) ** (
            -1
        ) * torch.sin(
            k * self.y
        ).unsqueeze(
            0
        ) * torch.sin(
            k * (self.x - 1)
        ).unsqueeze(
            1
        ) * (
            self.x.unsqueeze(1) > self.y.unsqueeze(0)
        )
        exact_green_output = torch.trapezoid(
            exact_green * f.reshape(1, -1), self.y, dim=1
        )

        # exact_green_output = self.u.T[idx : idx + 1, :].flatten()

        with torch.no_grad():
            green_output = self.green_network(self.training_input, f)
            print(f"f: {f.shape}")
            print(f"training_input: {self.training_input.shape}")
            print(f"Green's function output: {green_output.detach().cpu().numpy()}")
        print(f"exact_green_output: \n{exact_green_output.detach().cpu().numpy()}")
        error = (
            torch.abs(exact_green_output - green_output.flatten())
            .detach()
            .cpu()
            .numpy()
        )
        print(f"error = {error}")

        l2error = (error**2).sum() / (exact_green_output**2).sum()

        print(f"l2error = {l2error}")

        data = [
            go.Scatter(
                x=self.x.detach().cpu().numpy(),
                y=exact_green_output.detach().cpu().numpy(),
                mode="lines",
                line=go.scatter.Line(width=5),
                name="Exact Solution",
            ),
            go.Scatter(
                x=self.x.detach().cpu().numpy(),
                y=green_output.flatten().detach().cpu().numpy(),
                mode="lines",
                line=go.scatter.Line(width=3),
                name="Predicted Solution",
            ),
            go.Scatter(
                x=self.x.detach().cpu().numpy(),
                y=error,
                mode="lines",
                line=go.scatter.Line(width=3),
                name="Error",
            ),
        ]
        layout = go.Layout(
            template="plotly_white",
            width=1500,
            height=1500,
            font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
            # title=f"Linear Basis Function at {self.y[idx].item():.2f}",
        )
        figure = go.Figure(data=data, layout=layout)
        figure.write_html(self.work_dir / "green_function_output.html")
        figure.write_image(self.work_dir / "green_function_output.png")

        return self
