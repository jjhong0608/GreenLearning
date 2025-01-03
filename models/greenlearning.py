import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
from plotly import graph_objects as go
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn
from safetensors.torch import save_model, load_model
from torch import optim

from .MLP import MLP
from .loss import Loss


class GreenLearning:
    """
    A class for learning Green's functions and homogeneous solutions using neural networks.

    Attributes
    ----------
    green_network : MLP
        Neural network for modeling the Green's function.
    hom_network : MLP
        Neural network for modeling the homogeneous solution.
    adam_epochs : int
        Number of epochs for the Adam optimizer.
    lbfgs_epochs : int
        Number of epochs for the L-BFGS optimizer.
    work_dir : pathlib.Path
        Directory where logs and outputs are saved.
    device : torch.device
        The device used for computation, either 'cpu' or 'cuda'.

    Methods
    -------
    load_data(path: Union[str, pathlib.Path]) -> None
        Loads the input data from a .npz file.
    train() -> None
        Trains the networks using Adam and L-BFGS optimizers.
    """

    def __init__(
        self,
        green_layers: List[int],
        hom_layers: List[int],
        activation: str = "rational",
        adam_epochs: int = 1000,
        lbfgs_epochs: int = 5000,
        work_dir: Union[str, Path] = "work_dir",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initializes the GreenLearning class with specified network architectures and training parameters.

        Parameters
        ----------
        green_layers : List[int]
            List defining the architecture of the Green's function network.
        hom_layers : List[int]
            List defining the architecture of the homogeneous solution network.
        activation : str, optional
            Activation function used in the networks. Default is "rational".
        adam_epochs : int, optional
            Number of epochs for the Adam optimizer. Default is 1000.
        lbfgs_epochs : int, optional
            Number of epochs for the L-BFGS optimizer. Default is 50000.
        work_dir : Union[str, pathlib.Path], optional
            Directory where logs and outputs will be saved. Default is "work_dir".
        device : torch.device, optional
            Device used for computation, either 'cpu' or 'cuda'. Default is 'cpu'.
        """
        super().__init__()

        # Initialize networks
        self.green_network = MLP(green_layers, activation=activation).to(device)
        self.hom_network = MLP(hom_layers, activation=activation).to(device)

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

        # Initialize optimizers
        self._initialize_optimizers()

    def load_data(self, path: Union[str, Path]) -> "GreenLearning":
        """
        Loads data from a .npz file and prepares the training input.

        This method is responsible for loading the input data required for training
        the Green's function and homogeneous solution models. It validates the input
        file format, reads the data, and constructs the training input tensor.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            Path to the .npz file containing training data. The file should include
            the following arrays:
            - X: Discretized spatial domain points.
            - Y: Singular points within the domain.
            - U: Ground truth solution values.
            - F: Forcing terms or right-hand side values.

        Raises
        ------
        ValueError
            If the provided file is not in .npz format, an exception is raised to
            ensure correct data input.

        Side Effects
        ------------
        - Logs the file loading process using the logging module.
        - Converts the loaded data into PyTorch tensors and moves them to the
          specified device (CPU or CUDA).
        - Prepares a training input tensor by generating a grid of points from
          X and Y using `torch.meshgrid`.

        Returns
        -------
        self : GreenLearning
            Returns the instance of the GreenLearning class for method chaining.

        Example
        -------
        >>> gl = GreenLearning(...)
        >>> gl.load_data("data/training_data.npz")
        """
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

        # Generate training input by creating a grid from X and Y
        self.training_input = (
            torch.stack(
                torch.meshgrid(self.x, self.y, indexing="ij"), dim=2
            )  # Create a grid of (X, Y)
            .reshape(-1, 2)  # Flatten the grid to a 2D tensor of points
            .to(self.device)  # Move tensor to the specified device
        )

        return self

    def _initialize_optimizers(self) -> "GreenLearning":
        """
        Initializes Adam and L-BFGS optimizers for training.

        This method sets up two optimizers to train the Green's function and homogeneous
        solution networks. The Adam optimizer is used initially for fast convergence during
        the early training stages, and the L-BFGS optimizer is employed for fine-tuning.

        Optimizer Details
        -----------------
        1. Adam Optimizer:
           - Learning rate: 1e-3
           - Parameters: All trainable parameters of both the Green's function and
             homogeneous solution networks.

        2. L-BFGS Optimizer:
           - Learning rate: 1.0
           - Maximum iterations: `lbfgs_epochs` (total number of iterations for fine-tuning).
           - Gradient tolerance: 1e-8 (stopping criterion for gradients).
           - History size: 100 (number of past gradients used for approximating the Hessian).
           - Line search: "strong_wolfe" (ensures step size satisfies Wolfe conditions).

        Returns
        -------
        self : GreenLearning
            Returns the instance of the GreenLearning class for method chaining.
        """
        parameters = [
            {"params": self.green_network.parameters(), "lr": 1e-3},
            {"params": self.hom_network.parameters(), "lr": 1e-3},
        ]
        # Initialize the Adam optimizer
        self.adam = optim.Adam(parameters)

        # Initialize the L-BFGS optimizer with specified hyperparameters
        self.lbfgs = optim.LBFGS(
            list(self.green_network.parameters()) + list(self.hom_network.parameters()),
            lr=1.0,
            max_iter=self.lbfgs_epochs,
            max_eval=10**5,
            # max_eval=1250
            # * 1000
            # // 15000,  # Controls the maximum number of function evaluations
            tolerance_grad=1.0 * np.finfo(float).eps,  # Tolerance for gradient norm
            # tolerance_grad=1e-8,  # Tolerance for gradient norm
            tolerance_change=1e-20,  # Minimum change in loss for stopping
            history_size=100,  # Number of past updates to store for Hessian approximation
            line_search_fn="strong_wolfe",  # Line search strategy for step size determination
        )

        return self

    def green_function_output(
        self, path: Optional[Union[str, Path]] = None
    ) -> "GreenLearning":
        """
        Generates and saves the Green's function output to a specified directory.

        This method computes the Green's function values over a finely discretized
        spatial domain and saves the results in a `.npz` file.

        Parameters
        ----------
        path : Optional[Union[str, Path]]
            Directory where the Green's function output will be saved.

        Raises
        ------
        ValueError
            If the specified path is not a directory.

        Side Effects
        ------------
        - Creates the specified directory if it does not exist.
        - Saves the Green's function output as an `.npz` file in the directory.

        Returns
        -------
        self : GreenLearning
            Returns the instance of the GreenLearning class for method chaining.
        """
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

    def save_model(self, path: Optional[Union[str, Path]] = None) -> "GreenLearning":
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
        return self

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

    def train(self) -> "GreenLearning":
        """
        Trains the Green's function and homogeneous solution neural networks.

        This method performs a two-stage training process:
        1. Optimizes the networks using the Adam optimizer for fast convergence.
        2. Fine-tunes the networks using the L-BFGS optimizer for high precision.

        The training process includes:
        - Calculating the loss using a custom loss function that computes relative error
          between predicted and actual solutions.
        - Updating the neural network weights to minimize the loss.
        - Logging the progress and loss values to a log file.
        - Displaying a progress bar using the `rich` library.

        Returns
        -------
        self : GreenLearning
            Returns the instance of the GreenLearning class for method chaining.
        """
        # Initialize the custom loss function with necessary tensors.
        # The loss function uses domain points, singular points, RHS terms, and ground truth solutions.
        criterion = Loss(
            spatial_dom=self.x, singular_points=self.y, rhs=self.f, sol=self.u
        )

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
                total=self.lbfgs_epochs,
            )

            # Phase 1: Training using the Adam optimizer.
            for i in range(1, self.adam_epochs + 1):
                self.adam.zero_grad()  # Reset gradients from the previous step.

                # Compute outputs from the Green's function and homogeneous solution networks.
                green_output = self.green_network(self.training_input)
                hom_output = self.hom_network(self.x.view(-1, 1))

                # Calculate the loss between predicted and actual solutions.
                loss = criterion(green_output, hom_output)

                # Perform backpropagation to compute gradients.
                loss.backward()

                # Update the weights of the networks using the Adam optimizer.
                self.adam.step()

                # Log and update the progress bar for the current epoch.
                description = f"Epoch {i:>6}/{self.adam_epochs} - Loss: {loss:.4f}"
                logging.info(description, extra={"file": __file__, "function": "train"})
                progress.update(task, advance=1, description=f"[green]{description}")

            # Set the starting epoch for L-BFGS optimization.
            self.lbfgs_step = self.adam_epochs + 1

            # Phase 2: Fine-tuning using the L-BFGS optimizer.
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
                green_output = self.green_network(self.training_input)
                hom_output = self.hom_network(self.x.view(-1, 1))

                # Calculate the loss between predicted and actual solutions.
                loss = criterion(green_output, hom_output)

                # Perform backpropagation to compute gradients.
                loss.backward()

                # Update the description and progress bar for the current epoch.
                description = (
                    f"Epoch {self.lbfgs_step:>6}/{self.lbfgs_epochs} - Loss: {loss:.4f}"
                )
                self.lbfgs_step += 1
                progress.update(task, advance=1, description=f"[green]{description}")

                return loss

            # Perform the optimization using the L-BFGS optimizer and the closure function.
            while self.lbfgs_step <= self.lbfgs_epochs:
                self.lbfgs.step(closure)

        # Return the class instance to allow method chaining.
        return self

    def plot_green_function(self, path: Optional[Union[str, Path]] = None) -> None:
        if path is None:
            path = self.work_dir

        if not isinstance(path, Path):
            path = Path(path)

        x = np.linspace(self.x[0], self.x[-1], 1001)
        y = np.linspace(self.y[0], self.y[-1], 1001)

        k = 15
        exact_green = (k * np.sin(k)) ** (-1) * np.expand_dims(
            np.sin(k * x), 1
        ) * np.expand_dims(np.sin(k * (y - 1)), 0) * (
            np.expand_dims(x, 1) <= np.expand_dims(y, 0)
        ) + (
            k * np.sin(k)
        ) ** (
            -1
        ) * np.expand_dims(
            np.sin(k * y), 0
        ) * np.expand_dims(
            np.sin(k * (x - 1)), 1
        ) * (
            np.expand_dims(x, 1) > np.expand_dims(y, 0)
        )

        th_x = torch.tensor(x, device=self.device)
        th_y = torch.tensor(y, device=self.device)
        _input = (
            torch.stack(
                torch.meshgrid(th_x, th_y, indexing="ij"), dim=2
            )  # Create a 2D grid of (x, y) pairs.
            .reshape(-1, 2)  # Flatten the grid into a list of 2D points.
            .to(
                self.device
            )  # Move the tensor to the specified device (e.g., CPU or GPU).
        )

        with torch.no_grad():
            green_output = self.green_network(_input).reshape(x.shape[0], y.shape[0])

        green_output = green_output.detach().cpu().numpy()
        error = np.abs(exact_green - green_output)

        print(f"exact_green: {exact_green.shape}")
        print(f"green_output: {green_output.shape}")
        print(f"error: {error.shape}")

        data = [
            go.Heatmap(
                x=x,
                y=y,
                z=error,
                # showscale=False,
            )
        ]
        layout = go.Layout(
            title="Green's function error",
            template="plotly_white",
            width=1500,
            height=1500,
            font=go.layout.Font(size=25, family="Times New Roman", weight="bold"),
        )
        figure = go.Figure(data=data, layout=layout)
        figure.write_image(path / "green_function_error.png")
