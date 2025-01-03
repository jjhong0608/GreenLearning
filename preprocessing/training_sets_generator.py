from pathlib import Path
from typing import Union

import numpy as np
from scipy.integrate import solve_bvp

from .interpolator import Interpolator
from .random_function_generator import RandomFunctionGenerator


class GreenLearningDatasetGenerator:
    """
    A class to generate datasets for Green's function learning examples.

    Attributes
    ----------
    domain : np.ndarray
        Domain of the operator, specified as a range [start, end].
    save_path : Path
        Path where the generated dataset will be saved.
    lambda_ : float
        Covariance kernel parameter for random function generation.
    linear : bool
        Indicates whether the operator is linear or nonlinear.
    Nf : int
        Number of training points for random sampled functions (F).
    Nu : int
        Number of evaluation points for solutions (U).
    Nsample : int
        Total number of random sampled functions and solutions to generate.

    Methods
    -------
    generate_dataset(diff_op, exact_green=None)
        Generates the dataset and saves it as a .npz file.
    """

    def __init__(
        self,
        domain: Union[list, np.ndarray],
        save_path: Union[str, Path] = "Examples/poisson.npz",
        lambda_: float = 0.03,
        linear: bool = True,
    ):
        """
        Initializes the GreenLearningDatasetGenerator.

        Parameters
        ----------
        domain : Union[list, np.ndarray]
            Domain of the operator, defined as a list or 1D array [start, end].
        save_path : Union[str, Path], optional
            File path to save the generated dataset. Default is 'Examples/example.npz'.
        lambda_ : float, optional
            Covariance kernel parameter. Default is 0.03.
        linear : bool, optional
            Whether the differential operator is linear. Default is True.
        """
        self.domain = domain  # Store the operator's domain
        self.lambda_ = lambda_  # Covariance kernel width
        self.linear = linear  # Linear or nonlinear nature of the operator
        # self.Nf = 201  # Number of points in sampled function F
        # self.Nu = 101  # Number of evaluation points in solution U
        self.Nf = 201  # Number of points in sampled function F
        self.Nu = 101  # Number of evaluation points in solution U
        self.Nsample = 100  # Number of samples to generate
        self.save_path = save_path if isinstance(save_path, Path) else Path(save_path)

    def generate_dataset(self):
        """
        Generates datasets for training and evaluation of Green's functions.

        This method creates datasets for sampled functions and solutions
        of differential equations using a random function generator and a
        boundary value problem solver.

        Saves
        -----
        A dataset as a .npz file with the generated sampled functions (F),
        solutions (U), and other relevant information.

        Raises
        ------
        ValueError
            If the specified save path does not have a '.npz' suffix.
        """
        # Initialize the random function generator
        RFG = RandomFunctionGenerator(
            np.linspace(self.domain[0], self.domain[1], self.Nf),
            length_scale=self.lambda_,
        )

        # Prepare covariance matrix and Cholesky decomposition
        RFG.compute_covariance_matrix().compute_cholesky_decomposition()

        # Create uniform grids for evaluation points
        X = np.linspace(self.domain[0], self.domain[1], self.Nu)
        Y = np.linspace(self.domain[0], self.domain[1], self.Nf)

        # Initialize storage for solutions (U) and sampled functions (F)
        U = np.zeros((self.Nu, self.Nsample))
        F = np.zeros((self.Nf, self.Nsample))

        # Coefficient constant for the equation
        K = 0
        domain = X.copy()
        singular = Y.copy()

        # Generate random functions and solve corresponding differential equations
        for i in range(self.Nsample):
            # Generate a random sampled function
            rhs = RFG.generate_random_function()
            interpolator = Interpolator(singular, rhs)

            # Define the differential equation
            def fun(x, y):
                # Helmholts Equation
                return np.vstack((y[1], interpolator.evaluate(x) - (K**2) * y[0]))

            # Define the boundary conditions
            def bc(ya, yb):
                return np.array((ya[0], yb[0]))

            # Solve the boundary value problem
            u = solve_bvp(
                fun, bc, domain, np.zeros((2, domain.size)), tol=1e-13, bc_tol=1e-13
            ).sol(domain)

            # Store the solution and the sampled function
            U[:, i] = u[0]
            F[:, i] = rhs

        # Solve homogeneous boundary problem
        zero_bc = lambda ya, yb: np.array((ya[0], yb[0]))
        # Helmholts Equation
        hom_fun = lambda x, y: np.vstack((y[1], -(K**2) * y[0]))
        u_hom = solve_bvp(
            hom_fun,
            zero_bc,
            domain,
            np.zeros((2, domain.size)),
            tol=1e-13,
            bc_tol=1e-13,
        ).sol(domain)[0]

        print(f"u_hom: {u_hom}")

        # Normalize solutions if the operator is linear
        if np.all(np.isclose(u_hom, 0.0)) and self.linear:
            print("Solving homogeneous boundary problem for normalization.")
            scale = np.abs(U).max()
            U /= scale
            F /= scale

        # Create parent directory if it doesn't exist
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure file ends with '.npz'
        if not self.save_path.suffix == ".npz":
            raise ValueError(f"save path must have suffix '.npz'")

        # Save dataset as a .npz file
        np.savez(
            self.save_path,
            X=X,  # Evaluation points
            Y=Y,  # Sample points
            U=U,  # Solutions
            F=F,  # Sampled functions
            HOM_U=u_hom,  # Homogeneous solution
        )
        print(f"Dataset saved to {self.save_path}")


if __name__ == "__main__":
    # Example usage
    domain = [0, 1]
    generator = GreenLearningDatasetGenerator(domain)
    generator.generate_dataset()
