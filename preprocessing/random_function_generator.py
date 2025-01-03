from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cholesky


class RandomFunctionGenerator:
    """
    A class to generate smooth random functions using a Cholesky factor of a covariance kernel.

    Attributes
    ----------
    domain : np.ndarray
        The domain where the function is defined.
    length_scale : float
        Length scale parameter for the RBF (Gaussian) kernel.
    covariance_matrix : np.ndarray
        Covariance matrix computed based on the domain and kernel.
    L : np.ndarray
        Cholesky factor of the covariance matrix.

    Methods
    -------
    rbf_kernel(x, y, length_scale=None)
        Computes the RBF kernel value between two points x and y.
    compute_covariance_matrix()
        Computes the covariance matrix for the domain using the RBF kernel.
    compute_cholesky_decomposition()
        Computes the Cholesky decomposition of the covariance matrix.
    generate_random_function()
        Generates a smooth random function using the Cholesky factor.
    """

    def __init__(self, domain: np.ndarray, length_scale: float = 0.1):
        """
        Initializes the RandomFunctionGenerator class.

        Parameters
        ----------
        domain : np.ndarray
            The domain where the function is defined.
        length_scale : float, optional
            Length scale parameter for the RBF (Gaussian) kernel (default is 0.1).
        """
        self.domain = domain
        self.length_scale = length_scale
        self.covariance_matrix = None
        self.L = None

    def rbf_kernel(
        self, x: float, y: float, length_scale: Optional[float] = None
    ) -> float:
        """
        Computes the RBF (Gaussian) kernel value between two points.

        Parameters
        ----------
        x : float
            First point.
        y : float
            Second point.
        length_scale : float, optional
            Length scale for the RBF kernel (default is None, which uses class attribute).

        Returns
        -------
        float
            The RBF kernel value.
        """
        if length_scale is None:
            length_scale = self.length_scale
        return np.exp(-((x - y) ** 2) / (2 * length_scale**2))

    def compute_covariance_matrix(self) -> "RandomFunctionGenerator":
        """
        Computes the covariance matrix for the domain using the RBF kernel.

        Optimized with joblib to improve performance.
        """

        def compute_row(i):
            return [self.rbf_kernel(self.domain[i], xj) for xj in self.domain]

        self.covariance_matrix = np.array(
            Parallel(n_jobs=-1)(
                delayed(compute_row)(i) for i in range(len(self.domain))
            )
        )

        return self

    def compute_cholesky_decomposition(self) -> "RandomFunctionGenerator":
        """
        Computes the Cholesky decomposition of the covariance matrix.
        """
        if self.covariance_matrix is None:
            raise ValueError(
                "Covariance matrix not computed. Call 'compute_covariance_matrix' first."
            )
        # Add jitter for numerical stability
        self.L = cholesky(
            self.covariance_matrix + 1e-8 * np.eye(len(self.domain)), lower=True
        )

        return self

    def generate_random_function(self) -> np.ndarray:
        """
        Generates a smooth random function using the Cholesky factor.

        Returns
        -------
        np.ndarray
            A smooth random function evaluated at the domain points.
        """
        if self.L is None:
            raise ValueError(
                "Cholesky factor not computed. Call 'compute_cholesky_decomposition' first."
            )
        # Generate random normal vector
        u = np.random.randn(self.L.shape[1])
        # Compute the smooth random function
        return self.L @ u


if __name__ == "__main__":
    RFG = RandomFunctionGenerator(np.linspace(0, 1, 100))
    func = (
        RFG.compute_covariance_matrix()
        .compute_cholesky_decomposition()
        .generate_random_function()
    )
    print(func)
