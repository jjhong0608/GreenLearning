from typing import Union, List

import torch
from torch import nn


class Polyval:
    """Evaluate a polynomial at specific values using PyTorch.

    Parameters
    ----------
    coeffs : list of float
        Coefficients of the polynomial, ordered from highest degree to lowest.

    Methods
    -------
    evaluate(x: torch.Tensor) -> torch.Tensor
        Evaluate the polynomial for the input tensor `x`.
    """

    def __init__(self, coeffs: Union[List[float], float, torch.Tensor]) -> None:
        """
        Initialize the Polyval class with the coefficients of the polynomial.

        Parameters
        ----------
        coeffs : list of float or float
            Coefficients of the polynomial, ordered from highest degree to lowest.
        """
        self.coeffs = coeffs

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the polynomial at specific values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor where the polynomial is to be evaluated.

        Returns
        -------
        torch.Tensor
            The result of the polynomial evaluation.
        """
        result = torch.zeros_like(x)
        for i, coeff in enumerate(self.coeffs):
            result += coeff * x ** (len(self.coeffs) - i - 1)
        return result


class RationalActivation(nn.Module):
    """
    A PyTorch module implementing a rational activation function.

    Attributes
    ----------
    weights : Union[list, tuple]
        A tuple or list containing two lists of coefficients:
        - weights[0]: Coefficients of the numerator polynomial.
        - weights[1]: Coefficients of the denominator polynomial.

    Methods
    -------
    forward(x: torch.Tensor, weights: Union[list, tuple]) -> torch.Tensor
        Compute the rational activation given the input tensor `x` and weights.
    """

    def __init__(self):
        """Initialize the RationalActivation module."""
        super().__init__()
        self.weights = (
            nn.Parameter(torch.Tensor([1.1915, 1.5957, 0.5, 0.0218])),
            nn.Parameter(torch.Tensor([2.383, 0.0, 1.0])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the rational activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to which the rational activation is applied.

        Returns
        -------
        torch.Tensor
            Result of applying the rational activation function.
        """
        numerator = Polyval(self.weights[0])
        denominator = Polyval(self.weights[1])
        return torch.div(numerator.evaluate(x), denominator.evaluate(x))


def get_activation(name: str) -> nn.Module:
    """
    Retrieve a specified activation function by name.

    Parameters
    ----------
    name : str
        Name of the activation function. Options include:
        - 'relu': Rectified Linear Unit.
        - 'elu': Exponential Linear Unit.
        - 'selu': Scaled Exponential Linear Unit.
        - 'sigmoid': Sigmoid activation.
        - 'tanh': Hyperbolic tangent activation.
        - 'rational': Custom rational activation.

    Returns
    -------
    nn.Module
        The corresponding activation function module.

    Raises
    ------
    ValueError
        If the activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "rational": RationalActivation(),
    }
    if name in activations:
        return activations[name]
    else:
        raise ValueError(
            f"Unknown activation name: {name}. Available options are: {list(activations.keys())}"
        )
