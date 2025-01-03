from typing import List

import torch
from torch import nn

from .activations import get_activation


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model with customizable layers and activation functions.

    Parameters
    ----------
    layer_size : List[int]
        List specifying the sizes of each layer in the MLP. The first value is the input size, and the last value is the output size.
    activation : str, optional
        Name of the activation function to use (default is "rational"). Supported activations are defined in `get_activation`.

    Attributes
    ----------
    layers : nn.ModuleList
        List of fully connected layers in the MLP. Each layer is initialized with Xavier uniform initialization.
    activation : nn.Module
        Activation function used in the MLP, as defined by the `activation` parameter.

    Methods
    -------
    __init__(layer_size: List[int], activation: str = "rational")
        Initialize the MLP with specified layer sizes and activation function.
    forward(x: torch.Tensor) -> torch.Tensor
        Perform a forward pass through the MLP.
    """

    def __init__(self, layer_size: List[int], activation: str = "rational") -> None:
        """
        Initialize the MLP with the given layer sizes and activation function.

        Parameters
        ----------
        layer_size : List[int]
            List specifying the sizes of each layer in the MLP. The first value is the input size, and the last value is the output size.
        activation : str, optional
            Name of the activation function to use (default is "rational"). Supported activations are defined in `get_activation`.

        Attributes
        ----------
        layers : nn.ModuleList
            List of fully connected layers in the MLP. Each layer is initialized with Xavier uniform initialization.
        activation : nn.Module
            Activation function used in the MLP, as defined by the `activation` parameter.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(layer_size) - 1):
            linear_layer = nn.Linear(layer_size[i], layer_size[i + 1])
            nn.init.xavier_uniform_(linear_layer.weight)  # Xavier initialization
            self.layers.append(linear_layer)
            self.activations.append(get_activation(activation))

        # self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the MLP.

        The input tensor `x` is passed sequentially through each layer, followed by the activation function.
        The final layer does not apply the activation function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the MLP. The shape of `x` must match the input size defined in `layer_size`.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the MLP.

        Raises
        ------
        ValueError
            If the input tensor shape does not match the expected input size.

        Examples
        --------
        >>> layer_sizes = [10, 20, 5]
        >>> model = MLP(layer_sizes, activation="relu")
        >>> x = torch.randn(3, 10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([3, 5])
        """
        if x.shape[1] != self.layers[0].in_features:
            raise ValueError(
                f"Input size mismatch: Expected {self.layers[0].in_features}, but got {x.shape[1]}"
            )

        for layer, activation in zip(self.layers[:-1], self.activations):
            x = layer(x)
            x = activation(x)
            # x = self.activation(x)

        return self.layers[-1](x)
