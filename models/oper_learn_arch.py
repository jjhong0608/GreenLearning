import torch
from torch import nn
from .MLP import MLP


class DeepOperatorLearningArchitecture(nn.Module):
    def __init__(self, rhs_size: int, num_layers: int, hidden_size: int):
        super().__init__()
        self.basis_mlp = MLP([rhs_size] + [hidden_size] * num_layers)
        self.coeff_mlp = MLP([1] + [hidden_size] * num_layers)

    def forward(self, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.inner(self.coeff_mlp(x), self.basis_mlp(rhs))
