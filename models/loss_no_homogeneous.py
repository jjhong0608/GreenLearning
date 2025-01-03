import torch


class Loss:
    """
    Loss function for learning Green's function and homogeneous solutions.

    This class calculates the relative error between the predicted solutions
    (derived from the Green's function and homogeneous solution) and the actual solutions.

    Attributes
    ----------
    x : torch.Tensor
        The spatial domain tensor, representing discretized points along the domain. Shape: (number_of_points,).
    y : torch.Tensor
        The singular points tensor, representing specific locations in the domain. Shape: (number_of_singular_points,).
    f : torch.Tensor
        The right-hand side tensor, which specifies the forcing terms. Shape: (number_of_singular_points, number_of_samples).
    u : torch.Tensor
        The actual solution tensor, representing ground truth values. Shape: (number_of_points, number_of_samples).

    Methods
    -------
    __call__(green_output: torch.Tensor, hom_output: torch.Tensor) -> torch.Tensor
        Computes the mean relative error between predicted and actual solutions.
    """

    def __init__(
        self,
        spatial_dom: torch.Tensor,
        singular_points: torch.Tensor,
        rhs: torch.Tensor,
        sol: torch.Tensor,
        hom_sol: torch.Tensor,
    ) -> None:
        """
        Initializes the Loss class with necessary tensors.

        Parameters
        ----------
        spatial_dom : torch.Tensor
            The spatial domain tensor representing the discretized points.
            Shape: (number_of_points,).
        singular_points : torch.Tensor
            Tensor of singular points in the spatial domain.
            Shape: (number_of_singular_points,).
        rhs : torch.Tensor
            The right-hand side tensor, indicating forcing terms.
            Shape: (number_of_singular_points, number_of_samples).
        sol : torch.Tensor
            The actual solution tensor representing the ground truth values.
            Shape: (number_of_points, number_of_samples).
        """
        # Flattening ensures compatibility with tensor operations, even if higher dimensions are given.
        self.x = spatial_dom.flatten() if spatial_dom.dim() > 1 else spatial_dom
        self.y = (
            singular_points.flatten() if singular_points.dim() > 1 else singular_points
        )
        self.f = rhs
        self.u = sol
        self.hom_sol = hom_sol.reshape(-1, 1)

    def __call__(self, green_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative error between predicted solutions and actual solutions.

        The method integrates the Green's function output with the right-hand side (RHS)
        and combines it with the homogeneous solution. It then calculates the relative error
        by comparing the predicted and actual solutions.

        Parameters
        ----------
        green_output : torch.Tensor
            The output tensor from the Green's function model.
            Shape: (number_of_singular_points * number_of_points, 2).

        Returns
        -------
        torch.Tensor
            The mean relative error between predicted and actual solutions as a scalar.
        """
        # Reshape Green's function output as (number_of_singular_points, number_of_points)
        green_output = green_output.reshape(self.u.shape[0], self.f.shape[0]).T

        # Green's function output needs to match the dimensions of RHS for element-wise operations.
        green_rhs = self.f.unsqueeze(1) * green_output.unsqueeze(2)

        # Use the trapezoidal rule to approximate the integral, as it balances accuracy and efficiency.
        integ_green_rhs = torch.trapezoid(y=green_rhs, x=self.y, dim=0)

        # Residuals are computed by subtracting integrated Green's function and homogeneous solution from actual solutions.
        residual = self.u - integ_green_rhs - self.hom_sol

        # Compute the relative error as the mean over all samples, normalized by the actual solution norm.
        mean_relative_error = torch.mean(
            torch.sum(residual.square(), dim=0) / torch.sum(self.u.square(), dim=0)
        )

        return mean_relative_error


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    x = torch.linspace(0, 1, 51)
    y = torch.linspace(0, 1, 71)
    n_sample = 2
    f1 = lambda x: 4 * torch.pi**2 * torch.sin(x * torch.pi * 2)
    f2 = lambda x: 4 * torch.pi**2 * torch.cos(x * torch.pi * 2)
    u1 = lambda x: torch.sin(x * torch.pi * 2)
    u2 = lambda x: torch.cos(x * torch.pi * 2) - 1

    f = torch.stack([f1(y), f2(y)], dim=1)
    u = torch.stack([u1(x), u2(x)], dim=1)

    print(f"f.shape: {f.shape}")
    print(f"u.shape: {u.shape}")

    green_output = torch.zeros((y.shape[0], x.shape[0]))
    hom_output = torch.zeros((x.shape[0], 1))
    loss = Loss(x, y, f, u, hom_output)
    for i in range(green_output.shape[0]):
        for j in range(green_output.shape[1]):
            green_output[i, j] = x[j] * (1 - y[i]) if x[j] < y[i] else y[i] * (1 - x[j])

    green_output = green_output.T.reshape(-1, 1)
    print(loss(green_output))
