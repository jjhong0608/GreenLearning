from typing import Iterable, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer


class LevenbergMarquardt(Optimizer):
    """
    Implements the Levenberg-Marquardt optimization algorithm.

    Attributes
    ----------
    step_count : int
        Counter for the optimization steps.
    old_loss : float
        Loss value from the previous step.
    _loss : float
        Current loss value.

    Parameters
    ----------
    params : Iterable
        Model parameters to optimize.
    lr : float, optional
        Learning rate (default: 1e-3).
    mu : float, optional
        Initial damping parameter (default: 1e5).
    mu_update_interval : int, optional
        Number of steps before updating mu (default: 2).
    div_factor : float, optional
        Factor to reduce mu when loss decreases (default: 1.3).
    mul_factor : float, optional
        Factor to increase mu when loss increases (default: 3.0).

    Methods
    -------
    step(closure: Optional[Callable[[], float]] = None) -> float
        Performs a single optimization step.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        mu: float = 1e0,
        mu_update_interval: int = 2,
        div_factor: float = 1.3,
        mul_factor: float = 3.0,
    ):
        """
        Initialize the Levenberg-Marquardt optimizer with parameters.

        Parameters
        ----------
        params : Iterable
            Model parameters to optimize.
        lr : float, optional
            Learning rate for the optimizer.
        mu : float, optional
            Damping parameter controlling convergence.
        mu_update_interval : int, optional
            Interval for updating the damping parameter (μ).
        div_factor : float, optional
            Dividing factor for decreasing μ when the loss decreases.
        mul_factor : float, optional
            Multiplying factor for increasing μ when the loss increases.
        """
        defaults = dict(
            lr=lr,
            mu=mu,
            mu_update_interval=mu_update_interval,
            div_factor=div_factor,
            mul_factor=mul_factor,
        )
        super().__init__(params, defaults)
        self.step_count = 0  # Count the optimization steps
        self.old_loss = mu  # Store previous loss for comparison
        self._loss = mu  # Current loss value

    @property
    def loss(self) -> float:
        """Getter for the loss value."""
        return self._loss

    @loss.setter
    def loss(self, value: float):
        """Setter for the loss value."""
        self._loss = value

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> float:
        """
        Performs a single optimization step using the Levenberg-Marquardt algorithm.

        The algorithm adjusts the parameters to minimize the loss function using
        a combination of gradient descent and the Gauss-Newton method, dampened
        by a dynamically adjusted parameter `mu`.

        Parameters
        ----------
        closure : Optional[Callable[[], float]]
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        float
            The computed loss value after optimization step.
        """
        # Re-evaluate the loss if a closure is provided
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = self._loss

        if loss is None:
            raise RuntimeError("No loss value provided.")

        grads = []
        idxs = [0]
        for group in self.param_groups:
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grads.append(p.grad.view(-1))
                idxs.append(idxs[-1] + p.numel())

        grads = torch.cat(grads).reshape(1, -1)
        jtj = torch.matmul(grads.T, grads)
        diag_jtj = torch.diag(torch.diag(jtj))
        identity = torch.eye(jtj.size(0), device=jtj.device)
        # jtj_damped = jtj + mu * diag_jtj
        jtj_damped = jtj + mu * identity

        test = torch.matmul(grads.T, torch.tensor(loss).reshape(1, 1))
        # print(f"jtj_damped: {jtj_damped.shape}\n{jtj_damped}")
        # print(f"test: {test.shape}\n{test}")
        dps = torch.linalg.solve(
            jtj_damped, torch.matmul(grads.T, torch.tensor(loss).reshape(1, 1))
        ).reshape(-1)
        # jtj_inv = torch.linalg.solve(jtj_damped, identity)
        # dps = torch.matmul(jtj_inv, grads.T).reshape(-1)

        idx = 1
        # Iterate over the parameter groups in the optimizer
        for group in self.param_groups:
            # Extract hyperparameters for the current group
            mu = group["mu"]
            lr = group["lr"]
            mu_update_interval = group["mu_update_interval"]
            div_factor = group["div_factor"]
            mul_factor = group["mul_factor"]

            # Loop through the parameters in the group
            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip parameters with no gradients

                # Update the parameters
                p.add_(dps[idxs[idx - 1] : idxs[idx]].reshape(p.shape), alpha=-lr)
                idx += 1
                # print(f"d_p: {d_p}")

        # Update μ dynamically based on loss improvement
        if self.step_count % mu_update_interval == 0:
            if loss < self.old_loss:
                # Decrease μ when loss decreases
                mu = max(mu / div_factor, 1e-9)
            else:
                # Increase μ when loss increases
                mu = min(mu * mul_factor, 1e8)
            self.old_loss = loss  # Update old loss value
            group["mu"] = mu

        self.step_count += 1  # Increment step counter
        return loss  # Return the updated loss
