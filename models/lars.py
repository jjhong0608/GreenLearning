from typing import Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    Implements layer-wise adaptive rate scaling for SGD.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    `Large Batch Training of Convolutional Networks:
    <https://arxiv.org/abs/1708.03888>`_.

    Attributes
    ----------
    eps : float
        A small number to avoid dividing by zero.

    Methods
    -------
    __init__(params, lr=0.3, momentum=0, weight_decay=0, dampening=0, eta=0.001, nesterov=False, eps=1e-8, lars_exclude=False)
        Initializes the LARS optimizer with the specified hyperparameters.
    step(closure=None)
        Performs a single optimization step.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.3,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        eta: float = 0.001,
        nesterov: bool = False,
        eps: float = 1e-8,
        lars_exclude: bool = False,
    ) -> None:
        """
        Initializes the LARS optimizer with the specified hyperparameters.

        Parameters
        ----------
        params : Iterable
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr : float
            Base learning rate. Must be non-negative.
        momentum : float
            Momentum factor. Must be non-negative.
        weight_decay : float
            Weight decay (L2 penalty). Must be non-negative.
        dampening : float
            Dampening for momentum. Defaults to 0.
        eta : float
            LARS coefficient. Must be non-negative.
        nesterov : bool
            Enables Nesterov momentum.
        eps : float
            A small number to avoid dividing by zero.
        lars_exclude : bool, optional
            Whether to exclude specific layers from LARS scaling, by default False.
        """
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS coefficient value: {eta}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            lars_exclude=lars_exclude,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.eps = eps
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        torch.Tensor or None
            The loss after the optimization step, if provided by the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            eta = group["eta"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            lars_exclude = group.get("lars_exclude", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # Calculate local learning rate if LARS scaling is not excluded
                if lars_exclude:
                    local_lr = 1.0
                else:
                    weight_norm = torch.norm(p)
                    grad_norm = torch.norm(d_p)
                    if torch.all(weight_norm != 0) and torch.all(grad_norm != 0):
                        # Compute local learning rate for this layer
                        local_lr = (
                            eta
                            * weight_norm
                            / (grad_norm + weight_decay * weight_norm + self.eps)
                        ).item()
                    else:
                        local_lr = 1.0

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                # Update with momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = buf

                # Update parameter
                p.add_(-d_p)
        return loss
