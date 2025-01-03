import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler for learning rate.

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_iters : int
        Number of iterations for the warmup phase.
    start_factor : float
        Starting factor for the learning rate.
    end_factor : float
        Ending factor for the learning rate.
    last_epoch : int, optional
        The index of the last epoch. Default: -1.

    Methods
    -------
    __init__(optimizer, warmup_iters, start_factor=1.0/3, end_factor=1.0, last_epoch=-1)
        Initializes the LinearWarmupScheduler with the optimizer and other parameters.
    get_lr() -> list
        Computes the learning rate during the warmup phase.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Initializes the LinearWarmupScheduler with the given parameters.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer.
        warmup_iters : int
            Number of iterations for the warmup phase.
        start_factor : float, optional
            Starting factor for the learning rate, by default 1.0 / 3.
        end_factor : float, optional
            Ending factor for the learning rate, by default 1.0.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        assert (
            warmup_iters > 0
        ), "warmup_iters must be greater than zero to avoid division by zero."
        assert 0 <= start_factor <= 1.0, "start_factor must be between 0 and 1."
        assert 0 <= end_factor <= 1.0, "end_factor must be between 0 and 1."
        self.warmup_iters = warmup_iters
        self.start_factor = start_factor
        self.end_factor = end_factor
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """
        Computes the learning rate during the warmup phase.

        Returns
        -------
        list of float
            A list of learning rates for each parameter group.
        """
        if self.last_epoch == 0:
            lrs = [base_lr * self.start_factor for base_lr in self.base_lrs]
        else:
            lrs = [
                base_lr
                * (
                    1.0
                    + (self.end_factor - self.start_factor)
                    / (
                        self.warmup_iters * self.start_factor
                        + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                    )
                )
                for base_lr in self.base_lrs
            ]
        for i in range(len(self.base_lrs)):
            self.base_lrs[i] = lrs[i]
        return lrs


class WarmupCosineAnnealingScheduler:
    """
    Scheduler combining Linear Warmup and Cosine Annealing phases.

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_iters : int
        Number of iterations for the warmup phase.
    T_max : int
        Maximum number of iterations for the Cosine Annealing phase.
    eta_min : float, optional
        Minimum learning rate at the end of the Cosine Annealing phase.
    warmup_scheduler : LinearWarmupScheduler
        Linear warmup scheduler instance.
    cosine_scheduler : CosineAnnealingLR
        Cosine annealing scheduler instance.

    Methods
    -------
    __init__(optimizer, warmup_iters, T_max, start_factor=1.0/3, eta_min=0.0, last_epoch=-1)
        Initializes the WarmupCosineAnnealingScheduler with both warmup and cosine annealing phases.
    step(iteration)
        Updates the learning rate based on the current iteration.
    get_last_lr() -> list
        Gets the last computed learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        T_max: int,
        start_factor: float = 1.0 / 3,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initializes the scheduler combining Linear Warmup and Cosine Annealing phases.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer.
        warmup_iters : int
            Number of iterations for the warmup phase.
        T_max : int
            Maximum number of iterations for the Cosine Annealing phase.
        start_factor : float, optional
            Starting factor for the learning rate during warmup, by default 1.0 / 3.
        eta_min : float, optional
            Minimum learning rate at the end of the Cosine Annealing phase, by default 0.0.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        assert (
            T_max > warmup_iters
        ), f"T_max ({T_max}) must be greater than warmup_iters ({warmup_iters}) to avoid negative values."
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.T_max = (
            T_max - warmup_iters
        )  # Adjust T_max to account for the warmup iterations
        self.eta_min = eta_min
        self.warmup_scheduler = LinearWarmupScheduler(
            optimizer, warmup_iters, start_factor=start_factor
        )
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, self.T_max, eta_min=self.eta_min, last_epoch=last_epoch
        )
        self.last_lr = self.warmup_scheduler.get_last_lr()

    def step(self, iteration: int) -> None:
        """
        Updates the learning rate based on the current iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        if iteration < self.warmup_scheduler.warmup_iters:
            self.warmup_scheduler.step()
            self.last_lr = self.warmup_scheduler.get_last_lr()
        else:
            self.cosine_scheduler.step()
            self.last_lr = self.cosine_scheduler.get_last_lr()

    def get_last_lr(self) -> list:
        """
        Gets the last computed learning rate.

        Returns
        -------
        list of float
            A list of learning rates for each parameter group.
        """
        return self.last_lr
