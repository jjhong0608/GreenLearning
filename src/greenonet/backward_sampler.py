from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.integrate import solve_bvp
from torch import Tensor

from greenonet.sampler import ForwardSampler


class BackwardSampler(ForwardSampler):
    """
    Backward sampler for line-wise BVP generation.

    Steps per axial line:
    1. Sample RHS f from a GP-like RBF expansion.
    2. Solve -d/dx(a(x)u') + b(x)u' + c(x)u = f(x), u(0)=u(1)=0 via solve_bvp.
    3. Normalize (u, f) with the same rule as the forward sampler.
    """

    BVP_TOL = 1e-5
    BVP_MAX_NODES = 10000
    BVP_RETRIES = 3
    A_EPS = 1e-8

    def _sample_rhs(self, x: Tensor) -> Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        alpha = torch.randn(x.shape, device=self.device, dtype=self.dtype)
        centers = self._centers(x)
        scale_length = self._sample_scale_length()
        k = self._rbf_kernel(x, centers, scale_length)
        return k @ alpha

    def _solve_bvp_line(
        self,
        x: Tensor,
        f: Tensor,
        a_val: Tensor,
        ap_val: Tensor,
        b_val: Tensor,
        c_val: Tensor,
    ) -> Tensor:
        x_np = x.detach().cpu().numpy().astype(np.float64)
        f_np = f.detach().cpu().numpy().astype(np.float64)
        a_np = a_val.detach().cpu().numpy().astype(np.float64)
        ap_np = ap_val.detach().cpu().numpy().astype(np.float64)
        b_np = b_val.detach().cpu().numpy().astype(np.float64)
        c_np = c_val.detach().cpu().numpy().astype(np.float64)

        # Avoid numerical blow-up when coefficient is too close to zero.
        a_np = np.where(np.abs(a_np) < self.A_EPS, np.sign(a_np) * self.A_EPS, a_np)
        a_np = np.where(a_np == 0.0, self.A_EPS, a_np)

        def ode(x_eval: np.ndarray, y_eval: np.ndarray) -> np.ndarray:
            a_i = np.interp(x_eval, x_np, a_np)
            ap_i = np.interp(x_eval, x_np, ap_np)
            b_i = np.interp(x_eval, x_np, b_np)
            c_i = np.interp(x_eval, x_np, c_np)
            f_i = np.interp(x_eval, x_np, f_np)
            u_i = y_eval[0]
            du_i = y_eval[1]
            ddu_i = ((b_i - ap_i) * du_i + c_i * u_i - f_i) / a_i
            return np.vstack((du_i, ddu_i))

        def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[0], yb[0]], dtype=np.float64)

        y_guess = np.zeros((2, x_np.size), dtype=np.float64)
        amp = float(np.max(np.abs(f_np))) if f_np.size > 0 else 1.0
        amp = max(amp, 1e-6)

        for attempt in range(self.BVP_RETRIES):
            if attempt > 0:
                scale = 1e-2 * (attempt + 1)
                seed_u = scale * amp * x_np * (1.0 - x_np)
                seed_du = np.gradient(seed_u, x_np)
                y_guess = np.vstack((seed_u, seed_du))
            sol = solve_bvp(
                ode,
                bc,
                x_np,
                y_guess,
                tol=self.BVP_TOL,
                max_nodes=self.BVP_MAX_NODES,
                verbose=0,
            )
            if sol.success:
                u_np = sol.sol(x_np)[0]
                return torch.from_numpy(u_np).to(device=self.device, dtype=self.dtype)

        raise RuntimeError(
            "BackwardSampler BVP solve failed "
            f"(retries={self.BVP_RETRIES}, points={x_np.size})."
        )

    def generate_sample(
        self,
        x: Tensor,
        a_fun: Callable[[Tensor], Tensor],
        ap_fun: Callable[[Tensor], Tensor],
        b_fun: Callable[[Tensor], Tensor],
        c_fun: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = x.to(device=self.device, dtype=self.dtype)
        f = self._sample_rhs(x)

        a_val = a_fun(x)
        ap_val = ap_fun(x)
        b_val = b_fun(x)
        c_val = c_fun(x)

        u = self._solve_bvp_line(
            x=x,
            f=f,
            a_val=a_val,
            ap_val=ap_val,
            b_val=b_val,
            c_val=c_val,
        )

        u = u.detach()
        f = f.detach()
        u, f = self._normalize_sample(x, u, f)
        return u, f, a_val.detach(), ap_val.detach(), b_val.detach(), c_val.detach()
