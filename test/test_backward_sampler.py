import torch

from greenonet.axial import make_square_axial_lines
from greenonet.backward_sampler import BackwardSampler


class TestBackwardSampler:
    def test_generate_dataset_shapes_and_bc(self) -> None:
        lines = make_square_axial_lines(step_size=0.5, n_points_per_line=9)
        sampler = BackwardSampler(
            axial_lines=lines,
            data_size_per_each_line=2,
            scale_length=0.1,
            deterministic=True,
        )

        def a_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

        def zero_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        data = sampler.generate_dataset(
            a_fun=a_fun,
            ap_fun=zero_fun,
            b_fun=zero_fun,
            c_fun=zero_fun,
        )

        n_lines = len(lines.xaxial_lines)
        assert data.U.shape == (2, 2, n_lines, 9)
        assert data.F.shape == (2, 2, n_lines, 9)
        assert data.A.shape == (2, 2, n_lines, 9)
        assert data.AP.shape == (2, 2, n_lines, 9)
        assert data.B.shape == (2, 2, n_lines, 9)
        assert data.C.shape == (2, 2, n_lines, 9)
        assert data.COORDS.shape == (2, n_lines, 9, 2)
        assert torch.isfinite(data.U).all()
        assert torch.isfinite(data.F).all()
        assert data.U[..., 0].abs().max().item() < 1e-6
        assert data.U[..., -1].abs().max().item() < 1e-6

    def test_solve_bvp_line_matches_analytic_solution(self) -> None:
        lines = make_square_axial_lines(step_size=0.5, n_points_per_line=21)
        sampler = BackwardSampler(
            axial_lines=lines,
            data_size_per_each_line=1,
            scale_length=0.1,
            deterministic=True,
        )

        x = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
        u_exact = torch.sin(torch.pi * x)
        f = (torch.pi**2) * u_exact
        a_val = torch.ones_like(x)
        ap_val = torch.zeros_like(x)
        b_val = torch.zeros_like(x)
        c_val = torch.zeros_like(x)

        u_pred = sampler._solve_bvp_line(
            x=x,
            f=f,
            a_val=a_val,
            ap_val=ap_val,
            b_val=b_val,
            c_val=c_val,
        )
        max_abs = (u_pred - u_exact).abs().max().item()
        assert max_abs < 5e-3
        assert abs(float(u_pred[0])) < 1e-6
        assert abs(float(u_pred[-1])) < 1e-6
