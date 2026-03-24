import torch

from greenonet.axial import make_square_axial_lines
from greenonet.sampler import ForwardSampler


class TestForwardSamplerHelpers:
    def test_normalization_guard(self) -> None:
        sampler = ForwardSampler(
            axial_lines=make_square_axial_lines(step_size=0.5),
            data_size_per_each_line=1,
            scale_length=0.1,
        )
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        small = torch.tensor([1e-12, -1e-12], dtype=torch.float64)
        u_norm, f_norm = sampler._normalize_sample(x, small, small)
        assert torch.isfinite(u_norm).all()
        assert torch.isfinite(f_norm).all()
        assert u_norm.abs().max().item() <= 1.0 + 1e-9

    def test_simpson_integral(self) -> None:
        sampler = ForwardSampler(
            axial_lines=make_square_axial_lines(step_size=0.5),
            data_size_per_each_line=1,
            scale_length=0.1,
        )
        x = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
        y = x.clone()
        area = sampler._simpson_integral(x, y)
        assert torch.isclose(area, torch.tensor(0.5, dtype=torch.float64), atol=1e-12)

    def test_scale_length_sampling_range(self) -> None:
        sampler = ForwardSampler(
            axial_lines=make_square_axial_lines(step_size=0.5),
            data_size_per_each_line=1,
            scale_length=(0.05, 0.2),
        )
        values = [sampler._sample_scale_length() for _ in range(10)]
        assert all(0.05 <= v <= 0.2 for v in values)


class TestForwardSamplerDataset:
    def test_coords_match_axial_lines(self) -> None:
        lines = make_square_axial_lines(step_size=0.5)
        sampler = ForwardSampler(
            axial_lines=lines,
            data_size_per_each_line=1,
            scale_length=0.1,
            deterministic=True,
        )

        def zeros(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        data = sampler.generate_dataset(
            a_fun=zeros,
            ap_fun=zeros,
            b_fun=zeros,
            c_fun=zeros,
        )
        # First x-line sample coordinates should match the stored line coords
        line_coords = lines.xaxial_lines[0].coordinates
        # COORDS shape: (2, n_lines, m_points, 2)
        assert torch.allclose(data.COORDS[0, 0], line_coords)
        # Finite values after normalization with positive energy
        energy = sampler._simpson_integral(
            lines.xaxial_lines[0].x_coordinates, data.U[0, 0, 0] ** 2
        )
        assert torch.isfinite(data.U).all()
        assert energy > 0.0

    def test_coefficients_constant_across_samples(self) -> None:
        lines = make_square_axial_lines(step_size=0.5)
        sampler = ForwardSampler(
            axial_lines=lines,
            data_size_per_each_line=2,
            scale_length=0.1,
            deterministic=True,
        )

        def coeff_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return 1.0 + 0.0 * x + 0.0 * y

        data = sampler.generate_dataset(
            a_fun=coeff_fn,
            ap_fun=coeff_fn,
            b_fun=coeff_fn,
            c_fun=coeff_fn,
        )
        # All samples on the same line should share identical coefficients.
        assert torch.allclose(data.A[0], data.A[1])
        assert torch.allclose(data.AP[0], data.AP[1])
        assert torch.allclose(data.B[0], data.B[1])
        assert torch.allclose(data.C[0], data.C[1])
