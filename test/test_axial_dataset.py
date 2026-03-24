import torch

from greenonet.axial import make_square_axial_lines
from greenonet.sampler import ForwardSampler
from greenonet.data import AxialDataset, axial_collate_fn


def test_axial_dataset_packing_shapes() -> None:
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
    dataset = AxialDataset(data)
    coords, solutions, sources, a_vals, ap_vals, b_vals, c_vals = dataset[0]
    # coords shape: (2, n_lines, m_points, 2)
    assert coords.dim() == 4
    assert coords.shape[0] == 2
    assert solutions.shape[:3] == coords.shape[:3]
    assert sources.shape[:3] == coords.shape[:3]
    assert a_vals.shape[:3] == coords.shape[:3]
    assert ap_vals.shape[:3] == coords.shape[:3]
    assert b_vals.shape[:3] == coords.shape[:3]
    assert c_vals.shape[:3] == coords.shape[:3]
    assert dataset.__len__() == 1


def test_collate_keeps_single_coords() -> None:
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
    dataset = AxialDataset(data)
    coords, sol, src, a_vals, ap_vals, b_vals, c_vals = axial_collate_fn(
        [dataset[0], dataset[0]]
    )
    assert coords.shape[0] == 2
    assert sol.shape[0] == 2  # batch size
    assert src.shape[0] == 2
    assert a_vals.shape[0] == 2
