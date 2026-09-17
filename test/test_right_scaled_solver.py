import numpy as np
import pytest
import torch
from scipy.sparse.linalg import LinearOperator, lsmr

from cli.audit_right_scaled_solver import right_scale


def test_scale_is_inverse_square_root_not_inverse():
    np.testing.assert_allclose(right_scale(np.array([4.0, 9.0])), [0.5, 1 / 3])


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_invalid_scale_fails(value):
    with pytest.raises(ValueError):
        right_scale(np.array([1.0, value]))


def test_right_scaled_equation_and_inconsistent_least_squares():
    a = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 0.3]])
    b = np.array([0.5, -1.0, 2.0])
    p = right_scale(np.array([1.0, 100.0]))
    op = LinearOperator(
        a.shape, matvec=lambda z: a @ (p * z), rmatvec=lambda v: p * (a.T @ v)
    )
    result = p * lsmr(op, b, atol=1e-14, btol=1e-14)[0]
    np.testing.assert_allclose(result, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-12)
    np.testing.assert_allclose(a.T @ (a @ result - b), 0, atol=1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for CuPy GPU1 test"
)
def test_gpu_block_products_and_scaled_solvers():
    import importlib

    cp = pytest.importorskip("cupy")
    from cli.audit_right_scaled_solver import GpuBlocks
    from test.test_coupling_solver_pilot import make_system
    from greenonet.complex_axial_response_operator import (
        AxialResponseBlock,
        FrozenAxialResponseOperator,
        FrozenBidirectionalResponseOperator,
    )

    with (
        cp.cuda.Device(1),
        cp.cuda.ExternalStream(torch.cuda.current_stream(1).cuda_stream),
    ):
        raw = make_system([[2.0, 1.0], [0.0, 3.0]], [[1.0, 0.0], [2.0, 1.0]])
        axes = []
        for axis, blocks in zip(("x", "y"), raw.blocks):
            axes.append(
                FrozenAxialResponseOperator(
                    axis,
                    2,
                    tuple(
                        AxialResponseBlock(
                            torch.tensor(idx, device="cuda:1"),
                            torch.tensor(mat, device="cuda:1"),
                        )
                        for idx, mat in blocks
                    ),
                )
            )
        gpu = GpuBlocks(FrozenBidirectionalResponseOperator(*axes))
        v = cp.array([0.2, 0.7])
        np.testing.assert_allclose(
            cp.asnumpy(gpu.forward(v)), raw.forward(cp.asnumpy(v))
        )
        np.testing.assert_allclose(
            cp.asnumpy(gpu.adjoint(v)), raw.adjoint(cp.asnumpy(v))
        )
        p = cp.array([0.5, 0.3])
        op = gpu.operator(p)
        solve = importlib.import_module("cupyx.scipy.sparse.linalg")
        b = cp.array([1.0, 2.0])
        x, status = solve.gmres(op, b, rtol=1e-12, restart=2, maxiter=20)
        assert status == 0
        expected = np.linalg.solve(raw.matrix.toarray(), cp.asnumpy(b))
        np.testing.assert_allclose(cp.asnumpy(p * x), expected, atol=1e-10)
        fit = solve.lsmr(op, b, atol=1e-13, btol=1e-13, maxiter=20)
        np.testing.assert_allclose(cp.asnumpy(p * fit[0]), expected, atol=1e-10)
