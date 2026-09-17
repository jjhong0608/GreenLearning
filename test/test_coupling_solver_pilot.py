import numpy as np
import pytest
import torch
from scipy.sparse.linalg import gmres, lsmr

from cli.audit_coupling_solver_pilot import BlockSystem, direct_solutions
from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)


def make_system(x, y):
    return BlockSystem(
        FrozenBidirectionalResponseOperator(
            *[
                FrozenAxialResponseOperator(
                    axis,
                    len(x),
                    (
                        AxialResponseBlock(
                            torch.arange(len(x)), torch.tensor(a, dtype=torch.float64)
                        ),
                    ),
                )
                for axis, a in (("x", x), ("y", y))
            ]
        )
    )


def test_block_assembly_and_adjoint():
    system = make_system([[2.0, 1.0], [0.0, 3.0]], [[1.0, 0.0], [2.0, 1.0]])
    v = np.array([0.3, -0.7])
    np.testing.assert_allclose(system.operator @ v, system.matrix @ v)
    np.testing.assert_allclose(system.operator.rmatvec(v), system.matrix.T @ v)


def test_equation_least_squares_and_balance():
    system = make_system([[2.0, 1.0], [0.0, 3.0]], [[1.0, 0.0], [2.0, 1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]])
    values, info = direct_solutions(system.matrix.toarray(), rhs, "cpu")
    assert info["qr_full_rank_screen"]
    np.testing.assert_allclose(values["lu"], values["qr"], atol=1e-13)
    for i in range(2):
        g, status = gmres(system.operator, rhs[:, i], rtol=1e-12)
        least_squares = lsmr(system.operator, rhs[:, i], atol=1e-13, btol=1e-13)
        assert status == 0
        np.testing.assert_allclose(g, values["lu"][:, i], atol=1e-12)
        np.testing.assert_allclose(least_squares[0], g, atol=1e-12)
    f = np.array([2.0, -3.0])
    b = 0.5 * (system.axis(f, 1) - system.axis(f, 0))
    d = np.linalg.solve(system.matrix.toarray(), b)
    np.testing.assert_allclose((f / 2 + d) + (f / 2 - d), f)
    np.testing.assert_allclose(system.axis(f / 2 + d, 0), system.axis(f / 2 - d, 1))


def test_rank_deficient_direct_path_is_not_silently_accepted():
    with pytest.raises(ValueError, match="rank"):
        direct_solutions(np.diag([1.0, 0.0]), np.ones((2, 1)), "cpu")


def test_inconsistent_lsmr_stationarity_not_zero_residual():
    system = make_system([[1.0, 0.0], [0.0, 0.0]], np.zeros((2, 2)))
    rhs = np.ones(2)
    x, status, *_ = lsmr(system.operator, rhs, atol=1e-13, btol=1e-13)
    assert status == 2
    assert np.linalg.norm(system.operator @ x - rhs) > 0.9
    assert np.linalg.norm(system.operator.rmatvec(system.operator @ x - rhs)) < 1e-12
