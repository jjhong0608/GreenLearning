from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from greenonet.coefficients import CoefficientFunctions
from greenonet.fenicsx_samples.fenicsx_runtime import FenicsxRuntime
from greenonet.fenicsx_samples.geometry import RawComplexGeometryGrid


@dataclass(frozen=True)
class FenicsxSolveResult:
    """Full-grid solution and direction-split target arrays."""

    sol: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    balance_relative_residual: float


class TorchCoefficientMixin:
    """Evaluate project coefficient callables on NumPy coordinate arrays."""

    @staticmethod
    def coefficient_callback(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[np.ndarray], np.ndarray]:
        def callback(points: np.ndarray) -> np.ndarray:
            x_tensor = torch.as_tensor(points[0], dtype=torch.float64)
            y_tensor = torch.as_tensor(points[1], dtype=torch.float64)
            values = function(x_tensor, y_tensor)
            return values.detach().cpu().numpy().astype(np.float64, copy=False)

        return callback


class PointEvaluationMixin:
    """Evaluate DOLFINx functions at valid geometry points."""

    @staticmethod
    def evaluate_at_points(
        runtime: FenicsxRuntime,
        function: Any,
        points_xy: np.ndarray,
    ) -> np.ndarray:
        domain = getattr(function, "function_space").mesh
        points = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
        points[:, :2] = points_xy
        tree = runtime.dolfinx_geometry.bb_tree(domain, domain.topology.dim)
        candidates = runtime.dolfinx_geometry.compute_collisions_points(tree, points)
        colliding = runtime.dolfinx_geometry.compute_colliding_cells(
            domain,
            candidates,
            points,
        )
        cells: list[int] = []
        for index in range(points.shape[0]):
            links = colliding.links(index)
            if len(links) == 0:
                raise ValueError(
                    "A valid geometry point could not be located in the FEniCSx mesh: "
                    f"{points_xy[index].tolist()}."
                )
            cells.append(int(links[0]))
        raw_values = function.eval(points, np.asarray(cells, dtype=np.int32))
        return np.asarray(raw_values, dtype=np.float64).reshape(points.shape[0], -1)[
            :, 0
        ]


class FenicsxPdeSolver(TorchCoefficientMixin, PointEvaluationMixin):
    """Solve one complex-domain PDE sample and evaluate full-grid valid values."""

    def __init__(
        self,
        runtime: FenicsxRuntime,
        domain: Any,
        geometry: RawComplexGeometryGrid,
        coeffs: CoefficientFunctions,
        *,
        solution_degree: int,
        target_degree: int,
    ) -> None:
        self.runtime = runtime
        self.domain = domain
        self.geometry = geometry
        self.coeffs = coeffs
        self.solution_degree = solution_degree
        self.target_degree = target_degree
        self._solution_space = self._function_space(solution_degree)
        self._target_space = self._function_space(target_degree)
        self._coefficient_space = self._function_space(max(1, target_degree))

    def solve(self, rhs: np.ndarray) -> FenicsxSolveResult:
        if rhs.shape != self.geometry.full_grid_shape:
            raise ValueError(
                f"rhs must have shape {self.geometry.full_grid_shape}, got {rhs.shape}."
            )
        a_fun = self._interpolate_coefficient(self.coeffs.a_fun)
        bx_fun = self._interpolate_coefficient(self.coeffs.bx_fun)
        by_fun = self._interpolate_coefficient(self.coeffs.by_fun)
        c_fun = self._interpolate_coefficient(self.coeffs.c_fun)
        rhs_fun = self._interpolate_rhs(rhs)
        solution = self._solve_variational_problem(
            rhs_fun=rhs_fun,
            a_fun=a_fun,
            bx_fun=bx_fun,
            by_fun=by_fun,
            c_fun=c_fun,
        )
        phi = self._project(
            -((a_fun * solution.dx(0)).dx(0))
            + bx_fun * solution.dx(0)
            + 0.5 * c_fun * solution
        )
        psi = self._project(
            -((a_fun * solution.dx(1)).dx(1))
            + by_fun * solution.dx(1)
            + 0.5 * c_fun * solution
        )

        sol_valid = self.evaluate_at_points(
            self.runtime,
            solution,
            self.geometry.coords_valid,
        )
        phi_valid = self.evaluate_at_points(
            self.runtime, phi, self.geometry.coords_valid
        )
        psi_valid = self.evaluate_at_points(
            self.runtime, psi, self.geometry.coords_valid
        )
        sol_full = self.geometry.valid_values_to_full_grid(sol_valid)
        phi_full = self.geometry.valid_values_to_full_grid(phi_valid)
        psi_full = self.geometry.valid_values_to_full_grid(psi_valid)
        rhs_valid = rhs[
            self.geometry.valid_grid_y_index,
            self.geometry.valid_grid_x_index,
        ]
        residual = (phi_valid + psi_valid) - rhs_valid
        denominator = max(float(np.linalg.norm(rhs_valid)), 1.0e-12)
        return FenicsxSolveResult(
            sol=sol_full,
            phi=phi_full,
            psi=psi_full,
            balance_relative_residual=float(np.linalg.norm(residual) / denominator),
        )

    def _function_space(self, degree: int) -> Any:
        return self.runtime.fem.functionspace(
            self.domain,
            ("Lagrange", degree),
        )

    def _new_function(self, space: Any, name: str) -> Any:
        function = self.runtime.fem.Function(space)
        function.name = name
        return function

    def _interpolate_coefficient(
        self,
        coefficient: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Any:
        function = self._new_function(self._coefficient_space, "coefficient")
        function.interpolate(self.coefficient_callback(coefficient))
        return function

    def _interpolate_rhs(self, rhs: np.ndarray) -> Any:
        interpolator = RegularGridInterpolator(
            (self.geometry.grid_y, self.geometry.grid_x),
            rhs,
            bounds_error=False,
            fill_value=0.0,
        )

        def callback(points: np.ndarray) -> np.ndarray:
            query_points = np.column_stack((points[1], points[0]))
            values = interpolator(query_points).astype(np.float64, copy=False)
            return cast(np.ndarray, values)

        function = self._new_function(self._coefficient_space, "rhs")
        function.interpolate(callback)
        return function

    def _solve_variational_problem(
        self,
        *,
        rhs_fun: Any,
        a_fun: Any,
        bx_fun: Any,
        by_fun: Any,
        c_fun: Any,
    ) -> Any:
        ufl = self.runtime.ufl
        trial = ufl.TrialFunction(self._solution_space)
        test = ufl.TestFunction(self._solution_space)
        bilinear = (
            a_fun * ufl.dot(ufl.grad(trial), ufl.grad(test))
            + (bx_fun * trial.dx(0) + by_fun * trial.dx(1)) * test
            + c_fun * trial * test
        ) * ufl.dx
        linear = rhs_fun * test * ufl.dx
        problem = self.runtime.fem_petsc.LinearProblem(
            bilinear,
            linear,
            bcs=[self._homogeneous_dirichlet_bc()],
            petsc_options_prefix="greenonet_fenicsx_pde_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        return problem.solve()

    def _homogeneous_dirichlet_bc(self) -> Any:
        topology = self.domain.topology
        topology.create_connectivity(topology.dim - 1, topology.dim)
        facets = self.runtime.dolfinx_mesh.exterior_facet_indices(topology)
        dofs = self.runtime.fem.locate_dofs_topological(
            self._solution_space,
            topology.dim - 1,
            facets,
        )
        value = self.runtime.default_scalar_type(0)
        return self.runtime.fem.dirichletbc(value, dofs, self._solution_space)

    def _project(self, expression: Any) -> Any:
        ufl = self.runtime.ufl
        trial = ufl.TrialFunction(self._target_space)
        test = ufl.TestFunction(self._target_space)
        problem = self.runtime.fem_petsc.LinearProblem(
            trial * test * ufl.dx,
            expression * test * ufl.dx,
            petsc_options_prefix="greenonet_fenicsx_project_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        return problem.solve()
