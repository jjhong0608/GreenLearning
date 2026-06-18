from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FenicsxRuntime:
    """Lazy-loaded optional FEniCSx/Gmsh module bundle."""

    gmsh: Any
    mpi: Any
    fem: Any
    fem_petsc: Any
    dolfinx_mesh: Any
    dolfinx_geometry: Any
    gmshio: Any
    ufl: Any
    default_scalar_type: Any


class FenicsxImportMixin:
    """Import optional FEniCSx dependencies only when the generator needs them."""

    @staticmethod
    def load_runtime() -> FenicsxRuntime:
        try:
            gmsh = importlib.import_module("gmsh")
            mpi_module = importlib.import_module("mpi4py.MPI")
            dolfinx = importlib.import_module("dolfinx")
            fem = importlib.import_module("dolfinx.fem")
            fem_petsc = importlib.import_module("dolfinx.fem.petsc")
            dolfinx_mesh = importlib.import_module("dolfinx.mesh")
            dolfinx_geometry = importlib.import_module("dolfinx.geometry")
            gmshio = importlib.import_module("dolfinx.io.gmshio")
            ufl = importlib.import_module("ufl")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FEniCSx sample generation requires the optional green_fenicsx "
                "environment. Create it with `conda env create -f "
                "environment-fenicsx.yml` and run this CLI with "
                "`conda run -n green_fenicsx ...`."
            ) from exc
        return FenicsxRuntime(
            gmsh=gmsh,
            mpi=mpi_module,
            fem=fem,
            fem_petsc=fem_petsc,
            dolfinx_mesh=dolfinx_mesh,
            dolfinx_geometry=dolfinx_geometry,
            gmshio=gmshio,
            ufl=ufl,
            default_scalar_type=getattr(dolfinx, "default_scalar_type", float),
        )
