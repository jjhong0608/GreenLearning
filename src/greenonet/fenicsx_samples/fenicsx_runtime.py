from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


class GmshIOAdapter:
    """Normalize DOLFINx Gmsh IO API differences across supported versions."""

    def __init__(self, module: Any) -> None:
        self.module = module

    def model_to_mesh(
        self,
        model: Any,
        comm: Any,
        rank: int,
        *,
        gdim: int,
    ) -> tuple[Any, Any, Any]:
        return self._normalize_mesh_data(
            self.module.model_to_mesh(model, comm, rank, gdim=gdim)
        )

    def read_from_msh(
        self,
        filename: str,
        comm: Any,
        rank: int,
        *,
        gdim: int,
    ) -> tuple[Any, Any, Any]:
        return self._normalize_mesh_data(
            self.module.read_from_msh(filename, comm, rank=rank, gdim=gdim)
        )

    @staticmethod
    def _normalize_mesh_data(value: Any) -> tuple[Any, Any, Any]:
        if isinstance(value, tuple):
            if len(value) < 3:
                raise ValueError("DOLFINx Gmsh IO tuple return must contain 3 values.")
            return value[0], value[1], value[2]
        if hasattr(value, "mesh"):
            return (
                value.mesh,
                getattr(value, "cell_tags", None),
                getattr(value, "facet_tags", None),
            )
        raise TypeError(
            "DOLFINx Gmsh IO returned an unsupported mesh payload type: "
            f"{type(value)!r}."
        )


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
            gmsh_module = FenicsxImportMixin._load_gmsh_io_module()
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
            gmshio=GmshIOAdapter(gmsh_module),
            ufl=ufl,
            default_scalar_type=getattr(dolfinx, "default_scalar_type", float),
        )

    @staticmethod
    def _load_gmsh_io_module() -> Any:
        try:
            return importlib.import_module("dolfinx.io.gmsh")
        except ModuleNotFoundError:
            return importlib.import_module("dolfinx.io.gmshio")
