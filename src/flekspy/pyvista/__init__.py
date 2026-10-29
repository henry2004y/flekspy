"""Lightweight, optional PyVista readers for Tecplot and VTK meshes."""

from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["read_pyvista"]

SUPPORTED_SUFFIXES = frozenset(
    {".dat", ".vtk", ".vtu", ".vts", ".vtr", ".vti", ".vtp", ".vtm"}
)

_ZONE_RE = re.compile(r"\s*ZONE\b", re.IGNORECASE)
_VARIABLES_RE = re.compile(r"\s*VARIABLES\s*=", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")
_COORD_RE = re.compile(r'"([xyzXYZ])\s+[^"\n]*"')


def read_pyvista(filename: str | Path) -> "pv.DataSet | pv.MultiBlock":
    """Read an ASCII Tecplot DAT or VTK file as a native PyVista object.

    Supports .dat, .vtk, .vtu, .vts, .vtr, .vti, .vtp, and .vtm. DAT files
    retain all zones as a MultiBlock, including single-zone files. Bracketed
    units are removed from DAT variable names and coordinate names such as
    ``X AU`` become ``X`` so VTK recognizes them. Cleanup uses a temporary
    copy and never modifies the source. VTK arrays and metadata are unchanged.

    Requires the optional extra: ``pip install 'flekspy[pyvista]'``.
    """
    source = Path(filename)
    if not source.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported PyVista file extension: {suffix}")

    try:
        import pyvista as pv
    except ModuleNotFoundError as error:
        if error.name != "pyvista":
            raise
        raise ImportError(
            "PyVista loading requires pip install 'flekspy[pyvista]'"
        ) from error

    if suffix != ".dat":
        return pv.read(source, force_ext=suffix)

    with TemporaryDirectory(prefix="flekspy-") as directory:
        cleaned = Path(directory) / "mesh.dat"
        with (
            source.open(encoding="utf-8") as original,
            cleaned.open("w", encoding="utf-8") as output,
        ):
            variables = False
            for line in original:
                if _ZONE_RE.match(line):
                    output.write(line)
                    shutil.copyfileobj(original, output)
                    break
                if _VARIABLES_RE.match(line):
                    variables = True
                if variables:
                    line = _BRACKET_RE.sub("", line)
                    line = _COORD_RE.sub(r'"\1"', line)
                output.write(line)
        return pv.read(cleaned)
