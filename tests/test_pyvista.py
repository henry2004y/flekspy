"""Tests and benchmark for PyVista reader module."""

import time
import pytest
import pyvista as pv

from flekspy.pyvista import (
    read_pyvista,
    _ZONE_RE,
    _VARIABLES_RE,
    _BRACKET_RE,
    _COORD_RE,
)


def test_read_pyvista_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_pyvista("non_existent_file.dat")


def test_read_pyvista_unsupported_suffix(tmp_path):
    invalid_file = tmp_path / "mesh.txt"
    invalid_file.write_text("dummy content")
    with pytest.raises(ValueError, match="Unsupported PyVista file extension"):
        read_pyvista(invalid_file)


def test_read_pyvista_dat_cleaning(tmp_path):
    dat_content = (
        'TITLE = "Sample Mesh"\n'
        'VARIABLES = "X [AU]", "Y [AU]", "Z [AU]", "Rho [amu/cc]", "ux [km/s]"\n'
        'ZONE T="Zone 1", I=2, J=2, F=POINT\n'
        "0.0 0.0 0.0 1.0 10.0\n"
        "1.0 0.0 0.0 1.0 10.0\n"
        "0.0 1.0 0.0 1.0 10.0\n"
        "1.0 1.0 0.0 1.0 10.0\n"
    )
    dat_file = tmp_path / "test_mesh.dat"
    dat_file.write_text(dat_content)

    mesh = read_pyvista(dat_file)
    assert mesh is not None
    if isinstance(mesh, pv.MultiBlock):
        block = mesh[0]
    else:
        block = mesh

    array_names = block.array_names
    assert "Rho" in array_names
    assert "ux" in array_names


def test_regex_compilation_performance_benchmark():
    import re

    lines = [
        'VARIABLES = "X [AU]", "Y [AU]", "Z [AU]", "Rho [amu/cc]", "ux [km/s]", "uy [km/s]", "uz [km/s]", "Bx [nT]", "By [nT]", "Bz [nT]"\n'
    ] * 50000

    start_legacy = time.perf_counter()
    for line in lines:
        if re.match(r"\s*ZONE\b", line, re.IGNORECASE):
            break
        if re.match(r"\s*VARIABLES\s*=", line, re.IGNORECASE):
            pass
        line = re.sub(r"\s*\[[^\]]*\]", "", line)
        line = re.sub(r'"([xyzXYZ])\s+[^"\n]*"', r'"\1"', line)
    duration_legacy = time.perf_counter() - start_legacy

    start_optimized = time.perf_counter()
    for line in lines:
        if _ZONE_RE.match(line):
            break
        if _VARIABLES_RE.match(line):
            pass
        line = _BRACKET_RE.sub("", line)
        line = _COORD_RE.sub(r'"\1"', line)
    duration_optimized = time.perf_counter() - start_optimized

    assert duration_optimized <= duration_legacy
    speedup = duration_legacy / duration_optimized if duration_optimized > 0 else float("inf")
    print(f"Legacy: {duration_legacy:.4f}s, Optimized: {duration_optimized:.4f}s, Speedup: {speedup:.2f}x")
