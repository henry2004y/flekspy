"""
flekspy Public API.
"""

from pathlib import Path
import errno
from itertools import islice
import importlib

__all__ = [
    "load",
    "read_idl",
    "IDLAccessor",
    "YtFLEKSData",
    "extract_phase",
    "FLEKSTP",
    "AMReXParticleData",
    "xr",
]


def __getattr__(name):
    """
    Dynamically import modules and classes upon first access.
    """
    if name == "YtFLEKSData" or name == "extract_phase":
        module = importlib.import_module("flekspy.yt")
        return getattr(module, name)
    elif name == "FLEKSTP":
        module = importlib.import_module("flekspy.tp")
        return getattr(module, name)
    elif name == "AMReXParticleData":
        module = importlib.import_module("flekspy.amrex")
        return getattr(module, name)
    elif name == "read_idl" or name == "IDLAccessor":
        module = importlib.import_module("flekspy.idl")
        return getattr(module, name)
    elif name == "xr":
        return importlib.import_module("xarray")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def load(
    filename: str,
    iDomain: int = 0,
    iSpecies: int = 0,
    iFile: int = 0,
    readFieldData: bool = False,
    use_yt_loader: bool = False,
):
    """Load FLEKS data.

    Args:
        filename (str): Input file name pattern.
        iDomain (int, optional): Test particle domain index. Defaults to 0.
        iSpecies (int, optional): Test particle species index. Defaults to 0.
        iFile (int, optional): The index of the file to load if the pattern
            matches multiple files. Defaults to 0.
        readFieldData (bool, optional): Whether or not to read field data for test particles. Defaults to False.
        use_yt_loader (bool, optional): If True, forces the use of the yt loader for AMReX data. Defaults to False.

    Returns:
        FLEKS data: xarray.Dataset, YtFLEKSData, or FLEKSTP
    """
    p = Path(filename)
    file_generator = p.parent.rglob(p.name)
    # Advance the generator to the iFile-th position and get the file.
    selected_file_iter = islice(file_generator, iFile, iFile + 1)
    try:
        selected_file = next(selected_file_iter)
    except StopIteration:
        selected_file = None

    if selected_file is None:
        message = f"No files found matching pattern: '{filename}'"
        if iFile > 0:
            message += f" at index {iFile}"
        raise FileNotFoundError(errno.ENOENT, message, filename)
    filename = str(selected_file.resolve())

    filepath = Path(filename)
    basename = filepath.name

    if basename == "test_particles":
        FLEKSTP = getattr(importlib.import_module("flekspy.tp"), "FLEKSTP")
        return FLEKSTP(filename, iDomain=iDomain, iSpecies=iSpecies)
    elif filepath.suffix in [".out", ".outs"]:
        read_idl = getattr(importlib.import_module("flekspy.idl"), "read_idl")
        return read_idl(filename)
    elif basename.endswith("_amrex"):
        if use_yt_loader or "particle" not in basename:
            YtFLEKSData = getattr(importlib.import_module("flekspy.yt"), "YtFLEKSData")
            return YtFLEKSData(filename, readFieldData)
        else:
            AMReXParticleData = getattr(
                importlib.import_module("flekspy.amrex"), "AMReXParticleData"
            )
            return AMReXParticleData(filename)
    else:
        raise Exception("Error: unknown file format!")
