"""
flekspy Public API.
"""

from pathlib import Path
import errno
from flekspy.idl import IDLData
from flekspy.yt import FLEKSData, extract_phase
from flekspy.tp import FLEKSTP


def load(filename: str, iDomain=0, iSpecies=0, readFieldData: bool = False):
    """Load FLEKS data.

    Args:
        filename (str): Input file name pattern.
        iDomain (int, optional): Test particle domain index. Defaults to 0.
        iSpecies (int, optional): Test particle species index. Defaults to 0.
        readFieldData (bool, optional): Whether or not to read field data for test particles. Defaults to False.

    Returns:
        FLEKS data: IDLData, FLEKSData, or FLEKSTP
    """
    files = list(Path().glob(filename))
    if len(files) == 0:
        message = f"No files found matching pattern: '{filename}'"
        raise FileNotFoundError(errno.ENOENT, message, filename)
    filename = str(files[0].resolve())

    filepath = Path(filename)
    basename = filepath.name

    if basename == "test_particles":
        return FLEKSTP(filename, iDomain=iDomain, iSpecies=iSpecies)
    elif filepath.suffix in [".out", ".outs"]:
        return IDLData(filename)
    elif basename.endswith("_amrex"):
        return FLEKSData(filename, readFieldData)
    else:
        raise Exception("Error: unknown file format!")
