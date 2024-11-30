"""
flekspy Public API.
"""

import glob
import os
import errno
from flekspy.idl.idl_format_data import IDLData
from flekspy.yt.yt_interface import FLEKSData
from flekspy.tp.test_particles import FLEKSTP


def load(filename: str, iDomain=0, iSpecies=0, readFieldData=False):
    """Load FLEKS data.

    Args:
        filename (str): Input file name.
        iDomain (int, optional): Test particle domain index. Defaults to 0.
        iSpecies (int, optional): Test particle species index. Defaults to 0.
        readFieldData (bool, optional): Whether or not to read field data for test particles. Defaults to False.

    Returns:
        FLEKS data: IDLData, FLEKSData, or FLEKSTP
    """
    files = glob.glob(filename)
    if len(files) == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), files)
    filename = files[0]

    basename = os.path.basename(os.path.normpath(filename))

    if basename == "test_particles":
        return FLEKSTP(filename, iDomain=iDomain, iSpecies=iSpecies)
    elif basename.find(".") != -1 and basename.split(".")[-1] in ["out", "outs"]:
        return IDLData(filename)
    elif basename[-6:] == "_amrex":
        return FLEKSData(filename, readFieldData)
    else:
        raise Exception("Error: unknown file format!")
