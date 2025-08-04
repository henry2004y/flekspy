from typing import List, Union
import warnings

from .accessor import read_tp_data, Indices


def FLEKSTP(
    dirs: Union[str, List[str]],
    iDomain: int = 0,
    iSpecies: int = 0,
    iListStart: int = 0,
    iListEnd: int = -1,
    readAllFiles: bool = False,  # Kept for backward compatibility, but not used
):
    """
    Loads test particle data into an xarray.Dataset.

    .. deprecated:: 0.4.0
       The class-based `FLEKSTP` is deprecated. This function provides a
       backward-compatible way to load data, but users are encouraged to
       switch to the new `read_tp_data` function. The returned object is now
       an xarray.Dataset with a `.tp` accessor for test particle specific
       methods.

    Examples:
        >>> ds = FLEKSTP("path/to/data")
        >>> trajectory = ds.sel(particle=(cpu, pid))
        >>> ds.tp.plot_trajectory((cpu, pid))
    """
    warnings.warn(
        "The FLEKSTP class is deprecated and will be removed in a future version. "
        "Use the `read_tp_data` function instead, which returns an xarray.Dataset. "
        "The returned Dataset has a `.tp` accessor for specialized methods.",
        FutureWarning,
    )
    return read_tp_data(
        dirs=dirs,
        iDomain=iDomain,
        iSpecies=iSpecies,
        iListStart=iListStart,
        iListEnd=iListEnd,
    )
