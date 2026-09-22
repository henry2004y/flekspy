"""
IDLSeries: Sequence container for temporal series of IDL .out / .outs files.
Provides lazy-loaded LRU caching, header pre-scanning, temporal slicing,
nearest-time frame lookup, and optional dataset concatenation.
"""

from __future__ import annotations

from collections import OrderedDict
import glob
from pathlib import Path
import struct
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from flekspy.idl.idl import _get_file_head, read_idl
from flekspy.util.logger import get_logger

logger = get_logger(name=__name__)


def read_idl_header(filename: Union[str, Path]) -> Dict[str, Any]:
    """Read metadata header of an IDL (.out/.outs) file without loading grid arrays.

    Parameters
    ----------
    filename : str or Path
        Path to IDL file.

    Returns
    -------
    dict
        Extracted header attributes (time, iter, grid, ndim, variables, parameters, etc.).
    """
    fpath = str(Path(filename).expanduser().resolve())
    attrs: Dict[str, Any] = {
        "filename": fpath,
        "isOuts": fpath.endswith(".outs"),
        "npict": 1,
        "nInstance": 1,
    }

    with open(fpath, "rb") as f:
        rec_len_raw = f.read(4)
        if len(rec_len_raw) < 4:
            raise ValueError(f"File {fpath} is empty or corrupted.")
        rec_len = struct.unpack("<l", rec_len_raw)[0]
        attrs["fileformat"] = "binary" if rec_len in (79, 500) else "ascii"

    if attrs["fileformat"] == "ascii":
        with open(fpath, "r") as f:
            new_attrs, _ = _get_file_head(f, attrs)
    else:
        with open(fpath, "rb") as f:
            new_attrs, _ = _get_file_head(f, attrs)

    attrs.update(new_attrs)

    nsize = attrs["ndim"] + attrs["nvar"]
    variables = attrs.get("variables", [])
    varnames = tuple(variables)[:nsize]
    param_names = tuple(variables)[nsize:]

    if "parameters" in attrs and len(param_names) == len(attrs["parameters"]):
        attrs["parameters"] = dict(zip(param_names, attrs["parameters"]))
    else:
        attrs["parameters"] = {}

    attrs["variables"] = list(varnames)
    attrs.pop("pformat", None)
    return attrs


class IDLSeries(Sequence[xr.Dataset]):
    """Sequence container for a series of IDL (.out/.outs) output files.

    Provides fast metadata scanning, lazy loading with LRU cache, temporal slicing,
    and nearest-frame lookup by time or iteration.

    Parameters
    ----------
    files : str or Sequence of (str or Path)
        File path pattern (e.g. 'run/PC/*.out') or list of file paths.
    eager : bool, default False
        If True, loads all datasets into memory at initialization.
        If False, loads datasets on-demand when indexed.
    max_cache : int, default 32
        Maximum number of datasets to hold in memory when lazy loading.
    """

    def __init__(
        self,
        files: Union[str, Path, Sequence[Union[str, Path]]],
        eager: bool = False,
        max_cache: int = 32,
        _headers: Optional[List[Dict[str, Any]]] = None,
    ):
        if isinstance(files, (str, Path)):
            pattern = str(Path(files).expanduser())
            file_list = sorted(glob.glob(pattern))
            if not file_list:
                if Path(pattern).is_file():
                    file_list = [str(Path(pattern).resolve())]
                else:
                    raise FileNotFoundError(f"No IDL files matching '{files}' found.")
        else:
            file_list = [str(Path(f).expanduser().resolve()) for f in files]
            if not file_list:
                raise ValueError("Empty file list provided to IDLSeries.")

        self.eager = eager
        self.max_cache = max(1, max_cache)
        self._cache: OrderedDict[int, xr.Dataset] = OrderedDict()
        self._datasets: Optional[List[xr.Dataset]] = None

        if _headers is not None:
            self.headers = _headers
            self.filenames = file_list
        else:
            headers = []
            for fpath in file_list:
                try:
                    h = read_idl_header(fpath)
                    headers.append(h)
                except Exception as err:
                    logger.warning("Failed to parse header for %s: %s", fpath, err)

            if not headers:
                raise ValueError(f"Could not read valid headers from files matching '{files}'.")

            combined = sorted(headers, key=lambda h: (h.get("time", 0.0), h.get("iter", 0)))
            self.headers = combined
            self.filenames = [h["filename"] for h in combined]

        self.times = np.array([h.get("time", 0.0) for h in self.headers], dtype=float)
        self.iters = np.array([h.get("iter", 0) for h in self.headers], dtype=int)

        if self.eager:
            self._datasets = [read_idl(f) for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: Union[int, slice]) -> Union[xr.Dataset, IDLSeries]:
        if isinstance(index, slice):
            sliced_files = self.filenames[index]
            sliced_headers = self.headers[index]
            return self.__class__(
                sliced_files,
                eager=self.eager,
                max_cache=self.max_cache,
                _headers=sliced_headers,
            )

        n = len(self)
        if index < -n or index >= n:
            raise IndexError(f"Index {index} out of range for series of length {n}.")
        if index < 0:
            index += n

        if self.eager and self._datasets is not None:
            return self._datasets[index]

        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        ds = read_idl(self.filenames[index])
        if len(self._cache) >= self.max_cache:
            self._cache.popitem(last=False)
        self._cache[index] = ds
        return ds

    def __iter__(self) -> Iterator[xr.Dataset]:
        for i in range(len(self)):
            yield self[i]

    def get_frame(
        self,
        time: Optional[float] = None,
        iter: Optional[int] = None,
    ) -> Tuple[int, xr.Dataset]:
        """Find the frame nearest to the requested time or iteration."""
        if time is not None:
            idx = int(np.argmin(np.abs(self.times - time)))
        elif iter is not None:
            idx = int(np.argmin(np.abs(self.iters - iter)))
        else:
            raise ValueError("Must specify either 'time' or 'iter'.")

        return idx, self[idx]

    def clear_cache(self) -> None:
        """Clear memory cache of loaded datasets."""
        self._cache.clear()

    def to_dataset(self, vars: Optional[Sequence[str]] = None) -> xr.Dataset:
        """Concatenate all frames into a single xarray.Dataset with a 'time' dimension."""
        datasets = []
        for i in range(len(self)):
            ds = self[i]
            if vars is not None:
                ds = ds[list(vars)]
            datasets.append(ds)

        combined = xr.concat(datasets, dim="time")
        combined = combined.assign_coords(time=self.times)
        return combined

    def __repr__(self) -> str:
        grid_str = str(self.headers[0].get("grid", [])) if self.headers else "unknown"
        t_start = self.times[0] if len(self.times) > 0 else 0.0
        t_end = self.times[-1] if len(self.times) > 0 else 0.0
        vars_preview = self.headers[0].get("variables", [])[:6] if self.headers else []
        mode = "eager" if self.eager else f"lazy(lru={self.max_cache})"
        return (
            f"<{self.__class__.__name__}: {len(self)} frames, "
            f"t=[{t_start:.2f}, {t_end:.2f}], grid={grid_str}, "
            f"mode={mode}, vars={vars_preview}>"
        )
