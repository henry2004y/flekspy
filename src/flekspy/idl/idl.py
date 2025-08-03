import numpy as np
import matplotlib.pyplot as plt
import struct
import yt
import xarray as xr
from enum import IntEnum

from flekspy.util import get_unit
from flekspy.util import (
    DataContainer,
    DataContainer1D,
    DataContainer2D,
    DataContainer3D,
)


class IDLDataX:
    r"""
    A class used to handle the `*.out` format SWMF data.
    Example:
    >>> ds = IDLDataX("3d.out")
    >>> dc2d = ds.get_slice("y", 1)
    """

    def __init__(self, filename="none"):
        self.filename = filename
        self.isOuts = self.filename.endswith("outs")
        self.nInstance = None if self.isOuts else 1
        self.npict = 1
        self.fileformat = None
        self.variables = None
        self.unit = None
        self.iter = None
        self.time = None
        self.ndim = None
        self.gencoord = None
        self.grid = None
        self.end_char = None
        self.pformat = None

        self._raw_data_array = self.read_data()

        # Reshape data if ndim < 3
        shape = list(self._raw_data_array.shape) + [1] * (
            4 - self._raw_data_array.ndim
        )
        self._raw_data_array = np.reshape(self._raw_data_array, shape)

        coords = {}
        dims = []
        for i in range(self.ndim):
            dim_name = self.dims[i]
            dims.append(dim_name)
            dim_idx = self._varnames.index(dim_name)
            slicer = [0] * 3
            slicer[i] = slice(None, self.grid[i])
            coords[dim_name] = np.squeeze(
                self._raw_data_array[dim_idx, slicer[0], slicer[1], slicer[2]]
            )

        data_vars = {}
        for i, var_name in enumerate(self._varnames):
            if var_name not in self.dims:
                slicer = [i]
                for d in range(3):
                    if d < self.ndim:
                        slicer.append(slice(self.grid[d]))
                    else:
                        slicer.append(slice(1))
                data_slice = self._raw_data_array[tuple(slicer)]
                data_vars[var_name] = (dims, np.squeeze(data_slice))

        del self._raw_data_array
        del self._varnames

        self.data = xr.Dataset(data_vars, coords=coords)

        self.data.attrs["time"] = self.time
        self.data.attrs["iter"] = self.iter
        self.data.attrs["unit"] = self.unit
        self.data.attrs["gencoord"] = self.gencoord

    def __post_process_param__(self):

        planet_radius = 1.0

        # Not always correct.
        for var, val in zip(self.param_name, self.para):
            if var == "xSI":
                planet_radius = float(100 * val)

        self.registry = yt.units.unit_registry.UnitRegistry()
        self.registry.add("Planet_Radius", planet_radius, yt.units.dimensions.length)

    def __repr__(self):
        string = (
            f"filename    : {self.filename}\n"
            f"variables   : {self.variables}\n"
            f"unit        : {self.unit}\n"
            f"nInstance   : {self.nInstance}\n"
            f"npict       : {self.npict}\n"
            f"time        : {self.time}\n"
            f"nIter       : {self.iter}\n"
            f"ndim        : {self.ndim}\n"
            f"gencoord    : {self.gencoord}\n"
            f"grid        : {self.grid}\n"
        )
        return string

    def read_data(self):
        if self.fileformat is None:
            with open(self.filename, "rb") as f:
                EndChar = "<"  # Endian marker (default: little.)
                RecLenRaw = f.read(4)
                RecLen = (struct.unpack(EndChar + "l", RecLenRaw))[0]
                if RecLen != 79 and RecLen != 500:
                    self.fileformat = "ascii"
                else:
                    self.fileformat = "binary"

        if self.fileformat == "ascii":
            array = self.read_ascii()
        elif self.fileformat == "binary":
            try:
                array = self.read_binary()
            except:
                print(
                    "It seems the lengths of instances are different. Try slow reading..."
                )
                array = self.read_binary_slow()
        else:
            raise ValueError(f"Unknown format = {self.fileformat}")

        nsize = self.ndim + self.nvar
        self._varnames = tuple(self.variables)[0:nsize]
        self.param_name = self.variables[nsize:]
        self.__post_process_param__()

        return array

    def read_ascii(self):
        if self.nInstance is None:
            # Count the number of instances.
            with open(self.filename, "r") as f:
                for i, l in enumerate(f):
                    pass
                nLineFile = i + 1

            with open(self.filename, "r") as f:
                self.nInstanceLength, _ = self.read_ascii_instance(f)

            self.nInstance = round(nLineFile / self.nInstanceLength)

        nLineSkip = (self.npict) * self.nInstanceLength if self.isOuts else 0
        with open(self.filename, "r") as f:
            if nLineSkip > 0:
                for i, line in enumerate(f):
                    if i == nLineSkip - 1:
                        break
            _, array = self.read_ascii_instance(f)
        return array

    def read_ascii_instance(self, infile):
        self.get_file_head(infile)
        nrow = self.ndim + self.nvar
        ncol = self.npoints
        array = np.zeros((nrow, ncol))

        for i, line in enumerate(infile.readlines()):
            parts = line.split()

            if i >= self.npoints:
                break

            for j, p in enumerate(parts):
                array[j][i] = float(p)

        shapeNew = np.append([nrow], self.grid)
        array = np.reshape(array, shapeNew, order="F")
        nline = 5 + self.npoints if self.nparam > 0 else 4 + self.npoints

        return nline, array

    def read_binary(self):
        if self.nInstance is None:
            with open(self.filename, "rb") as f:
                _, n_bytes = self.read_binary_instance(f)
                self.nInstanceLength = n_bytes
                f.seek(0, 2)
                endPos = f.tell()
            self.nInstance = round(endPos / self.nInstanceLength)

        with open(self.filename, "rb") as f:
            if self.isOuts:
                f.seek((self.npict) * self.nInstanceLength, 0)
            return self.read_binary_instance(f)[0]

    def read_binary_slow(self):
        with open(self.filename, "rb") as f:
            if self.isOuts:
                # Skip previous instances
                for i in range(self.npict):
                    self.read_binary_instance(f)
            return self.read_binary_instance(f)[0]

    def get_file_head(self, infile):
        if self.fileformat == "binary":
            # On the first try, we may fail because of wrong-endianess.
            # If that is the case, swap that endian and try again.
            self.end_char = "<"  # Endian marker (default: little)
            self.endian = "little"
            record_len_raw = infile.read(4)

            record_len = (struct.unpack(self.end_char + "l", record_len_raw))[0]
            if (record_len > 10000) or (record_len < 0):
                self.end_char = ">"
                self.endian = "big"
                record_len = (struct.unpack(self.end_char + "l", record_len_raw))[0]

            headline = (
                (
                    struct.unpack(
                        "{0}{1}s".format(self.end_char, record_len),
                        infile.read(record_len),
                    )
                )[0]
                .strip()
                .decode()
            )
            self.unit = headline.split()[0]

            (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
            self.pformat = "f"
            # Parse rest of header; detect double-precision file
            if record_len > 20:
                self.pformat = "d"
            (self.iter, self.time, self.ndim, self.nparam, self.nvar) = struct.unpack(
                "{0}l{1}3l".format(self.end_char, self.pformat), infile.read(record_len)
            )
            self.gencoord = self.ndim < 0
            self.ndim = abs(self.ndim)
            # Get gridsize
            (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))

            self.grid = np.array(
                struct.unpack(
                    "{0}{1}l".format(self.end_char, self.ndim),
                    infile.read(record_len),
                )
            )
            self.npoints = abs(self.grid.prod())

            # Read parameters stored in file
            self.read_parameters(infile)

            # Read variable names
            self.read_variable_names(infile)
        else:
            # Read the top header line
            headline = infile.readline().strip()
            self.unit = headline.split()[0]

            # Read & convert iters, time, etc. from next line
            parts = infile.readline().split()
            self.iter = int(parts[0])
            self.time = float(parts[1])
            self.ndim = int(parts[2])
            self.gencoord = self.ndim < 0
            self.ndim = abs(self.ndim)
            self.nparam = int(parts[3])
            self.nvar = int(parts[4])

            # Read & convert grid dimensions
            grid = [int(x) for x in infile.readline().split()]
            self.grid = np.array(grid)
            self.npoints = abs(self.grid.prod())

            # Read parameters stored in file
            self.para = np.zeros(self.nparam)
            if self.nparam > 0:
                self.para[:] = infile.readline().split()

            # Read variable names
            names = infile.readline().split()

            # Save grid names (e.g. "x" or "r") and save associated params
            self.dims = names[0 : self.ndim]
            self.variables = np.array(names)

            # Create string representation of time
            self.strtime = "%4.4ih%2.2im%06.3fs" % (
                np.floor(self.time / 3600.0),
                np.floor(self.time % 3600.0 / 60.0),
                self.time % 60.0,
            )

    def read_binary_instance(self, infile):
        n_bytes_start = infile.tell()
        self.get_file_head(infile)
        nrow = self.ndim + self.nvar

        if self.pformat == "f":
            dtype = np.float32
        else:
            dtype = np.float64

        array = np.empty((nrow, self.npoints), dtype=dtype)
        dtype_str = f"{self.end_char}{self.pformat}"

        # Get the grid points
        (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
        buffer = infile.read(record_len)
        grid_data = np.frombuffer(
            buffer, dtype=dtype_str, count=self.npoints * self.ndim
        )
        array[0 : self.ndim, :] = grid_data.reshape((self.ndim, self.npoints))

        # Get the actual data and sort
        for i in range(self.ndim, self.nvar + self.ndim):
            (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
            buffer = infile.read(record_len)
            array[i, :] = np.frombuffer(buffer, dtype=dtype_str, count=self.npoints)
        # Consume the last record length
        infile.read(4)

        shape_new = np.append([nrow], self.grid)
        array = np.reshape(array, shape_new, order="F")
        n_bytes_end = infile.tell()

        return array, n_bytes_end - n_bytes_start

    def read_parameters(self, infile):
        """Reads parameters from the binary file."""
        self.para = np.zeros(self.nparam)
        if self.nparam > 0:
            (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
            self.para[:] = struct.unpack(
                "{0}{1}{2}".format(self.end_char, self.nparam, self.pformat),
                infile.read(record_len),
            )

    def read_variable_names(self, infile):
        """Reads variable names from the binary file."""
        (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
        names = (
            struct.unpack(
                "{0}{1}s".format(self.end_char, record_len), infile.read(record_len)
            )
        )[0]
        if str is not bytes:
            names = names.decode()

        names.strip()
        names = names.split()

        # Save grid names (e.g. "x" or "r") and save associated params
        self.dims = names[0 : self.ndim]
        self.variables = np.array(names)

        self.strtime = "{0:04d}h{1:02d}m{2:06.3f}s".format(
            int(self.time // 3600), int(self.time % 3600 // 60), self.time % 60
        )

    def get_domain(self) -> xr.Dataset:
        """Return data as an xarray.Dataset."""
        return self.data

    def get_slice(self, norm, cut_loc) -> xr.Dataset:
        """Get a 2D slice from the 3D IDL data.
        Args:
            norm: str
                The normal direction of the slice from "x", "y" or "z"
            cur_loc: float
                The position of slicing.
        Return: xarray.Dataset
        """
        return self.data.sel({norm: cut_loc}, method="nearest")

    def plot(self, *dvname, **kwargs):
        """Plot 1D IDL outputs.
        Args:
            *dvname (str): variable names
            **kwargs: keyword argument to be passed to `plot`.
        """
        if self.ndim != 1:
            raise ValueError("plot() is for 1D data only.")

        nvar = len(dvname)
        if nvar == 0:
            return

        f, axes = plt.subplots(nvar, 1, constrained_layout=True, sharex=True)
        if nvar == 1:
            axes = [axes]

        for i, var in enumerate(dvname):
            self.data[var].plot(ax=axes[i], **kwargs)

        return axes

    def pcolormesh(self, *dvname, scale: bool = True, **kwargs):
        """Plot 2D pcolormeshes of variables.
        Args:
            *dvname (str): variable names
            scale (bool): whether to scale the plots according to the axis range.
                Default True.
            **kwargs: keyword arguments to be passed to `pcolormesh`.
        """
        if self.ndim != 2:
            raise ValueError("pcolormesh() is for 2D data only.")

        nvar = len(dvname)
        if nvar == 0:
            return

        f, axes = plt.subplots(
            nvar,
            1,
            constrained_layout=True,
            sharex=True,
            sharey=True,
            figsize=kwargs.pop("figsize", (6, 4 * nvar)),
        )
        if nvar == 1:
            axes = [axes]

        for i, var in enumerate(dvname):
            if "cmap" not in kwargs:
                kwargs["cmap"] = "turbo"

            self.data[var].plot.pcolormesh(ax=axes[i], **kwargs)
            axes[i].set_title(var)

        if scale:
            x_coords = self.data.coords[self.dims[0]]
            y_coords = self.data.coords[self.dims[1]]
            aspect_ratio = (y_coords.max() - y_coords.min()) / (
                x_coords.max() - x_coords.min()
            )
            for ax in axes:
                ax.set_aspect(float(aspect_ratio.values))

        return axes

    def get_data(self, loc: np.ndarray) -> np.ndarray:
        """Extract data at a given point using bilinear interpolation.
        Args:
            loc (np.ndarray): 2D/3D point location.
        Returns:
            np.ndarray: 1D array of saved variables at the survey point.
        """
        if not 1 <= self.ndim <= 3:
            raise ValueError(f"get_data not supported for ndim={self.ndim}")

        point = {self.dims[i]: loc[i] for i in range(self.ndim)}

        interp_data = self.data.interp(point)

        return np.array([interp_data[var].values for var in self.data.data_vars])

    def extract_data(self, sat: np.ndarray) -> np.ndarray:
        """Extract data at a series of locations.
        Args:
            sat (np.ndarray): 2D/3D point locations, shape (n_points, n_dims).
        Returns:
            np.ndarray: 2D array of variables at each point.
        """
        if sat.ndim != 2 or sat.shape[1] < self.ndim:
            raise ValueError(
                "Input `sat` must be a 2D array with shape (n_points, n_dims)"
            )

        points = {}
        for i in range(self.ndim):
            points[self.dims[i]] = ("points", sat[:, i])

        interp_data = self.data.interp(points)

        return interp_data.to_array().values.T
