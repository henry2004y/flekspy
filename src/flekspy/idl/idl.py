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


class Selector:
    def __getitem__(self, keys) -> list:
        self.indices = list(keys)
        if len(self.indices) < 3:
            self.indices += [0] * (3 - len(self.indices))
        return self.indices


class Dataframe:
    def __init__(self):
        self.array = None
        self.name = None
        # Selector for the spatial indices
        self.cut = Selector()
        self.cut[:, :, :]

    def setData(self, dataIn, nameIn):
        assert nameIn.size == dataIn.shape[0]
        assert dataIn.ndim <= 4
        shape = list(dataIn.shape) + [1] * (4 - dataIn.ndim)
        self.array = np.reshape(dataIn, shape)
        self.name = tuple(nameIn)

    def fixDataSize(self):
        self.name = tuple(self.name)
        assert len(self.name) == self.array.shape[0]
        assert self.array.ndim <= 4
        shape = list(self.array.shape) + [1] * (4 - self.array.ndim)
        self.array = np.reshape(self.array, shape)

    def __getitem__(self, keys):
        """Example:
        d["varname", 3:, 1:4:2, 3]
        """
        if type(keys) is str:
            # If the spatial indices are not specified, use self.cut
            keys = [keys] + self.cut.indices
        else:
            keys = list(keys) + [slice(None, None, None)] * (4 - len(keys))
        ivar = self.name.index(keys[0])

        return np.squeeze(self.array[ivar, keys[1], keys[2], keys[3]])


class IDLData(object):
    r"""
    A class used to handle the `*.out` format SWMF data.

    Example:
    >>> ds = IDLData("3d.out")
    >>> dc2d = ds.get_slice("y", 1)
    """

    class Indices(IntEnum):
        """Defines constant indices for IDL data."""

        X = 0
        Y = 1
        Z = 2

    def __init__(self, filename="none"):
        self.filename = filename
        self.isOuts = self.filename[-4:] == "outs"
        self.data = Dataframe()
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

        self.read_data()

    def __post_process_param__(self):

        planet_radius = 1.0

        # Not always correct.
        for var, val in zip(self.param_name, self.para):
            if var == "xSI":
                planet_radius = float(100 * val)

        self.registry = yt.units.unit_registry.UnitRegistry()
        self.registry.add("Planet_Radius", planet_radius, yt.units.dimensions.length)

    def __repr__(self):
        str = (
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
        return str

    def get_domain(self) -> DataContainer:
        """Return data as data container."""
        dataSets = {}
        for varname in self.data.name:
            idx = self.data.name.index(varname)
            unit = get_unit(varname, self.unit)
            dataSets[varname] = yt.YTArray(
                np.squeeze(self.data.array[idx, :, :, :]), unit, registry=self.registry
            )

        labels = ["", "", ""]
        axes = [None, None, None]
        for idim in range(self.ndim):
            name = self.variables[idim]
            idx = self.data.name.index(name)
            labels[idim] = name.upper()
            unit = get_unit("X", self.unit)
            name = name.upper()
            if name in ["X", "Y", "Z"]:
                axes[idim] = yt.YTArray(
                    self.data.array[idx, :, :, :], unit, registry=self.registry
                )

        if self.gencoord and self.ndim == 2:
            dc = DataContainer2D(
                dataSets,
                np.squeeze(axes[0]),
                np.squeeze(axes[1]),
                labels[0],
                labels[1],
                step=self.iter,
                time=self.time,
                filename=self.filename,
                gencoord=True,
            )
        elif self.ndim == 1:
            dc = DataContainer1D(
                dataSets,
                np.squeeze(axes[0]),
                labels[0],
                step=self.iter,
                time=self.time,
                filename=self.filename,
            )
        elif self.ndim == 2:
            dc = DataContainer2D(
                dataSets,
                np.squeeze(axes[0])[:, 0],
                np.squeeze(axes[1])[0, :],
                labels[0],
                labels[1],
                step=self.iter,
                time=self.time,
                filename=self.filename,
            )
        else:
            dc = DataContainer3D(
                dataSets,
                axes[0][:, 0, 0],
                axes[1][0, :, 0],
                axes[2][0, 0, :],
                step=self.iter,
                time=self.time,
                filename=self.filename,
            )

        return dc

    def get_slice(self, norm, cut_loc) -> DataContainer2D:
        """Get a 2D slice from the 3D IDL data.

        Args:
            norm: str
                The normal direction of the slice from "x", "y" or "z"

            cur_loc: float
                The position of slicing.

        Return: DataContainer2D
        """
        domain = self.get_domain()
        ds = domain.get_slice(norm, cut_loc)

        return ds

    def save_data(self, saveName, saveFormat="ascii"):
        # Currently only support ascii output.
        self.save_ascii_instance(saveName)

    def save_ascii_instance(self, saveName):
        with open(saveName, "w") as f:
            f.write(self.unit + "\n")
            f.write(
                "{0:d}\t{1:e}\t{2:d}\t{3:d}\t{4:d}\n".format(
                    self.iter, self.time, self.ndim, self.nparam, self.nvar
                )
            )
            [f.write("{0:d}\t".format(i)) for i in self.grid]
            f.write("\n")
            if self.nparam > 0:
                [f.write("{0:e}\t".format(i)) for i in self.para]
                f.write("\n")
            [f.write(i + " ") for i in self.variables]
            f.write("\n")

            nk = self.grid[2] if self.ndim > 2 else 1
            nj = self.grid[1] if self.ndim > 1 else 1
            ni = self.grid[0]
            for kk in range(nk):
                for jj in range(nj):
                    for ii in range(ni):
                        [
                            f.write("{0:e}\t".format(i))
                            for i in self.data.array[:, ii, jj, kk]
                        ]
                        f.write("\n")

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
            self.read_ascii()
        elif self.fileformat == "binary":
            try:
                self.read_binary()
            except:
                print(
                    "It seems the lengths of instances are different. Try slow reading..."
                )
                self.read_binary_slow()
        else:
            raise ValueError(f"Unknown format = {self.fileformat}")

        nsize = self.ndim + self.nvar
        self.data.name = tuple(self.variables)[0:nsize]
        self.data.fixDataSize()
        self.param_name = self.variables[nsize:]
        self.__post_process_param__()

    def read_ascii(self):
        if self.nInstance is None:
            # Count the number of instances.
            with open(self.filename, "r") as f:
                for i, l in enumerate(f):
                    pass
                nLineFile = i + 1

            with open(self.filename, "r") as f:
                self.nInstanceLength = self.read_ascii_instance(f)

            self.nInstance = round(nLineFile / self.nInstanceLength)

        nLineSkip = (self.npict) * self.nInstanceLength if self.isOuts else 0
        with open(self.filename, "r") as f:
            if nLineSkip > 0:
                for i, line in enumerate(f):
                    if i == nLineSkip - 1:
                        break
            self.read_ascii_instance(f)

    def read_ascii_instance(self, infile):
        self.get_file_head(infile)
        nrow = self.ndim + self.nvar
        ncol = self.npoints
        self.data.array = np.zeros((nrow, ncol))

        for i, line in enumerate(infile.readlines()):
            parts = line.split()

            if i >= self.npoints:
                break

            for j, p in enumerate(parts):
                self.data.array[j][i] = float(p)

        shapeNew = np.append([nrow], self.grid)
        self.data.array = np.reshape(self.data.array, shapeNew, order="F")
        nline = 5 + self.npoints if self.nparam > 0 else 4 + self.npoints

        return nline

    def read_binary(self):
        if self.nInstance is None:
            with open(self.filename, "rb") as f:
                self.read_binary_instance(f)
                self.nInstanceLength = f.tell()
                f.seek(0, 2)
                endPos = f.tell()
            self.nInstance = round(endPos / self.nInstanceLength)

        with open(self.filename, "rb") as f:
            if self.isOuts:
                f.seek((self.npict) * self.nInstanceLength, 0)
            self.read_binary_instance(f)

    def read_binary_slow(self):
        with open(self.filename, "rb") as f:
            if self.isOuts:
                # Skip previous instances
                for i in range(self.npict):
                    self.read_binary_instance(f)
            self.read_binary_instance(f)

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
        self.get_file_head(infile)
        nrow = self.ndim + self.nvar

        if self.pformat == "f":
            dtype = np.float32
        else:
            dtype = np.float64

        self.data.array = np.empty((nrow, self.npoints), dtype=dtype)
        dtype_str = f"{self.end_char}{self.pformat}"

        # Get the grid points
        (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
        buffer = infile.read(record_len)
        grid_data = np.frombuffer(
            buffer, dtype=dtype_str, count=self.npoints * self.ndim
        )
        self.data.array[0 : self.ndim, :] = grid_data.reshape((self.ndim, self.npoints))

        # Get the actual data and sort
        for i in range(self.ndim, self.nvar + self.ndim):
            (old_len, record_len) = struct.unpack(self.end_char + "2l", infile.read(8))
            buffer = infile.read(record_len)
            self.data.array[i, :] = np.frombuffer(
                buffer, dtype=dtype_str, count=self.npoints
            )
        # Consume the last record length
        infile.read(4)

        shape_new = np.append([nrow], self.grid)
        self.data.array = np.reshape(self.data.array, shape_new, order="F")

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

    def plot(self, *dvname, **kwargs):
        """Plot 1D IDL outputs.

        Args:
            *dvname (str): variable names
            **kwargs: keyword argument to be passed to `plot`.
        """
        x = self.data["x"]
        nvar = len(dvname)

        f, axes = plt.subplots(nvar, 1, constrained_layout=True, sharex=True)
        axes = np.array(axes)  # in case nrows == ncols == 1
        axes = axes.reshape(-1)
        for isub, ax in zip(range(nvar), axes):
            w = self.data[dvname[isub]]
            p = ax.plot(x, w, **kwargs)

            ax.set_xlabel("x", fontsize=16)
            ax.set_ylabel(dvname[isub], fontsize=16)

        return axes

    def pcolormesh(self, *dvname, scale: bool = True, **kwargs):
        """Plot 2D pcolormeshes of variables.

        Args:
            *dvname (str): variable names
            scale (bool): whether to scale the plots according to the axis range.
                Default True.
        """
        x = self.data["x"]
        y = self.data["y"]

        nvar = len(dvname)

        f, axes = plt.subplots(
            nvar, 1, constrained_layout=True, sharex=True, sharey=True
        )
        axes = np.array(axes)  # in case nRow = nCol = 1
        aspect = (y.max() - y.min()) / (x.max() - x.min())
        axes = axes.reshape(-1)

        for isub, ax in zip(range(nvar), axes):
            w = self.data[dvname[isub]]
            p = ax.pcolormesh(x, y, w, cmap="turbo", **kwargs)
            cb = f.colorbar(p, ax=ax, pad=0.02)
            if scale:
                ax.set_aspect(aspect)
            ax.set_xlabel("X", fontsize=16)
            ax.set_ylabel("Y", fontsize=16)
            ax.set_title(dvname[isub], fontsize=16)

        return axes

    def extract_data(self, sat: np.ndarray) -> np.ndarray:
        """Extract data at a series of locations.

        Args:
            sat (np.ndarray): 2D/3D point locations.

        Returns:
            np.ndarray: 2D array of variables at each point.
        """
        satData = None
        if sat.ndim == 2 and sat.shape[1] >= self.ndim:
            nVar = self.nvar + self.ndim
            nPoint = sat.shape[0]
            satData = np.zeros((nPoint, nVar))
            for i in range(nPoint):
                satData[i, :] = np.squeeze(self.get_data(sat[i, :]))
        return satData

    def get_data(self, loc: np.ndarray) -> np.ndarray:
        """Extract data at a given point using bilinear interpolation.

        Args:
            loc (np.ndarray): 2D/3D point location.

        Returns:
            np.ndarray: 1D array of saved variables at the survey point.
        """
        if self.ndim == 2:
            # Find the indices of the surrounding grid points
            i1, j1 = 0, 0
            while self.data["x"][i1, 0] < loc[self.Indices.X]:
                i1 += 1
            while self.data["y"][0, j1] < loc[self.Indices.Y]:
                j1 += 1
            i0 = i1 - 1
            j0 = j1 - 1

            # Calculate the weights
            wx0 = (self.data["x"][i1, 0] - loc[self.Indices.X]) / (
                self.data["x"][i1, 0] - self.data["x"][i0, 0]
            )
            wy0 = (self.data["y"][0, j1] - loc[self.Indices.Y]) / (
                self.data["y"][0, j1] - self.data["y"][0, j0]
            )
            wx1 = 1.0 - wx0
            wy1 = 1.0 - wy0

            # Calculate the interpolated values
            res = (
                self.data.array[:, i0, j0] * wx0 * wy0
                + self.data.array[:, i0, j1] * wx0 * wy1
                + self.data.array[:, i1, j0] * wx1 * wy0
                + self.data.array[:, i1, j1] * wx1 * wy1
            )
        elif self.ndim == 3:
            i1, j1, k1 = 0, 0, 0
            while self.data["x"][i1, 0, 0] < loc[self.Indices.X]:
                i1 += 1
            while self.data["y"][0, j1, 0] < loc[self.Indices.Y]:
                j1 += 1
            while self.data["z"][0, 0, k1] < loc[self.Indices.Z]:
                k1 += 1

            i0 = i1 - 1
            j0 = j1 - 1
            k0 = k1 - 1

            wx0 = (self.data["x"][i1, 0, 0] - loc[self.Indices.X]) / (
                self.data["x"][i1, 0, 0] - self.data["x"][i0, 0, 0]
            )
            wy0 = (self.data["y"][0, j1, 0] - loc[self.Indices.Y]) / (
                self.data["y"][0, j1, 0] - self.data["y"][0, j0, 0]
            )
            wz0 = (self.data["z"][0, 0, k1] - loc[self.Indices.Z]) / (
                self.data["z"][0, 0, k1] - self.data["z"][0, 0, k0]
            )

            wx1 = 1.0 - wx0
            wy1 = 1.0 - wy0
            wz1 = 1.0 - wz0

            w = np.array(
                [
                    [
                        [wx0 * wy0 * wz0, wx0 * wy0 * wz1],
                        [wx0 * wy1 * wz0, wx0 * wy1 * wz1],
                    ],
                    [
                        [wx1 * wy0 * wz0, wx1 * wy0 * wz1],
                        [wx1 * wy1 * wz0, wx1 * wy1 * wz1],
                    ],
                ]
            )

            res = np.sum(
                w * self.data.array[:, i0 : i0 + 2, j0 : j0 + 2, k0 : k0 + 2],
                axis=(1, 2, 3),
            )

        return res


class IDLDataX(IDLData):
    def __init__(self, filename="none"):
        super().__init__(filename)

        coords = {}
        dims = []
        if self.ndim >= 1:
            dims.append(self.dims[0])
            x_idx = list(self.data.name).index(self.dims[0])
            coords[self.dims[0]] = np.squeeze(self.data.array[x_idx, :self.grid[0], 0, 0])
        if self.ndim >= 2:
            dims.append(self.dims[1])
            y_idx = list(self.data.name).index(self.dims[1])
            coords[self.dims[1]] = np.squeeze(self.data.array[y_idx, 0, :self.grid[1], 0])
        if self.ndim >= 3:
            dims.append(self.dims[2])
            z_idx = list(self.data.name).index(self.dims[2])
            coords[self.dims[2]] = np.squeeze(self.data.array[z_idx, 0, 0, :self.grid[2]])

        data_vars = {}
        for i, var_name in enumerate(self.data.name):
            if var_name not in self.dims:
                data_slice = self.data.array[i, :self.grid[0], :self.grid[1] if self.ndim > 1 else 1, :self.grid[2] if self.ndim > 2 else 1]
                data_vars[var_name] = (dims, np.squeeze(data_slice))

        self.data = xr.Dataset(data_vars, coords=coords)

        self.data.attrs['time'] = self.time
        self.data.attrs['iter'] = self.iter
        self.data.attrs['unit'] = self.unit
        self.data.attrs['gencoord'] = self.gencoord

    def get_domain(self):
        """Return data as an xarray.Dataset."""
        return self.data

    def get_slice(self, norm, cut_loc):
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
            nvar, 1, constrained_layout=True, sharex=True, sharey=True,
            figsize=kwargs.pop('figsize', (6, 4*nvar))
        )
        if nvar == 1:
            axes = [axes]

        for i, var in enumerate(dvname):
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'turbo'

            self.data[var].plot.pcolormesh(ax=axes[i], **kwargs)
            axes[i].set_title(var)

        if scale:
            x_coords = self.data.coords[self.dims[0]]
            y_coords = self.data.coords[self.dims[1]]
            aspect_ratio = (y_coords.max() - y_coords.min()) / (x_coords.max() - x_coords.min())
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
            raise ValueError("Input `sat` must be a 2D array with shape (n_points, n_dims)")

        points = {}
        for i in range(self.ndim):
            points[self.dims[i]] = ('points', sat[:, i])

        interp_data = self.data.interp(points)

        return interp_data.to_array().values.T
