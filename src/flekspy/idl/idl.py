import numpy as np
import struct
import xarray as xr
from scipy.constants import mu_0, e
import xugrid as xu
from scipy.spatial import Delaunay
from flekspy.util.logger import get_logger

logger = get_logger(name=__name__)


def _read_and_process_data(filename):
    attrs = {"filename": filename}
    attrs["isOuts"] = filename.endswith("outs")
    attrs["npict"] = 1
    attrs["nInstance"] = None if attrs["isOuts"] else 1

    with open(filename, "rb") as f:
        rec_len_raw = f.read(4)
        rec_len = struct.unpack("<l", rec_len_raw)[0]
        if rec_len != 79 and rec_len != 500:
            attrs["fileformat"] = "ascii"
        else:
            attrs["fileformat"] = "binary"

    if attrs["fileformat"] == "ascii":
        array, new_attrs = _read_ascii(filename, attrs)
    elif attrs["fileformat"] == "binary":
        try:
            array, new_attrs = _read_binary(filename, attrs)
        except Exception:
            logger.warning(
                "It seems the lengths of instances are different. Try slow reading...",
                exc_info=True,
            )
            array, new_attrs = _read_binary_slow(filename, attrs)
    else:
        raise ValueError(f"Unknown format = {attrs['fileformat']}")

    attrs.update(new_attrs)

    nsize = attrs["ndim"] + attrs["nvar"]
    varnames = tuple(attrs["variables"])[0:nsize]
    attrs["param_name"] = attrs["variables"][nsize:]

    # Reshape data if ndim < 3
    shape = list(array.shape) + [1] * (4 - array.ndim)
    array = np.reshape(array, shape)

    if attrs.get("gencoord", False):
        if attrs["ndim"] == 2:
            x_coord_name = attrs["dims"][0]
            y_coord_name = attrs["dims"][1]
            # The varnames is a tuple of strings.
            x_index = varnames.index(x_coord_name)
            y_index = varnames.index(y_coord_name)
            node_x = np.squeeze(array[x_index, ...])
            node_y = np.squeeze(array[y_index, ...])

            # Create grid topology from points via Delaunay triangulation.
            points = np.vstack((node_x, node_y)).T
            triangulation = Delaunay(points)
            faces = triangulation.simplices

            grid = xu.Ugrid2d(
                node_x=node_x,
                node_y=node_y,
                fill_value=-1,
                face_node_connectivity=faces,
                name="mesh2d",  # UGRID required name.
            )

            data_vars = {}
            for i, var_name in enumerate(varnames):
                if var_name not in attrs["dims"]:
                    data_slice = np.squeeze(array[i, ...])
                    # Data is located at the nodes of the grid.
                    data_vars[var_name] = (grid.node_dimension, data_slice)
            dataset = xu.UgridDataset(xr.Dataset(data_vars), grids=[grid])
        else:
            # Fallback to old behavior for 1D or 3D unstructured grids.
            data_vars = {}
            dims = ("n_points",)
            for i, var_name in enumerate(varnames):
                data_slice = np.squeeze(array[i, ...])
                data_vars[var_name] = (dims, data_slice)

            dataset = xr.Dataset(data_vars)
    else:
        coords = {}
        dims = []
        for i in range(attrs["ndim"]):
            dim_name = attrs["dims"][i]
            dims.append(dim_name)
            dim_idx = varnames.index(dim_name)

            start = array[dim_idx, 0, 0, 0]
            stop_slicer = [0] * 3
            stop_slicer[i] = -1
            stop = array[(dim_idx,) + tuple(stop_slicer)]

            coords[dim_name] = np.linspace(start, stop, attrs["grid"][i])

        data_vars = {}
        for i, var_name in enumerate(varnames):
            if var_name not in attrs["dims"]:
                slicer = [i]
                for d in range(3):
                    if d < attrs["ndim"]:
                        slicer.append(slice(attrs["grid"][d]))
                    else:
                        slicer.append(slice(1))
                data_slice = array[tuple(slicer)]
                data_vars[var_name] = (dims, np.squeeze(data_slice))

        dataset = xr.Dataset(data_vars, coords=coords)
    dataset.attrs = attrs
    # TODO: Implement a more robust unit handling system.
    return dataset


def _read_ascii(filename, attrs):
    if attrs.get("nInstance") is None:
        with open(filename, "r") as f:
            for i, l in enumerate(f):
                pass
            nLineFile = i + 1

        with open(filename, "r") as f:
            nInstanceLength, _, _ = _read_ascii_instance(f, attrs)
            attrs["nInstanceLength"] = nInstanceLength

        attrs["nInstance"] = round(nLineFile / attrs["nInstanceLength"])

    nLineSkip = (attrs["npict"]) * attrs["nInstanceLength"] if attrs["isOuts"] else 0
    with open(filename, "r") as f:
        if nLineSkip > 0:
            for i, line in enumerate(f):
                if i == nLineSkip - 1:
                    break
        _, array, new_attrs = _read_ascii_instance(f, attrs)
    attrs.update(new_attrs)
    return array, attrs


def _read_ascii_instance(infile, attrs):
    new_attrs, _ = _get_file_head(infile, attrs)
    attrs.update(new_attrs)
    nrow = attrs["ndim"] + attrs["nvar"]
    ncol = attrs["npoints"]
    array = np.zeros((nrow, ncol))

    for i, line in enumerate(infile.readlines()):
        parts = line.split()

        if i >= attrs["npoints"]:
            break

        for j, p in enumerate(parts):
            array[j][i] = float(p)

    shapeNew = np.append([nrow], attrs["grid"])
    array = np.reshape(array, shapeNew, order="F")
    nline = 5 + attrs["npoints"] if attrs["nparam"] > 0 else 4 + attrs["npoints"]

    return nline, array, attrs


def _read_binary(filename, attrs):
    if attrs.get("nInstance") is None:
        with open(filename, "rb") as f:
            _, n_bytes, new_attrs = _read_binary_instance(f, attrs)
            attrs.update(new_attrs)
            attrs["nInstanceLength"] = n_bytes
            f.seek(0, 2)
            endPos = f.tell()
        attrs["nInstance"] = round(endPos / attrs["nInstanceLength"])

    with open(filename, "rb") as f:
        if attrs["isOuts"]:
            f.seek((attrs["npict"]) * attrs["nInstanceLength"], 0)
        array, _, new_attrs = _read_binary_instance(f, attrs)
        attrs.update(new_attrs)
        return array, attrs


def _read_binary_slow(filename, attrs):
    with open(filename, "rb") as f:
        if attrs["isOuts"]:
            # Skip previous instances
            for i in range(attrs["npict"]):
                _read_binary_instance(f, attrs)
        array, _, new_attrs = _read_binary_instance(f, attrs)
        attrs.update(new_attrs)
        return array, attrs


def _get_file_head(infile, attrs):
    new_attrs = {}
    end_char = ""
    if attrs["fileformat"] == "binary":
        record_len_raw = infile.read(4)
        record_len = struct.unpack("<l", record_len_raw)[0]

        if (record_len > 10000) or (record_len < 0):
            end_char = ">"
            new_attrs["endian"] = "big"
            record_len = struct.unpack(">l", record_len_raw)[0]
        else:
            end_char = "<"
            new_attrs["endian"] = "little"

        headline = (
            (
                struct.unpack(
                    f"{end_char}{record_len}s",
                    infile.read(record_len),
                )
            )[0]
            .strip()
            .decode()
        )
        new_attrs["unit"] = headline.split()[0]

        (_, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))
        new_attrs["pformat"] = "f"
        if record_len > 20:
            new_attrs["pformat"] = "d"
        (
            new_attrs["iter"],
            new_attrs["time"],
            new_attrs["ndim"],
            new_attrs["nparam"],
            new_attrs["nvar"],
        ) = struct.unpack(
            f"{end_char}l{new_attrs['pformat']}3l",
            infile.read(record_len),
        )
        new_attrs["gencoord"] = new_attrs["ndim"] < 0
        new_attrs["ndim"] = abs(new_attrs["ndim"])
        (_, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))

        new_attrs["grid"] = np.array(
            struct.unpack(
                f"{end_char}{new_attrs['ndim']}l",
                infile.read(record_len),
            )
        )
        new_attrs["npoints"] = abs(new_attrs["grid"].prod())

        para_attrs = _read_parameters(infile, new_attrs, end_char)
        new_attrs.update(para_attrs)

        var_attrs = _read_variable_names(infile, new_attrs, end_char)
        new_attrs.update(var_attrs)
    else:
        headline = infile.readline().strip()
        new_attrs["unit"] = headline.split()[0]
        parts = infile.readline().split()
        new_attrs["iter"] = int(parts[0])
        new_attrs["time"] = float(parts[1])
        new_attrs["ndim"] = int(parts[2])
        new_attrs["gencoord"] = new_attrs["ndim"] < 0
        new_attrs["ndim"] = abs(new_attrs["ndim"])
        new_attrs["nparam"] = int(parts[3])
        new_attrs["nvar"] = int(parts[4])
        grid = [int(x) for x in infile.readline().split()]
        new_attrs["grid"] = np.array(grid)
        new_attrs["npoints"] = abs(new_attrs["grid"].prod())
        new_attrs["parameters"] = np.zeros(new_attrs["nparam"])
        if new_attrs["nparam"] > 0:
            new_attrs["parameters"][:] = infile.readline().split()
        names = infile.readline().split()
        new_attrs["dims"] = names[0 : new_attrs["ndim"]]
        new_attrs["variables"] = np.array(names)
        new_attrs["strtime"] = (
            f"{int(new_attrs['time'] // 3600):04d}h{int(new_attrs['time'] % 3600 // 60):02d}m{new_attrs['time'] % 60:06.3f}s"
        )
    return new_attrs, end_char


def _read_binary_instance(infile, attrs):
    n_bytes_start = infile.tell()
    new_attrs, end_char = _get_file_head(infile, attrs)
    attrs.update(new_attrs)

    nrow = attrs["ndim"] + attrs["nvar"]

    if attrs["pformat"] == "f":
        dtype = np.float32
    else:
        dtype = np.float64

    array = np.empty((nrow, attrs["npoints"]), dtype=dtype)
    dtype_str = f"{end_char}{attrs['pformat']}"

    (_, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))
    buffer = infile.read(record_len)
    grid_data = np.frombuffer(
        buffer, dtype=dtype_str, count=attrs["npoints"] * attrs["ndim"]
    )
    array[0 : attrs["ndim"], :] = grid_data.reshape((attrs["ndim"], attrs["npoints"]))

    for i in range(attrs["ndim"], attrs["nvar"] + attrs["ndim"]):
        (_, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))
        buffer = infile.read(record_len)
        array[i, :] = np.frombuffer(buffer, dtype=dtype_str, count=attrs["npoints"])
    infile.read(4)

    shape_new = np.append([nrow], attrs["grid"])
    array = np.reshape(array, shape_new, order="F")
    n_bytes_end = infile.tell()

    return array, n_bytes_end - n_bytes_start, attrs


def _read_parameters(infile, attrs, end_char):
    new_attrs = {}
    new_attrs["parameters"] = np.zeros(attrs["nparam"])
    if attrs["nparam"] > 0:
        (old_len, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))
        new_attrs["parameters"][:] = struct.unpack(
            f"{end_char}{attrs['nparam']}{attrs['pformat']}",
            infile.read(record_len),
        )
    return new_attrs


def _read_variable_names(infile, attrs, end_char):
    new_attrs = {}
    (_, record_len) = struct.unpack(f"{end_char}2l", infile.read(8))
    names = (
        struct.unpack(f"{end_char}{record_len}s", infile.read(record_len))
    )[0]
    names = names.decode()
    names.strip()
    names = names.split()

    new_attrs["dims"] = names[0 : attrs["ndim"]]
    new_attrs["variables"] = np.array(names)
    new_attrs["strtime"] = (
        f"{int(attrs['time'] // 3600):04d}h{int(attrs['time'] % 3600 // 60):02d}m{attrs['time'] % 60:06.3f}s"
    )
    return new_attrs


@xr.register_dataset_accessor("idl")
class IDLAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_slice(self, norm, cut_loc) -> xr.Dataset:
        """Get a 2D slice from the 3D IDL data.
        Args:
            norm (str): The normal direction of the slice from "x", "y" or "z"
            cur_loc (float): The position of slicing.
        Return: xarray.Dataset
        """
        return self._obj.sel({norm: cut_loc}, method="nearest")

    def get_pressure_anisotropy(self, species: int) -> xr.DataArray:
        """
        Calculates the pressure anisotropy for a given species.

        The pressure anisotropy is defined as the ratio of perpendicular
        to parallel pressure, calculated with respect to the local magnetic
        field direction.

        This method requires the dataset to contain the magnetic field
        components ('Bx', 'By', 'Bz') and the full pressure tensor for the
        specified species (e.g., 'pXXS0', 'pYYS0', etc. for species 0).

        Args:
            species (int): The species index for which to calculate the
                           pressure anisotropy.

        Returns:
            xarray.DataArray: A DataArray containing the pressure anisotropy,
                              with the same dimensions as the input data.
        """

        # Extract pressure tensor components
        p_components = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
        pxx, pyy, pzz, pxy, pxz, pyz = (
            self._obj[f"p{c}S{species}"] for c in p_components
        )

        # Extract magnetic field components
        bx, by, bz = (self._obj[c] for c in ["Bx", "By", "Bz"])

        # Calculate magnetic field magnitude and unit vector
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        b_hat_x = bx / b_mag
        b_hat_y = by / b_mag
        b_hat_z = bz / b_mag

        # Calculate parallel pressure
        p_parallel = (
            pxx * b_hat_x**2
            + pyy * b_hat_y**2
            + pzz * b_hat_z**2
            + 2
            * (
                pxy * b_hat_x * b_hat_y
                + pyz * b_hat_y * b_hat_z
                + pxz * b_hat_x * b_hat_z
            )
        )

        # Calculate total pressure (trace of the pressure tensor)
        p_total = pxx + pyy + pzz

        # Calculate perpendicular pressure
        p_perp = (p_total - p_parallel) / 2.0

        # Calculate pressure anisotropy
        anisotropy = p_perp / p_parallel
        anisotropy.name = f"pressure_anisotropy_S{species}"

        return anisotropy

    def get_current_density(self) -> xr.Dataset:
        """
        Calculates the current density vector from the curl of the magnetic field.

        This method computes the current density J = (1/mu0) * curl(B).
        It requires the dataset to contain 2D or 3D magnetic field components
        ('Bx', 'By', 'Bz') on a structured grid.

        For 2D data, it is assumed that the simulation is in a plane (e.g., XY)
        and the gradients in the missing dimension (e.g., z) are zero.

        If the dataset attribute "unit" is "PLANETARY", it is assumed that the
        magnetic field is in nT, and the length unit is "rPlanet". The resulting
        current density is returned in µA/m^2.
        Otherwise, the magnetic field is assumed to be in Tesla.

        Returns:
            xarray.Dataset: A Dataset containing the three components of the
                            current density ('jx', 'jy', 'jz').

        Raises:
            KeyError: If the magnetic field components are not in the dataset.
            ValueError: If the data is not 2D or 3D.
            NotImplementedError: If the grid is unstructured.
        """
        # Check if magnetic field components are present
        if not all(c in self._obj for c in ["Bx", "By", "Bz"]):
            raise KeyError(
                "Magnetic field components ('Bx', 'By', 'Bz') not found in the dataset."
            )

        if self._obj.attrs.get("gencoord", False):
            raise NotImplementedError(
                "Current density calculation is not supported for unstructured grids yet."
            )

        if self._obj.attrs["ndim"] not in [2, 3]:
            raise ValueError("Current density calculation requires 2D or 3D data.")

        bx, by, bz = (self._obj[c] for c in ["Bx", "By", "Bz"])

        # Get coordinate names and values in the order of the data dimensions
        param_names = list(self._obj.attrs["param_name"])
        params = self._obj.attrs["parameters"]
        is_planetary = self._obj.attrs.get("unit") == "PLANETARY"

        # Get length conversion from attrs
        if is_planetary:
            length_index = param_names.index("rPlanet")
            length = params[length_index]
        else:
            length = 1.0

        coords = [self._obj[dim].values * length for dim in bx.dims]  # [m]

        # Calculate gradients of each magnetic field component
        dbx_d_dims = np.gradient(bx.values, *coords)
        dby_d_dims = np.gradient(by.values, *coords)
        dbz_d_dims = np.gradient(bz.values, *coords)

        grad_bx = dict(zip(bx.dims, dbx_d_dims))
        grad_by = dict(zip(by.dims, dby_d_dims))
        grad_bz = dict(zip(bz.dims, dbz_d_dims))

        # jx = d(Bz)/dy - d(By)/dz
        jx = grad_bz.get("y", 0.0) - grad_by.get("z", 0.0)

        # jy = d(Bx)/dz - d(Bz)/dx
        jy = grad_bx.get("z", 0.0) - grad_bz.get("x", 0.0)

        # jz = d(By)/dx - d(Bx)/dy
        jz = grad_by.get("x", 0.0) - grad_bx.get("y", 0.0)

        # Handle units and convert to µA/m^2
        if self._obj.attrs.get("unit") == "PLANETARY":
            # B is in nT, curl(B) is in nT/m. Convert to T/m by 1e-9.
            # J = curl(B_T) / mu0 = curl(B_nT * 1e-9) / mu0
            # Final conversion to µA/m^2
            conversion_factor = (1e-9 / mu_0) * 1e6
        else:
            # Assuming B is in T, curl(B) is in T/m
            # J = curl(B) / mu0
            # Final conversion to µA/m^2
            conversion_factor = (1.0 / mu_0) * 1e6

        jx *= conversion_factor
        jy *= conversion_factor
        jz *= conversion_factor

        current_density = xr.Dataset(
            {
                "jx": (bx.dims, jx, {"units": "µA/m^2"}),
                "jy": (by.dims, jy, {"units": "µA/m^2"}),
                "jz": (bz.dims, jz, {"units": "µA/m^2"}),
            },
            coords=self._obj.coords,
        )

        return current_density

    def get_current_density_from_definition(self, species: list[int]) -> xr.Dataset:
        """
        Calculates the current density from its definition J = n * q * v.

        This method computes the current density by summing the contributions
        from the specified particle species. It requires the dataset to contain
        the mass density (e.g., 'rhoS0') and velocity components (e.g., 'uxS0',
        'uyS0', 'uzS0') for each species. The particle mass and charge for each
        species are retrieved dynamically from the dataset's 'param_name' and
        'parameters' attributes.

        If the dataset attribute "unit" is "PLANETARY", it is assumed that the
        mass densities are in [amu/cc], velocities are in [km/s], mass is in
        [amu] and charge is normalized to the elementary charge. The resulting
        current density is returned in µA/m^2. Otherwise, SI units are
        assumed and the result is also returned in µA/m^2.

        Args:
            species (list[int]): A list of species indices for which to
                                 calculate the current density.

        Returns:
            xarray.Dataset: A Dataset containing the three components of the
                            total current density ('jx', 'jy', 'jz'), with
                            units attribute set to 'µA/m^2'.
        """
        total_jx = None
        total_jy = None
        total_jz = None

        param_names = list(self._obj.attrs["param_name"])
        params = self._obj.attrs["parameters"]
        is_planetary = self._obj.attrs.get("unit") == "PLANETARY"

        for s in species:
            # Get mass density and velocity for the species
            mass_density = self._obj[f"rhoS{s}"]
            ux = self._obj[f"uxS{s}"]
            uy = self._obj[f"uyS{s}"]
            uz = self._obj[f"uzS{s}"]

            # Get particle mass and charge from attrs
            mass_index = param_names.index(f"mS{s}")
            charge_index = param_names.index(f"qS{s}")
            particle_mass = params[mass_index]
            charge = params[charge_index]

            if is_planetary:
                charge *= e

            # Calculate number density
            number_density = mass_density / particle_mass

            # Calculate current density for the species
            jx = number_density * charge * ux
            jy = number_density * charge * uy
            jz = number_density * charge * uz

            if total_jx is None:
                total_jx = jx
                total_jy = jy
                total_jz = jz
            else:
                total_jx += jx
                total_jy += jy
                total_jz += jz

        # Apply conversion factors to get to µA/m^2
        if is_planetary:
            # j_raw has units (1/cc) * C * (km/s)
            # convert to A/m^2: (1e6/m^3) * C * (1e3 m/s) -> factor 1e9
            # convert to uA/m^2: factor 1e9 * 1e6 = 1e15
            conversion_factor = 1e15
        else:  # SI
            # j_raw has units (1/m^3) * C * (m/s) = A/m^2
            # convert to uA/m^2: factor 1e6
            conversion_factor = 1e6

        total_jx *= conversion_factor
        total_jy *= conversion_factor
        total_jz *= conversion_factor

        current_density = xr.Dataset(
            {
                "jx": (total_jx.dims, total_jx.values, {"units": "µA/m^2"}),
                "jy": (total_jy.dims, total_jy.values, {"units": "µA/m^2"}),
                "jz": (total_jz.dims, total_jz.values, {"units": "µA/m^2"}),
            },
            coords=self._obj.coords,
        )

        return current_density


def read_idl(filename):
    """
    Read IDL format file.
    """
    return _read_and_process_data(filename)
