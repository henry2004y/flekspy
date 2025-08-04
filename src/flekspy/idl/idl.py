import numpy as np
import struct
import yt
import xarray as xr


def _read_and_process_data(filename):
    attrs = {"filename": filename}
    attrs["isOuts"] = filename.endswith("outs")
    attrs["npict"] = 1
    attrs["nInstance"] = None if attrs["isOuts"] else 1

    with open(filename, "rb") as f:
        EndChar = "<"  # Endian marker (default: little.)
        RecLenRaw = f.read(4)
        RecLen = (struct.unpack(EndChar + "l", RecLenRaw))[0]
        if RecLen != 79 and RecLen != 500:
            attrs["fileformat"] = "ascii"
        else:
            attrs["fileformat"] = "binary"

    if attrs["fileformat"] == "ascii":
        array, new_attrs = _read_ascii(filename, attrs)
    elif attrs["fileformat"] == "binary":
        try:
            array, new_attrs = _read_binary(filename, attrs)
        except Exception:
            print(
                "It seems the lengths of instances are different. Try slow reading..."
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

    coords = {}
    dims = []
    for i in range(attrs["ndim"]):
        dim_name = attrs["dims"][i]
        dims.append(dim_name)
        dim_idx = varnames.index(dim_name)
        slicer = [0] * 3
        slicer[i] = slice(None, attrs["grid"][i])
        coords[dim_name] = np.squeeze(array[dim_idx, slicer[0], slicer[1], slicer[2]])

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
    _post_process_param(dataset)
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
    new_attrs = _get_file_head(infile, attrs)
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
    if attrs["fileformat"] == "binary":
        new_attrs["end_char"] = "<"
        new_attrs["endian"] = "little"
        record_len_raw = infile.read(4)

        record_len = (struct.unpack(new_attrs["end_char"] + "l", record_len_raw))[0]
        if (record_len > 10000) or (record_len < 0):
            new_attrs["end_char"] = ">"
            new_attrs["endian"] = "big"
            record_len = (struct.unpack(new_attrs["end_char"] + "l", record_len_raw))[0]

        headline = (
            (
                struct.unpack(
                    "{0}{1}s".format(new_attrs["end_char"], record_len),
                    infile.read(record_len),
                )
            )[0]
            .strip()
            .decode()
        )
        new_attrs["unit"] = headline.split()[0]

        (old_len, record_len) = struct.unpack(
            new_attrs["end_char"] + "2l", infile.read(8)
        )
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
            "{0}l{1}3l".format(new_attrs["end_char"], new_attrs["pformat"]),
            infile.read(record_len),
        )
        new_attrs["gencoord"] = new_attrs["ndim"] < 0
        new_attrs["ndim"] = abs(new_attrs["ndim"])
        (old_len, record_len) = struct.unpack(
            new_attrs["end_char"] + "2l", infile.read(8)
        )

        new_attrs["grid"] = np.array(
            struct.unpack(
                "{0}{1}l".format(new_attrs["end_char"], new_attrs["ndim"]),
                infile.read(record_len),
            )
        )
        new_attrs["npoints"] = abs(new_attrs["grid"].prod())

        para_attrs = _read_parameters(infile, new_attrs)
        new_attrs.update(para_attrs)

        var_attrs = _read_variable_names(infile, new_attrs)
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
        new_attrs["para"] = np.zeros(new_attrs["nparam"])
        if new_attrs["nparam"] > 0:
            new_attrs["para"][:] = infile.readline().split()
        names = infile.readline().split()
        new_attrs["dims"] = names[0 : new_attrs["ndim"]]
        new_attrs["variables"] = np.array(names)
        new_attrs["strtime"] = (
            f"{int(new_attrs['time'] // 3600):04d}h{int(new_attrs['time'] % 3600 // 60):02d}m{new_attrs['time'] % 60:06.3f}s"
        )
    return new_attrs


def _read_binary_instance(infile, attrs):
    n_bytes_start = infile.tell()
    new_attrs = _get_file_head(infile, attrs)
    attrs.update(new_attrs)

    nrow = attrs["ndim"] + attrs["nvar"]

    if attrs["pformat"] == "f":
        dtype = np.float32
    else:
        dtype = np.float64

    array = np.empty((nrow, attrs["npoints"]), dtype=dtype)
    dtype_str = f"{attrs['end_char']}{attrs['pformat']}"

    (old_len, record_len) = struct.unpack(attrs["end_char"] + "2l", infile.read(8))
    buffer = infile.read(record_len)
    grid_data = np.frombuffer(
        buffer, dtype=dtype_str, count=attrs["npoints"] * attrs["ndim"]
    )
    array[0 : attrs["ndim"], :] = grid_data.reshape((attrs["ndim"], attrs["npoints"]))

    for i in range(attrs["ndim"], attrs["nvar"] + attrs["ndim"]):
        (old_len, record_len) = struct.unpack(attrs["end_char"] + "2l", infile.read(8))
        buffer = infile.read(record_len)
        array[i, :] = np.frombuffer(buffer, dtype=dtype_str, count=attrs["npoints"])
    infile.read(4)

    shape_new = np.append([nrow], attrs["grid"])
    array = np.reshape(array, shape_new, order="F")
    n_bytes_end = infile.tell()

    return array, n_bytes_end - n_bytes_start, attrs


def _read_parameters(infile, attrs):
    new_attrs = {}
    new_attrs["para"] = np.zeros(attrs["nparam"])
    if attrs["nparam"] > 0:
        (old_len, record_len) = struct.unpack(attrs["end_char"] + "2l", infile.read(8))
        new_attrs["para"][:] = struct.unpack(
            "{0}{1}{2}".format(attrs["end_char"], attrs["nparam"], attrs["pformat"]),
            infile.read(record_len),
        )
    return new_attrs


def _read_variable_names(infile, attrs):
    new_attrs = {}
    (old_len, record_len) = struct.unpack(attrs["end_char"] + "2l", infile.read(8))
    names = (
        struct.unpack(
            "{0}{1}s".format(attrs["end_char"], record_len), infile.read(record_len)
        )
    )[0]
    names = names.decode()
    names.strip()
    names = names.split()

    new_attrs["dims"] = names[0 : attrs["ndim"]]
    new_attrs["variables"] = np.array(names)
    new_attrs["strtime"] = "{0:04d}h{1:02d}m{2:06.3f}s".format(
        int(attrs["time"] // 3600), int(attrs["time"] % 3600 // 60), attrs["time"] % 60
    )
    return new_attrs


def _post_process_param(ds):
    planet_radius = 1.0
    # Not always correct.
    if "param_name" in ds.attrs and "para" in ds.attrs:
        for var, val in zip(ds.attrs["param_name"], ds.attrs["para"]):
            if var == "xSI":
                planet_radius = float(100 * val)

    registry = yt.units.unit_registry.UnitRegistry()
    registry.add("Planet_Radius", planet_radius, yt.units.dimensions.length)
    ds.attrs["registry"] = registry


@xr.register_dataset_accessor("idl")
class IDLAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_slice(self, norm, cut_loc) -> xr.Dataset:
        """Get a 2D slice from the 3D IDL data.
        Args:
            norm: str
                The normal direction of the slice from "x", "y" or "z"
            cur_loc: float
                The position of slicing.
        Return: xarray.Dataset
        """
        return self._obj.sel({norm: cut_loc}, method="nearest")


def read_idl(filename):
    """
    Read IDL format file.
    """
    return _read_and_process_data(filename)
