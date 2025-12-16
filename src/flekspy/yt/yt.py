from pathlib import Path
import numpy as np

import yt
from yt.funcs import setdefaultattr
from yt.frontends.boxlib.api import BoxlibHierarchy, BoxlibDataset
from yt.fields.field_info_container import FieldInfoContainer
from yt.data_objects.selection_objects.data_selection_objects import (
    YTSelectionContainer,
)
from yt.visualization.profile_plotter import PhasePlot

import xarray as xr

from flekspy.util import get_unit
import flekspy.xarray

class FLEKSFieldInfo(FieldInfoContainer):
    l_units = "code_length"
    v_units = "code_velocity"
    p_units = "code_pressure"
    b_units = "code_magnetic"
    e_units = "code_magnetic * code_velocity"
    rho_units = "code_density"
    mass_units = "code_mass"

    _base_fields = (
        ("Bx", (b_units, ["magnetic_field_x"], r"B_x")),
        ("By", (b_units, ["magnetic_field_y"], r"B_y")),
        ("Bz", (b_units, ["magnetic_field_z"], r"B_z")),
        ("Ex", (e_units, [], r"E_x")),
        ("Ey", (e_units, [], r"E_y")),
        ("Ez", (e_units, [], r"E_z")),
        ("X", (l_units, [], r"X")),
        ("Y", (l_units, [], r"Y")),
        ("Z", (l_units, [], r"Z")),
    )

    _species_fields_template = (
        ("rho", (rho_units, [], r"\rho")),
        ("ux", (v_units, [], r"u_x")),
        ("uy", (v_units, [], r"u_y")),
        ("uz", (v_units, [], r"u_z")),
        ("pxx", (p_units, [], r"P_{xx}")),
        ("pyy", (p_units, [], r"P_{yy}")),
        ("pzz", (p_units, [], r"P_{zz}")),
        ("pxy", (p_units, [], r"P_{xy}")),
        ("pxz", (p_units, [], r"P_{xz}")),
        ("pyz", (p_units, [], r"P_{yz}")),
    )

    known_other_fields = _base_fields

    known_particle_fields = (
        ("particle_weight", (mass_units, ["p_w"], r"weight")),
        ("particle_position_x", (l_units, ["p_x"], "x")),
        ("particle_position_y", (l_units, ["p_y"], "y")),
        ("particle_position_z", (l_units, ["p_z"], "z")),
        ("particle_velocity_x", (v_units, ["p_ux"], r"u_x")),
        ("particle_velocity_y", (v_units, ["p_uy"], r"u_y")),
        ("particle_velocity_z", (v_units, ["p_uz"], r"u_z")),
    )

    extra_union_fields = ((mass_units, "particle_mass"),)

    def __init__(self, ds, field_list):
        super(FLEKSFieldInfo, self).__init__(ds, field_list)

        # setup nodal flag information
        for field in ds.index.raw_fields:
            finfo = self.__getitem__(("raw", field))
            finfo.nodal_flag = ds.nodal_flags[field]

    def setup_fluid_fields(self):
        import re

        from yt.fields.magnetic_field import setup_magnetic_field_aliases

        for field in self.known_other_fields:
            fname = field[0]
            # Try to alias to boxlib first, then raw
            original_name = ("boxlib", fname)
            if original_name not in self:
                original_name = ("raw", fname)

            self.alias(("mesh", fname), original_name)

        # Dynamically alias species fields from the input dataset
        # Use field_list instead of index.raw_fields to avoid forcing index construction
        species_regex = re.compile(r"^(?P<base>.*)s(?P<idx>\d+)$")

        # field_list typically contains tuples like ('boxlib', 'varname') or ('raw', 'varname')
        # We need to scan for on-disk fields.
        for ftype, fname in self.ds.field_list:
            # We are interested in fields that might be raw fluid fields
            if ftype not in ("boxlib", "raw"):
                continue

            match = species_regex.match(fname)
            if match:
                base = match.group("base")
                for template_field, props in self._species_fields_template:
                    if base == template_field:
                        # Construct proper LaTeX display name merging subscripts
                        base_display = props[2]
                        suffix = f"s{match.group('idx')}"
                        if "_" in base_display:
                            name_part, sub_part = base_display.split("_", 1)
                            if sub_part.startswith("{") and sub_part.endswith("}"):
                                sub_content = sub_part[1:-1]
                                display_name = f"{name_part}_{{{sub_content},{suffix}}}"
                            else:
                                display_name = f"{name_part}_{{{sub_part},{suffix}}}"
                        else:
                            display_name = f"{base_display}_{{{suffix}}}"

                        # Alias ("mesh", fname) to the found field
                        self.alias(
                            ("mesh", fname), (ftype, fname), units=props[0]
                        )

                        if ("mesh", fname) in self:
                            self[("mesh", fname)].display_name = display_name
                        break

        # TODO: Yuxi: I do not know the purpose of the following function call.
        setup_magnetic_field_aliases(self, "FLEKS", ["B%s" % ax for ax in "xyz"])

    def setup_fluid_aliases(self):
        super(FLEKSFieldInfo, self).setup_fluid_aliases("mesh")

    def setup_particle_fields(self, ptype):
        super(FLEKSFieldInfo, self).setup_particle_fields(ptype)


class FLEKSHierarchy(BoxlibHierarchy):

    def __init__(self, ds, dataset_type="boxlib_native"):
        super(FLEKSHierarchy, self).__init__(ds, dataset_type)

        is_checkpoint = True
        for ptype in self.ds.particle_types:
            self._read_particles(ptype, is_checkpoint)

    def _detect_output_fields(self):
        super(FLEKSHierarchy, self)._detect_output_fields()

        # now detect the optional, non-cell-centered fields
        self.raw_file = self.ds.output_dir + "/raw_fields/"
        self.raw_fields = []
        self.field_list += [("raw", f) for f in self.raw_fields]
        self.raw_field_map = {}
        self.ds.nodal_flags = {}
        self.raw_field_nghost = {}


class YtFLEKSData(BoxlibDataset):
    """
    Read and plot the AMReX format output from FLEKS.

    Args:
        output_dir (str): the path to the data.

    Examples:
        >>> import flekspy
        >>> ds = flekspy.YtFLEKSData("3d_particle*n00004750_amrex")
    """

    _index_class = FLEKSHierarchy
    _field_info_class = FLEKSFieldInfo

    def __init__(
        self,
        output_dir,
        read_field_data=False,
        cparam_filename=None,
        fparam_filename=None,
        dataset_type="boxlib_native",
        storage_filename=None,
        units_override=None,
        unit_system="mks",
    ):
        self.default_fluid_type = "mesh"
        self.default_field = ("mesh", "density")
        self.fluid_types = ("mesh", "index", "raw")
        self.read_field_data = read_field_data

        super(YtFLEKSData, self).__init__(
            output_dir,
            cparam_filename,
            fparam_filename,
            dataset_type,
            storage_filename,
            units_override,
            unit_system,
        )

    def _parse_parameter_file(self):
        super(YtFLEKSData, self)._parse_parameter_file()

        fleks_header = Path(self.output_dir) / "FLEKSHeader"
        with open(fleks_header, "r") as f:
            plot_string = f.readline().lower()
            self.radius = float(f.readline())  # should be in unit [m]

        # It seems the second argument should be in the unit of [cm].
        self.unit_registry.add(
            "Planet_Radius", 100 * self.radius, yt.units.dimensions.length
        )

        if plot_string.find("si") != -1:
            self.parameters["fleks_unit"] = "si"
        elif plot_string.find("planet") != -1:
            self.parameters["fleks_unit"] = "planet"
        elif plot_string.find("pic") != -1:
            self.parameters["fleks_unit"] = "pic"
        else:
            self.parameters["fleks_unit"] = "unknown"

        output_dir_path = Path(self.output_dir)
        header_paths_generator = output_dir_path.glob("*/Header")
        particle_types = [p.parent.name for p in header_paths_generator]

        if len(particle_types) > 0 and not self.read_field_data:
            self.parameters["particles"] = 1
            self.particle_types = tuple(particle_types)
            self.particle_types_raw = self.particle_types
        else:
            self.particle_types = ()
            self.particle_types_raw = ()

    def _set_code_unit_attributes(self):
        unit = self.parameters["fleks_unit"]

        setdefaultattr(self, "time_unit", self.quan(1, get_unit("time", unit)))
        setdefaultattr(self, "length_unit", self.quan(1, get_unit("X", unit)))
        setdefaultattr(self, "mass_unit", self.quan(1, get_unit("mass", unit)))
        setdefaultattr(self, "velocity_unit", self.quan(1, get_unit("u", unit)))
        setdefaultattr(self, "magnetic_unit", self.quan(1, get_unit("B", unit)))
        setdefaultattr(self, "density_unit", self.quan(1, get_unit("rho", unit)))
        setdefaultattr(self, "pressure_unit", self.quan(1, get_unit("p", unit)))

    def pvar(self, var):
        return (self.particle_types[0], var)

    def get_slice(self, norm: str, cut_loc: float) -> xr.Dataset:
        r"""
        Returns a xarray.Dataset object that contains a slice along the normal direction.

        Args:
            norm (str): slice normal direction in "x", "y" or "z"
            cut_loc (float): cut location along the normal direction
        """

        axDir = {"X": 0, "Y": 1, "Z": 2}
        idir = axDir[norm.upper()]

        if not isinstance(cut_loc, yt.units.yt_array.YTArray):
            cut_loc = self.arr(cut_loc, "code_length")

        # Define the slice range
        slice_dim = self.domain_dimensions.copy()
        slice_dim[idir] = 1

        left_edge = self.domain_left_edge.copy()
        right_edge = self.domain_right_edge.copy()

        dd = (right_edge[idir] - left_edge[idir]) * 1e-6
        left_edge[idir] = cut_loc - dd
        right_edge[idir] = cut_loc + dd

        abArr = self.arbitrary_grid(left_edge, right_edge, slice_dim)

        data_vars = {}
        for var in self.field_list:
            if abArr[var].size != 0:  # remove empty sliced particle output
                data_vars[var[1]] = np.squeeze(abArr[var])

        axLabes = {0: ("Y", "Z"), 1: ("X", "Z"), 2: ("X", "Y")}
        ax_names = axLabes[idir]

        coords = {}
        for axis_label in ax_names:
            ax_dir = axDir[axis_label]
            coords[axis_label] = np.linspace(
                self.domain_left_edge[ax_dir],
                self.domain_right_edge[ax_dir],
                self.domain_dimensions[ax_dir],
            )

        # Process grid data (2D) and particle data (1D) separately
        grid_vars = {}
        particle_vars = {}

        for key, value in data_vars.items():
            if value.ndim == 2:
                grid_vars[key] = value
            elif value.ndim == 1:
                particle_vars[key] = (("particle_index",), value)

        # The arbitrary_grid returns data in (X, Y) order, so no transpose is needed.
        ds_grid = xr.Dataset(
            {key: (ax_names, value) for key, value in grid_vars.items()},
            coords=coords,
        )

        ds_particle = xr.Dataset(particle_vars)

        # Merge datasets and set particle positions as coordinates
        ds = xr.merge([ds_grid, ds_particle])
        particle_coords = [
            "particle_position_x",
            "particle_position_y",
            "particle_position_z",
        ]
        for coord_name in particle_coords:
            if coord_name in ds:
                ds = ds.set_coords(coord_name)

        ds.attrs["cut_norm"] = norm
        ds.attrs["cut_loc"] = cut_loc
        ds.attrs["time"] = self.current_time
        ds.attrs["nstep"] = self.parameters.get("nstep", -1)
        ds.attrs["filename"] = self.output_dir
        return ds

    def get_domain(self) -> xr.Dataset:
        """
        Read all the simulation data into a 3D box.
        This method only reads grid data, not particle data.
        """
        domain = self.covering_grid(
            level=0, left_edge=self.domain_left_edge, dims=self.domain_dimensions
        )

        ax_names = ["X", "Y", "Z"]
        dims = ax_names[: self.dimensionality]
        coords = {
            ax_names[idim]: np.linspace(
                self.domain_left_edge[idim],
                self.domain_right_edge[idim],
                self.domain_dimensions[idim],
            )
            for idim in range(self.dimensionality)
        }

        data_vars = {}
        # Filter for non-particle fields as covering_grid does not support them
        grid_fields = [f for f in self.field_list if f[0] not in self.particle_types]

        for var_tuple in grid_fields:
            var_name = var_tuple[1]
            value = domain[var_tuple]
            data_vars[var_name] = (dims, value)

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["time"] = self.current_time
        ds.attrs["nstep"] = self.parameters.get("nstep", -1)
        ds.attrs["filename"] = self.output_dir
        return ds

    def _get_profile(
        self,
        x_field,
        y_field,
        z_field,
        region: YTSelectionContainer | None = None,
        x_bins: int = 128,
        y_bins: int = 128,
        domain_size: tuple | None = None,
    ):
        if region is None:
            region = self.box(self.domain_left_edge, self.domain_right_edge)

        # The bins should be uniform instead of logarithmic
        logs = {self.pvar(x_field): False, self.pvar(y_field): False}

        bin_fields = [self.pvar(x_field), self.pvar(y_field)]
        if domain_size is not None:
            extrema = {
                self.pvar(x_field): (domain_size[0], domain_size[1]),
                self.pvar(y_field): (domain_size[2], domain_size[3]),
            }
        else:
            extrema = None
        profile = yt.create_profile(
            data_source=region,
            bin_fields=bin_fields,
            fields=self.pvar(z_field),
            n_bins=[x_bins, y_bins],
            weight_field=None,
            extrema=extrema,
            logs=logs,
        )

        return profile

    def get_phase(
        self,
        x_field,
        y_field,
        z_field,
        region: YTSelectionContainer | None = None,
        x_bins: int = 128,
        y_bins: int = 128,
        domain_size: tuple | None = None,
    ):
        """Get particle phase space distribution.

        Args:
            region: YTSelectionContainer
                Spatial region to be selected, such as all_data, box, region, or sphere.

            x_field & y_field: string
                The x-/y- axes, from "p_ux", "p_uy", "p_uz", "p_x", "p_y" or "p_z".

            z_field: string
                It is usually the particle weight: "p_w".

            domain_size: tuple
                Axis range of 4 elements: x_min, x_max, y_min, y_max

        Examples:
            >>> x, y, w = ds.get_phase("p_ux", "p_uy", "p_w", domain_size=(-1, 1, -1, 1))
        """
        profile = self._get_profile(
            x_field,
            y_field,
            z_field,
            region=region,
            x_bins=x_bins,
            y_bins=y_bins,
            domain_size=domain_size,
        )

        return (
            profile.x.ndarray_view(),
            profile.y.ndarray_view(),
            profile[self.pvar(z_field)].ndarray_view(),
        )

    def plot_phase(
        self,
        x_field,
        y_field,
        z_field,
        region: YTSelectionContainer | None = None,
        unit_type: str = "planet",
        x_bins: int = 128,
        y_bins: int = 128,
        domain_size: tuple | None = None,
        font_size: float = 18,
        figure_size: float = 8,
        customized: bool = False,
    ) -> PhasePlot:
        """Plot particle phase space distribution.

        Args:
            region: YTSelectionContainer
                Spatial region to be selected, such as all_data, box, region, or sphere.

            x_field & y_field: string
                The x-/y- axes, from "p_ux", "p_uy", "p_uz", "p_x", "p_y" or "p_z".

            z_field: string
                It is usually the particle weight: "p_w".

            unit_type: string
                The unit system of the plots. "planet" or "si".

            domain_size: tuple
                Axis range of 4 elements: x_min, x_max, y_min, y_max

        Examples:
            >>> pp = ds.plot_phase("p_ux", "p_uy", "p_w", domain_size=(-1, 1, -1, 1))
            >>> pp.show()
        """
        profile = self._get_profile(
            x_field,
            y_field,
            z_field,
            region=region,
            x_bins=x_bins,
            y_bins=y_bins,
            domain_size=domain_size,
        )

        pp = yt.PhasePlot.from_profile(
            profile, fontsize=font_size, figure_size=figure_size
        )
        pp.set_log(pp.fields[0], False)
        pp.set_unit(self.pvar(x_field), get_unit(x_field, unit_type))
        pp.set_unit(self.pvar(y_field), get_unit(y_field, unit_type))

        if customized:
            pp.set_cmap(pp.fields[0], "turbo")
            pp.set_font(
                {
                    "family": "DejaVu Sans",
                }
            )
            pp.render()
            ax = pp.plots[pp.fields[0]].axes
            ax.tick_params(length=10, width=3)
            ax.tick_params(which="minor", length=5, width=3, color="tab:gray")
            vl = ax.axvline(0, color="w", lw=2, linestyle="--", zorder=200)
            hl = ax.axhline(0, color="w", lw=2, linestyle="--", zorder=200)
            ax.add_line(vl)
            ax.add_line(hl)
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(2)

        return pp

    def plot_particles(
        self,
        x_field,
        y_field,
        z_field,
        region: YTSelectionContainer | None = None,
        unit_type: str = "planet",
        x_bins: int = 128,
        y_bins: int = 128,
        **kwargs,
    ):
        r"""Plot the particle position of particles inside a box.

        Args:
            x_field & y_field: str
                The x- y- axes, from "p_x", "p_y", "p_z".

            z_field: str
                color variable, usually the particle weight "p_w".

            region: YTSelectionContainer
                Spatial region to be selected, such as all_data, box, region, or sphere.

            unit_type: str
                The unit system of the plots. "planet" or "si".

        See more at https://yt-project.org/doc/reference/api/yt.visualization.particle_plots.html#yt.visualization.particle_plots.ParticlePlot

        Examples:
            >>> pp = ds.plot_particles([8, -1, -1], [10, 0, 0], "p_x", "p_y", "p_w", unit_type="planet")
            >>> pp.show()
        """
        if region is None:
            region = self.box(self.domain_left_edge, self.domain_right_edge)

        nmap = {
            "p_x": "particle_position_x",
            "p_y": "particle_position_y",
            "p_z": "particle_position_z",
        }
        pp = yt.ParticlePlot(
            self,
            self.pvar(nmap[x_field]),
            self.pvar(nmap[y_field]),
            self.pvar(z_field),
            data_source=region,
            **kwargs,
        )
        pp.set_axes_unit((get_unit(x_field, unit_type), get_unit(y_field, unit_type)))

        return pp


def extract_phase(pp: PhasePlot):
    """Extract phase space distribution from PhasePlot object.

    Args:
        pp (PhasePlot): YT PhasePlot object.

    Returns:
        x, y, f: Horizontal, vertical coordinates of the plot, and phase space density.
    """
    f = pp.profile.field_data[pp.fields[0]]
    x = pp.profile.x
    y = pp.profile.y

    return (x, y, f)
