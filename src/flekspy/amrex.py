import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Optional, Any, Union, Type
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class AMReXParticleHeader:
    """
    This class is designed to parse and store the information
    contained in an AMReX particle header file.
    """

    version_string: str
    real_type: Union[Type[np.float64], Type[np.float32]]
    int_type: Type[np.int32]
    dim: int
    num_int_base: int
    num_real_base: int
    real_component_names: List[str]
    int_component_names: List[str]
    num_real_extra: int
    num_int_extra: int
    num_int: int
    num_real: int
    is_checkpoint: bool
    num_particles: int
    max_next_id: int
    finest_level: int
    num_levels: int
    grids_per_level: np.ndarray
    grids: List[List[Tuple[int, ...]]]

    def __init__(self, header_filename: Union[str, Path]):

        self.real_component_names = []
        self.int_component_names = []
        with open(header_filename, "r") as f:
            self.version_string = f.readline().strip()

            particle_real_type = self.version_string.split("_")[-1]
            if particle_real_type == "double":
                self.real_type = np.float64
            elif particle_real_type == "single":
                self.real_type = np.float32
            else:
                raise RuntimeError("Did not recognize particle real type.")
            self.int_type = np.int32

            self.dim = int(f.readline().strip())
            self.num_int_base = 2
            self.num_real_base = self.dim

            if self.dim == 3:
                self.real_component_names = ["x", "y", "z"]
            elif self.dim == 2:
                self.real_component_names = ["x", "y"]

            self.int_component_names = ["particle_id", "particle_cpu"]

            self.num_real_extra = int(f.readline().strip())
            for i in range(self.num_real_extra):
                self.real_component_names.append(f.readline().strip())
            self.num_int_extra = int(f.readline().strip())
            for i in range(self.num_int_extra):
                self.int_component_names.append(f.readline().strip())
            self.num_int = self.num_int_base + self.num_int_extra
            self.num_real = self.num_real_base + self.num_real_extra
            self.is_checkpoint = bool(int(f.readline().strip()))
            self.num_particles = int(f.readline().strip())
            self.max_next_id = int(f.readline().strip())
            self.finest_level = int(f.readline().strip())
            self.num_levels = self.finest_level + 1

            if not self.is_checkpoint:
                self.num_int_base = 0
                self.num_int_extra = 0
                self.num_int = 0

            self.grids_per_level = np.zeros(self.num_levels, dtype="int64")
            for level_num in range(self.num_levels):
                self.grids_per_level[level_num] = int(f.readline().strip())

            self.grids = [[] for _ in range(self.num_levels)]
            for level_num in range(self.num_levels):
                for grid_num in range(self.grids_per_level[level_num]):
                    entry = [int(val) for val in f.readline().strip().split()]
                    self.grids[level_num].append(tuple(entry))

    def __repr__(self) -> str:
        """
        Returns a string representation of the header contents.
        """
        level_info = "\n".join(
            [
                f"  Level {level_num}: {self.grids_per_level[level_num]} grids"
                for level_num in range(self.num_levels)
            ]
        )
        return (
            f"Version string: {self.version_string}\n"
            f"Dimensions: {self.dim}\n"
            f"Number of integer components: {self.num_int}\n"
            f"Integer component names: {self.int_component_names}\n"
            f"Number of real components: {self.num_real}\n"
            f"Real component names: {self.real_component_names}\n"
            f"Is checkpoint: {self.is_checkpoint}\n"
            f"Number of particles: {self.num_particles}\n"
            f"Max next ID: {self.max_next_id}\n"
            f"Finest level: {self.finest_level}\n"
            f"Number of levels: {self.num_levels}\n"
            f"{level_info}"
        )

    @property
    def idtype_str(self) -> str:
        return f"({self.num_int},)i4"

    @property
    def rdtype_str(self) -> str:
        if self.real_type == np.float64:
            return f"({self.num_real},)f8"
        elif self.real_type == np.float32:
            return f"({self.num_real},)f4"
        raise RuntimeError("Unrecognized real type.")


def read_amrex_binary_particle_file(
    fn: Union[str, Path], header: AMReXParticleHeader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns the particle data stored in a particular
    plot file. It returns two numpy arrays, the
    first containing the particle integer data, and the second the
    particle real data.
    """
    ptype = "particles"
    base_fn = Path(fn) / ptype

    idtype = header.idtype_str
    fdtype = header.rdtype_str

    idata = np.empty((header.num_particles, header.num_int), dtype=header.int_type)
    rdata = np.empty((header.num_particles, header.num_real), dtype=header.real_type)

    ip = 0
    for lvl, level_grids in enumerate(header.grids):
        for which, count, where in level_grids:
            if count == 0:
                continue
            fn = base_fn / f"Level_{lvl}" / f"DATA_{which:05d}"

            with open(fn, "rb") as f:
                f.seek(where)
                if header.is_checkpoint:
                    ints = np.fromfile(f, dtype=idtype, count=count)
                    idata[ip : ip + count] = ints

                floats = np.fromfile(f, dtype=fdtype, count=count)
                rdata[ip : ip + count] = floats
            ip += count

    return idata, rdata


class AMReXParticleData:
    """
    This class provides an interface to the particle data in a plotfile.
    Data is loaded lazily upon first access to `idata` or `rdata`.
    """

    output_dir: Path
    ptype: str
    _idata: Optional[np.ndarray]
    _rdata: Optional[np.ndarray]
    level_boxes: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
    header: AMReXParticleHeader
    dim: int
    time: float
    left_edge: List[float]
    right_edge: List[float]
    domain_dimensions: List[int]

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.ptype = "particles"

        self._idata = None
        self._rdata = None

        self.level_boxes = []

        self._parse_main_header()
        self.header = AMReXParticleHeader(self.output_dir / self.ptype / "Header")
        self._parse_particle_h_files()

    def _load_data(self) -> None:
        """Loads the particle data from disk if it has not been loaded yet."""
        if self._idata is None:
            self._idata, self._rdata = read_amrex_binary_particle_file(
                self.output_dir, self.header
            )

    @property
    def idata(self) -> np.ndarray:
        """Lazily loads and returns the integer particle data."""
        self._load_data()
        assert self._idata is not None
        return self._idata

    @property
    def rdata(self) -> np.ndarray:
        """Lazily loads and returns the real particle data."""
        self._load_data()
        assert self._rdata is not None
        return self._rdata

    def _parse_main_header(self) -> None:
        header_path = self.output_dir / "Header"
        with open(header_path, "r") as f:
            f.readline()  # version string
            num_fields = int(f.readline())
            # skip field names
            for _ in range(num_fields):
                f.readline()

            self.dim = int(f.readline())
            self.time = float(f.readline())
            f.readline()  # prob_refine_ratio

            self.left_edge = [float(v) for v in f.readline().strip().split()]
            self.right_edge = [float(v) for v in f.readline().strip().split()]
            f.readline()
            # TODO check a 3D particle file for correctness!
            dim_line = f.readline().strip()
            matches = re.findall(r"\d+", dim_line)
            coords = [int(num) for num in matches]
            x1, y1, x2, y2, z1, z2 = coords
            dim_x = x2 - x1 + 1
            dim_y = y2 - y1 + 1
            dim_z = z2 - z1 + 1

            self.domain_dimensions = [dim_x, dim_y, dim_z]

    def _parse_particle_h_files(self) -> None:
        """Parses the Particle_H files to get the box arrays for each level."""
        self.level_boxes = [[] for _ in range(self.header.num_levels)]
        for level_num in range(self.header.num_levels):
            particle_h_path = (
                self.output_dir / self.ptype / f"Level_{level_num}" / "Particle_H"
            )
            if not particle_h_path.exists():
                continue

            with open(particle_h_path, "r") as f:
                lines = f.readlines()

            boxes = []
            # The first line is `(num_boxes level`, e.g. `(20 0`.
            # The rest of the lines are box definitions, e.g. `((0,0) (15,7) (0,0))`
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("((") and line.endswith("))"):
                    try:
                        parts = [int(x) for x in re.findall(r"-?\d+", line)]
                        if self.header.dim == 2 and len(parts) >= 4:
                            lo_corner = (parts[0], parts[1])
                            hi_corner = (parts[2], parts[3])
                            boxes.append((lo_corner, hi_corner))
                        elif self.header.dim == 3 and len(parts) >= 6:
                            lo_corner = (parts[0], parts[1], parts[2])
                            hi_corner = (parts[3], parts[4], parts[5])
                            boxes.append((lo_corner, hi_corner))
                    except (ValueError, IndexError):
                        continue  # Not a valid box line
            self.level_boxes[level_num] = boxes

    def __repr__(self) -> str:
        repr_str = (
            f"AMReXParticleData from {self.output_dir}\n"
            f"Time: {self.time}\n"
            f"Dimensions: {self.dim}\n"
            f"Domain Dimensions: {self.domain_dimensions}\n"
            f"Domain Edges: {self.left_edge} to {self.right_edge}\n"
            f"Integer component names: {self.header.int_component_names}\n"
            f"Real component names: {self.header.real_component_names}"
        )
        if self._idata is not None:
            repr_str += (
                f"\nParticle data shape (int): {self._idata.shape}\n"
                f"Particle data shape (real): {self._rdata.shape}"
            )
        else:
            repr_str += "\nParticle data: Not loaded (access .idata or .rdata to load)"
        return repr_str

    def select_particles_in_region(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Selectively loads real component data for particles that fall within a
        specified rectangular region.

        This method first converts the physical range into an index-based range,
        then identifies which grid files intersect with that range, and finally
        reads only the necessary data. This avoids loading the entire dataset
        into memory. Integer data is skipped for efficiency.

        Args:
            x_range (tuple, optional): A tuple (min, max) for the x-axis boundary.
            y_range (tuple, optional): A tuple (min, max) for the y-axis boundary.
            z_range (tuple, optional): A tuple (min, max) for the z-axis boundary.
                                       For 2D data, this is ignored.

        Returns:
            np.ndarray: A numpy array containing the real data for the
                        selected particles.
        """

        # Convert physical range to index range
        dx = [
            (self.right_edge[i] - self.left_edge[i]) / self.domain_dimensions[i]
            for i in range(self.dim)
        ]

        target_idx_ranges: List[Optional[Tuple[int, int]]] = []
        ranges = [x_range, y_range, z_range]
        for i in range(self.dim):
            if ranges[i]:
                idx_min = int((ranges[i][0] - self.left_edge[i]) / dx[i])
                idx_max = int((ranges[i][1] - self.left_edge[i]) / dx[i])
                target_idx_ranges.append((idx_min, idx_max))
            else:
                target_idx_ranges.append(None)

        # Find overlapping grids based on index ranges
        overlapping_grids: List[Tuple[int, int]] = []
        for level_num, boxes in enumerate(self.level_boxes):
            for grid_index, (lo_corner, hi_corner) in enumerate(boxes):
                box_overlap = True
                for i in range(self.dim):
                    if target_idx_ranges[i]:
                        box_min_idx, box_max_idx = lo_corner[i], hi_corner[i]
                        target_min_idx, target_max_idx = target_idx_ranges[i]
                        if box_max_idx < target_min_idx or box_min_idx > target_max_idx:
                            box_overlap = False
                            break
                if box_overlap:
                    overlapping_grids.append((level_num, grid_index))

        selected_rdata: List[np.ndarray] = []
        idtype = self.header.idtype_str
        fdtype = self.header.rdtype_str

        for level_num, grid_index in overlapping_grids:
            try:
                grid_data = self.header.grids[level_num][grid_index]
            except IndexError:
                continue

            which, count, where = grid_data
            if count == 0:
                continue

            fn = (
                self.output_dir
                / self.ptype
                / f"Level_{level_num}"
                / f"DATA_{which:05d}"
            )
            with open(fn, "rb") as f:
                f.seek(where)

                if self.header.is_checkpoint:
                    bytes_to_skip = count * np.dtype(idtype).itemsize
                    f.seek(bytes_to_skip, 1)

                floats = np.fromfile(f, dtype=fdtype, count=count)

                mask = np.ones(count, dtype=bool)
                for i in range(self.dim):
                    if ranges[i]:
                        mask &= (floats[:, i] >= ranges[i][0]) & (
                            floats[:, i] <= ranges[i][1]
                        )

                if np.any(mask):
                    selected_rdata.append(floats[mask])

        final_rdata = (
            np.concatenate(selected_rdata)
            if selected_rdata
            else np.empty((0, self.header.num_real), dtype=self.header.real_type)
        )
        return final_rdata

    def plot_phase(
        self,
        x_variable: str,
        y_variable: str,
        bins: Union[int, Tuple[int, int]] = 100,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **imshow_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 2D phase space distribution for any two selected variables.

        This function creates a 2D weighted histogram to visualize the particle
        density. If a 'weight' component is present in the data, it will be
        used for the histogram weighting. Otherwise, a standard (unweighted)
        histogram is generated.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            bins (int or tuple, optional): The number of bins. This can be a
                                           single integer for the same number of
                                           bins in each dimension, or a two-element
                                           tuple for different numbers of bins in the
                                           x and y dimension, respectively.
                                           Defaults to 100.
            hist_range (list of lists, optional): The leftmost and rightmost edges of the
                                             bins along each dimension. It should be
                                             in the format [[xmin, xmax], [ymin, ymax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for the x-axis boundary.
            y_range (tuple, optional): A tuple (min, max) for the y-axis boundary.
            z_range (tuple, optional): A tuple (min, max) for the z-axis boundary.
                                       For 2D data, this is ignored.
            normalize (bool, optional): If True, the histogram is normalized to
                                        form a probability density. Defaults to False.
            title (str, optional): The title for the plot. Defaults to "Phase Space Distribution".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            **imshow_kwargs: Additional keyword arguments to be passed to `ax.imshow()`.
                             This can be used to control colormaps (`cmap`), normalization (`norm`), etc.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
                   This allows for further customization of the plot after its creation.
        """
        # --- 1. Select data ---
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            logger.warning("No particles to plot.")
            return None

        # --- 2. Map component names to column indices ---
        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        # --- 3. Validate input variable names ---
        if x_variable not in component_map or y_variable not in component_map:
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]

        # --- 4. Extract the relevant data columns ---
        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]

        # --- 5. Create the 2D histogram ---
        weights = None
        if "weight" in component_map:
            weight_index = component_map["weight"]
            weights = rdata[:, weight_index]
            cbar_label = "Weighted Particle Density"
        else:
            cbar_label = "Particle Count"

        H, xedges, yedges = np.histogram2d(
            x_data, y_data, bins=bins, range=hist_range, weights=weights
        )

        if normalize:
            total = H.sum()
            if total > 0:
                H /= total
            if weights is not None:
                cbar_label = "Normalized Weighted Density"
            else:
                cbar_label = "Normalized Density"

        # --- 6. Plot the resulting histogram as a color map ---
        fig, ax = plt.subplots(figsize=(8, 6))

        # Default imshow settings that can be overridden by user
        imshow_settings = {
            "interpolation": "nearest",
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
        }
        imshow_settings.update(imshow_kwargs)

        im = ax.imshow(H.T, **imshow_settings)

        # --- 7. Add labels and a color bar for context ---
        final_title = title if title is not None else "Phase Space Distribution"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.minorticks_on()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

        # --- 8. Return the plot objects ---
        return fig, ax

    def plot_phase_3d(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        bins: Union[int, Tuple[int, int, int]] = 50,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **scatter_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 3D phase space distribution for any three selected variables.

        This function creates a 3D histogram and visualizes it as a scatter plot,
        where the color of each point corresponds to the particle density in that bin.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            z_variable (str): The name of the variable for the z-axis.
            bins (int or tuple, optional): The number of bins for each dimension.
                                           Defaults to 50.
            hist_range (list of lists, optional): The range for the bins in the format
                                             [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for filtering particles by x-position.
            y_range (tuple, optional): A tuple (min, max) for filtering particles by y-position.
            z_range (tuple, optional): A tuple (min, max) for filtering particles by z-position.
            normalize (bool, optional): If True, normalize the histogram to form a
                                        probability density. Defaults to False.
            title (str, optional): The title for the plot. Defaults to "3D Phase Space Distribution".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            **scatter_kwargs: Additional keyword arguments to be passed to `ax.scatter()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
        """
        # --- 1. Select data ---
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            logger.warning("No particles to plot.")
            return None

        # --- 2. Map component names to column indices ---
        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        # --- 3. Validate input variable names ---
        if (
            x_variable not in component_map
            or y_variable not in component_map
            or z_variable not in component_map
        ):
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]
        z_index = component_map[z_variable]

        # --- 4. Extract the relevant data columns ---
        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]
        z_data = rdata[:, z_index]
        sample = np.vstack([x_data, y_data, z_data]).T

        # --- 5. Create the 3D histogram ---
        weights = None
        if "weight" in component_map:
            weight_index = component_map["weight"]
            weights = rdata[:, weight_index]
            cbar_label = "Weighted Particle Density"
        else:
            cbar_label = "Particle Count"

        H, edges = np.histogramdd(sample, bins=bins, range=hist_range, weights=weights)

        if normalize:
            total = H.sum()
            if total > 0:
                H /= total
            if weights is not None:
                cbar_label = "Normalized Weighted Density"
            else:
                cbar_label = "Normalized Density"

        # --- 6. Prepare data for scatter plot ---
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2

        # Create a meshgrid of bin centers
        x_grid, y_grid, z_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

        # Flatten the arrays for scatter plot
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        density = H.flatten()

        # Filter out empty bins
        non_empty = density > 0
        x_flat = x_flat[non_empty]
        y_flat = y_flat[non_empty]
        z_flat = z_flat[non_empty]
        density = density[non_empty]


        # --- 7. Plot the resulting histogram as a 3D scatter plot ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter_settings = {
            "c": density,
            "cmap": "viridis",
            "s": 20, # a default size
        }
        scatter_settings.update(scatter_kwargs)


        sc = ax.scatter(x_flat, y_flat, z_flat, **scatter_settings)

        # --- 8. Add labels and a color bar ---
        final_title = title if title is not None else "3D Phase Space Distribution"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.set_zlabel(final_zlabel, fontsize="x-large")

        cbar = fig.colorbar(sc)
        cbar.set_label(cbar_label)

        # --- 9. Return the plot objects ---
        return fig, ax

    @staticmethod
    def _plot_plane_quadrants(ax, H, edges, fixed_coord, cmap, **surface_kwargs):
        """Helper function to plot a single plane divided into four quadrants."""
        nx, ny, nz = H.shape
        x_edges, y_edges, z_edges = edges

        min_val, max_val = H.min(), H.max()
        cmap = plt.get_cmap(cmap)

        slice_index = {"x": nx // 2, "y": ny // 2, "z": nz // 2}[fixed_coord]
        plane_idx = {
            "x": (slice_index, slice(None), slice(None)),
            "y": (slice(None), slice_index, slice(None)),
            "z": (slice(None), slice(None), slice_index),
        }[fixed_coord]
        plane_data = H[plane_idx]

        coord_edges = {
            "x": (y_edges, z_edges),
            "y": (x_edges, z_edges),
            "z": (x_edges, y_edges),
        }[fixed_coord]
        n0, n1 = plane_data.shape
        edges0, edges1 = coord_edges

        centers0 = (edges0[:-1] + edges0[1:]) / 2
        centers1 = (edges1[:-1] + edges1[1:]) / 2

        quadrant_data = [
            plane_data[: n0 // 2, : n1 // 2],
            plane_data[: n0 // 2, n1 // 2 :],
            plane_data[n0 // 2 :, : n1 // 2],
            plane_data[n0 // 2 :, n1 // 2 :],
        ]
        quadrant_centers0 = [
            centers0[: n0 // 2],
            centers0[: n0 // 2],
            centers0[n0 // 2 :],
            centers0[n0 // 2 :],
        ]
        quadrant_centers1 = [
            centers1[: n1 // 2],
            centers1[n1 // 2 :],
            centers1[: n1 // 2],
            centers1[n1 // 2 :],
        ]

        for i in range(4):
            q_data = quadrant_data[i]
            q_centers0 = quadrant_centers0[i]
            q_centers1 = quadrant_centers1[i]

            if q_data.size == 0:
                continue

            denominator = max_val - min_val
            if denominator == 0:
                # All values are the same, so assign a single color.
                normalized_data = np.full(q_data.shape, 0.5)
            else:
                normalized_data = (q_data - min_val) / denominator
            facecolors = cmap(normalized_data)


            C0, C1 = np.meshgrid(q_centers0, q_centers1, indexing="ij")

            if fixed_coord == "x":
                X = np.full_like(C0, (x_edges[nx // 2] + x_edges[nx // 2 + 1]) / 2)
                Y, Z = C0, C1
            elif fixed_coord == "y":
                Y = np.full_like(C0, (y_edges[ny // 2] + y_edges[ny // 2 + 1]) / 2)
                X, Z = C0, C1
            else:  # fixed_coord == 'z'
                Z = np.full_like(C0, (z_edges[nz // 2] + z_edges[nz // 2 + 1]) / 2)
                X, Y = C0, C1
            ax.plot_surface(
                X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, shade=False, **surface_kwargs
            )

    def plot_intersecting_planes(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        bins: Union[int, Tuple[int, int, int]] = 50,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        cmap: str = "viridis",
        **surface_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 3D phase space distribution using three intersecting planes.

        This function creates a 3D histogram and visualizes the density on three
        orthogonal planes that intersect at the center of the histogrammed data.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            z_variable (str): The name of the variable for the z-axis.
            bins (int or tuple, optional): The number of bins for each dimension.
                                           Defaults to 50.
            hist_range (list of lists, optional): The range for the bins in the format
                                             [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for filtering particles by x-position.
            y_range (tuple, optional): A tuple (min, max) for filtering particles by y-position.
            z_range (tuple, optional): A tuple (min, max) for filtering particles by z-position.
            normalize (bool, optional): If True, normalize the histogram to form a
                                        probability density. Defaults to False.
            title (str, optional): The title for the plot. Defaults to "Intersecting Planes of Phase Space".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            cmap (str, optional): The colormap to use for the planes. Defaults to "viridis".
            **surface_kwargs: Additional keyword arguments to be passed to `ax.plot_surface()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
        """
        # --- 1. Select data ---
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            logger.warning("No particles to plot.")
            return None

        # --- 2. Map component names to column indices ---
        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        # --- 3. Validate input variable names ---
        if (
            x_variable not in component_map
            or y_variable not in component_map
            or z_variable not in component_map
        ):
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]
        z_index = component_map[z_variable]

        # --- 4. Extract the relevant data columns ---
        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]
        z_data = rdata[:, z_index]
        sample = np.vstack([x_data, y_data, z_data]).T

        # --- 5. Create the 3D histogram ---
        weights = None
        cbar_label = "Particle Count"
        if "weight" in component_map:
            weight_index = component_map["weight"]
            weights = rdata[:, weight_index]
            cbar_label = "Weighted Particle Density"

        H, edges = np.histogramdd(sample, bins=bins, range=hist_range, weights=weights)

        if normalize:
            total = H.sum()
            if total > 0:
                H /= total
            if weights is not None:
                cbar_label = "Normalized Weighted Density"
            else:
                cbar_label = "Normalized Density"

        # --- 6. Plot the intersecting planes ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        self._plot_plane_quadrants(ax, H, edges, "x", cmap, **surface_kwargs)
        self._plot_plane_quadrants(ax, H, edges, "y", cmap, **surface_kwargs)
        self._plot_plane_quadrants(ax, H, edges, "z", cmap, **surface_kwargs)

        # --- 7. Add labels and title ---
        final_title = title if title is not None else "Intersecting Planes of Phase Space"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.set_zlabel(final_zlabel, fontsize="x-large")

        # --- 8. Add a colorbar ---
        norm = plt.Normalize(vmin=H.min(), vmax=H.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for the mappable
        fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1, label=cbar_label)


        # --- 9. Return the plot objects ---
        return fig, ax
