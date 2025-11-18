import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Optional, Union, Type, Callable
from scipy import constants as const

from .plotting import AMReXPlottingMixin


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


class AMReXParticleData(AMReXPlottingMixin):
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

    def fit_gmm(
        self,
        n_components: int,
        x_variable: str,
        y_variable: str,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        transform: Optional[
            Callable[[np.ndarray], Tuple[np.ndarray, List[str]]]
        ] = None,
    ) -> "GaussianMixture":
        """
        Fits a Gaussian Mixture Model (GMM) to the phase space distribution.

        Args:
            n_components (int): The number of mixture components (Gaussian distributions).
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            x_range (tuple, optional): A tuple (min, max) for the x-axis boundary.
            y_range (tuple, optional): A tuple (min, max) for the y-axis boundary.
            z_range (tuple, optional): A tuple (min, max) for the z-axis boundary.
                                       For 2D data, this is ignored.
            transform (callable, optional):
                A function that takes the particle data (`rdata`, a NumPy array)
                and returns a tuple: (`transformed_rdata`, `new_component_names`).
                This allows for fitting derived quantities or changing coordinate systems.
                If provided, `x_variable` and `y_variable` should refer to names
                in `new_component_names`. Defaults to None.

        Returns:
            sklearn.mixture.GaussianMixture: The fitted GMM model.
        """
        # --- 1. Select data ---
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            raise ValueError("No particles to fit GMM.")

        # --- 2. Apply transformation if provided ---
        component_names = self.header.real_component_names
        if transform:
            rdata, component_names = transform(rdata)

        # --- 3. Map component names to column indices ---
        component_map = {name: i for i, name in enumerate(component_names)}

        # --- 4. Validate input variable names ---
        x_variable = self._resolve_alias(x_variable)
        y_variable = self._resolve_alias(y_variable)
        if x_variable not in component_map or y_variable not in component_map:
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]

        # --- 5. Extract the relevant data columns ---
        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]

        data = np.vstack([x_data, y_data]).T

        from sklearn.mixture import GaussianMixture

        # --- 5. Fit GMM ---
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)

        return gmm

    @staticmethod
    def get_gmm_temperatures(
        gmm: "GaussianMixture", particle_mass: float = 1.0, isotropic: bool = True
    ) -> List[dict]:
        """
        Extracts physical temperatures from a fitted GMM.

        This method calls `get_gmm_parameters` to get the squared thermal
        velocities and then converts them to temperatures in Kelvin.

        Args:
            gmm ("GaussianMixture"): The fitted GMM model.
            particle_mass (float): The mass of the particle species in atomic mass units (amu).
                                   Defaults to 1.0.
            isotropic (bool, optional): If True, assumes an isotropic Maxwellian
                                        distribution and returns a single scalar temperature.
                                        If False, assumes a Bi-Maxwellian distribution and returns
                                        parallel and perpendicular temperatures. Defaults to True.

        Returns:
            list of dict: A list of dictionaries, one for each Gaussian component.
                          - Isotropic: {'center': [mx, my], 'temperature': T}
                          - Bi-Maxwellian: {'center': [mx, my], 'T_parallel': T_par, 'T_perpendicular': T_perp}
        """
        from ..util.gmm import get_gmm_parameters

        parameters = get_gmm_parameters(gmm, isotropic=isotropic)
        mass_in_kg = particle_mass * const.m_u

        if isotropic:
            return [
                {
                    "center": p["center"],
                    "temperature": mass_in_kg * p["v_th_sq"] / const.k,
                }
                for p in parameters
            ]
        else:
            return [
                {
                    "center": p["center"],
                    "T_parallel": mass_in_kg * p["v_parallel_sq"] / const.k,
                    "T_perpendicular": mass_in_kg * p["v_perp_sq"] / const.k,
                }
                for p in parameters
            ]
