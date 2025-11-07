import numpy as np
from pathlib import Path
import re
import logging
from typing import List, Tuple, Optional, Union

from .header import AMReXParticleHeader
from .loader import read_amrex_binary_particle_file
from . import plotting

logger = logging.getLogger(__name__)


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

    plot_phase = plotting.plot_phase
    plot_phase_subplots = plotting.plot_phase_subplots
    _prepare_3d_histogram_data = plotting._prepare_3d_histogram_data
    plot_phase_3d = plotting.plot_phase_3d
    _plot_plane = plotting._plot_plane
    plot_intersecting_planes = plotting.plot_intersecting_planes

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
