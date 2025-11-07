import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Type


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
