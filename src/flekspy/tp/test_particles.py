from typing import List, Tuple, Dict, Union, Callable

from flekspy.util.logger import get_logger
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py

logger = get_logger(name=__name__)
from matplotlib.colors import Normalize, LogNorm
from pathlib import Path
import numpy as np
import polars as pl
import glob
import struct
from itertools import islice
from enum import IntEnum
from scipy.constants import proton_mass, elementary_charge, mu_0, epsilon_0

EARTH_RADIUS_KM = 6378


class Indices(IntEnum):
    """Defines constant indices for test particles."""

    TIME = 0
    X = 1
    Y = 2
    Z = 3
    VX = 4
    VY = 5
    VZ = 6
    BX = 7
    BY = 8
    BZ = 9
    EX = 10
    EY = 11
    EZ = 12
    DBXDX = 13
    DBXDY = 14
    DBXDZ = 15
    DBYDX = 16
    DBYDY = 17
    DBYDZ = 18
    DBZDX = 19
    DBZDY = 20
    DBZDZ = 21


class FLEKSTP(object):
    """
    A class that is used to read and plot test particles. Each particle ID consists of
    a CPU index, a particle index on each CPU, and a location index.
    By default, 7 real numbers saved for each step: time + position + velocity.
    Additional field information are also stored if available.

    This class is a lazy, iterable container. It avoids loading all data into memory
    at once, making it efficient for large datasets. You can access particle
    trajectories using standard container operations.

    Args:
        dirs (str): the path to the test particle dataset.

    Examples:
    >>> tp = FLEKSTP("res/run1/PC/test_particles", iSpecies=1)
    >>> len(tp)
    10240
    >>> trajectory = tp[0]
    >>> tp.plot_trajectory(tp.IDs[3])
    >>> tp.save_trajectory(tp.IDs[5], format="csv")
    >>> tp.save_trajectory(tp.IDs[5], format="parquet")
    >>> ids, pData = tp.read_particles_at_time(0.0, doSave=False)
    >>> f = tp.plot_location(pData)
    """

    def __init__(
        self,
        dirs: str,
        iDomain: int = 0,
        iSpecies: int = 0,
        unit: str = "planetary",
        mass: float = proton_mass,
        charge: float = elementary_charge,
        iListStart: int = 0,
        iListEnd: int = -1,
        use_cache: bool = False,
    ):
        self.use_cache = use_cache
        self.unit = unit
        if self.unit not in {"planetary", "SI"}:
            raise ValueError(
                f"Unknown unit: '{self.unit}'. Must be 'planetary' or 'SI'."
            )
        self._trajectory_cache = {}
        self.mass = mass
        self.charge = charge

        header = Path(dirs + "/Header")
        if header.exists():
            with open(header, "r") as f:
                self.nReal = int(f.readline())
        else:
            raise FileNotFoundError(f"Header file not found in {dirs}")

        self.iSpecies = iSpecies
        self.pfiles = list()

        self.pfiles.extend(
            glob.glob(f"{dirs}/FLEKS{iDomain}_particle_species_{iSpecies}_n*")
        )

        self.pfiles.sort()

        if iListEnd == -1:
            iListEnd = len(self.pfiles)
        self.pfiles = self.pfiles[iListStart:iListEnd]

        self.particle_locations: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        for p_filename in self.pfiles:
            plist_filename = p_filename.replace(
                "_particle_species_", "_particle_list_species_"
            )
            plist = self.read_particle_list(plist_filename)
            for pID, ploc in plist.items():
                self.particle_locations.setdefault(pID, []).append((p_filename, ploc))

        self.IDs = sorted(self.particle_locations.keys())

        self.filetime = []
        for filename in self.pfiles:
            record = self._read_the_first_record(filename)
            if record is None:
                continue
            self.filetime.append(record[Indices.TIME])

    def __repr__(self):
        return (
            f"Particles species ID: {self.iSpecies}\n"
            f"Particle mass [kg]  : {self.mass}\n"
            f"Particle charge [C] : {self.charge}\n"
            f"Unit system         : {self.unit}\n"
            f"Number of particles : {len(self.IDs)}\n"
            f"First time tag      : {self.filetime[0] if self.filetime else 'N/A'}\n"
            f"Last  time tag      : {self.filetime[-1] if self.filetime else 'N/A'}\n"
        )

    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    def __getitem__(self, key):
        if isinstance(key, int):
            # Treat as an index
            pID = self.IDs[key]
        elif isinstance(key, tuple):
            # Treat as a pID
            pID = key
        else:
            raise TypeError(
                "Particle ID must be a tuple (cpu, id) or an integer index."
            )

        # If caching is not used, read directly and return.
        if not self.use_cache:
            return self.read_particle_trajectory(pID)

        # Caching is enabled, use the cache.
        if pID in self._trajectory_cache:
            return self._trajectory_cache[pID]
        else:
            trajectory = self.read_particle_trajectory(pID)
            self._trajectory_cache[pID] = trajectory
            return trajectory

    def getIDs(self):
        return self.IDs

    def read_particle_list(self, filename: str) -> Dict[Tuple[int, int], int]:
        """
        Read and return a list of the particle IDs.
        """
        record_format = "iiQ"  # 2 integers + 1 unsigned long long
        record_size = struct.calcsize(record_format)
        record_struct = struct.Struct(record_format)
        nByte = Path(filename).stat().st_size
        nPart = nByte // record_size
        plist = {}

        with open(filename, "rb") as f:
            for _ in range(nPart):
                dataChunk = f.read(record_size)
                (cpu, id, loc) = record_struct.unpack(dataChunk)
                plist.update({(cpu, id): loc})
        return plist

    def _read_the_first_record(self, filename: str) -> Union[List[float], None]:
        """
        Get the first record stored in one file.
        """
        dataList = list()
        with open(filename, "rb") as f:
            while True:
                binaryData = f.read(4 * 4)

                if not binaryData:
                    break  # EOF

                (cpu, idtmp, nRecord, weight) = struct.unpack("iiif", binaryData)
                if nRecord > 0:
                    binaryData = f.read(4 * self.nReal)
                    dataList = dataList + list(
                        struct.unpack("f" * self.nReal, binaryData)
                    )
                    return dataList

    def read_particles_at_time(
        self, time: float, doSave: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the information of all the particles at a given time.
        If doSave, save to a CSV file with the name "particles_t***.csv".

        Note that the time tags in filetime do not include the last saved time.

        Returns:
            ids: a numpy array of tuples contains the particle IDs.
            pData: a numpy real array with the particle weight, location and velocity.

        Examples:
        >>> ids, pData = pt.read_particles_at_time(3700, doSave=True)
        """
        nFile = len(self.pfiles)
        if time < self.filetime[0]:
            raise Exception(f"There are no particles at time {time}.")
        iFile = 0
        while iFile < nFile - 1:
            if time < self.filetime[iFile + 1]:
                break
            iFile += 1

        filename = self.pfiles[iFile]

        dataList: list[float] = []
        idList: list[tuple] = []
        with open(filename, "rb") as f:
            while True:
                binaryData = f.read(4 * 4)
                if not binaryData:
                    break  # EOF

                (cpu, idtmp, nRecord, weight) = struct.unpack("iiif", binaryData)
                binaryData = f.read(4 * self.nReal * nRecord)
                allRecords = list(struct.unpack("f" * nRecord * self.nReal, binaryData))
                for i in range(nRecord):
                    if allRecords[self.nReal * i + Indices.TIME] >= time:
                        dataList.append(
                            allRecords[self.nReal * i : self.nReal * (i + 1)]
                        )
                        idList.append((cpu, idtmp))
                        break
                    elif (
                        i == nRecord - 1
                        and allRecords[self.nReal * i + Indices.TIME] < time
                    ):
                        continue

        npData = np.array(dataList)
        idData = np.array(idList, dtype="i,i")
        # Selected time is larger than the last saved time
        if idData.size == 0:
            raise Exception(f"There are no particles at time {time}.")

        if doSave:
            filename = f"particles_t{time}.csv"
            header = "cpu,iid,time,x,y,z,vx,vy,vz"
            if self.nReal == 10:
                header += ",bx,by,bz"
            elif self.nReal == 13:
                header += ",bx,by,bz,ex,ey,ez"
            elif self.nReal == 22:
                header += ",dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz"

            with open(filename, "w") as f:
                f.write(header + "\n")
                for id_row, data_row in zip(idData, npData):
                    f.write(
                        f"{id_row[0]},{id_row[1]},{','.join(str(x) for x in data_row)}\n"
                    )

        return idData, npData

    def save_trajectory(
        self,
        pID: Tuple[int, int],
        filename: str = None,
        shiftTime: bool = False,
        scaleTime: bool = False,
        format: str = "csv",
    ) -> None:
        """
        Save the trajectory of a particle to a file.
        Args:
            pID: particle ID.
            filename (str, optional): The name of the file to save the trajectory to.
                                      If None, a default name will be generated.
            shiftTime (bool): If set to True, set the initial time to be 0.
            scaleTime (bool): If set to True, scale the time into [0,1] range.
            format (str): The output format, either "csv" or "parquet".
        Example:
        >>> tp.save_trajectory((3,15), format="parquet")
        """
        pData_lazy = self[pID]
        if filename is None:
            filename = f"trajectory_{pID[0]}_{pID[1]}.{format}"

        if self.unit == "planetary":
            header_cols = [
                "time [s]",
                "X [R]",
                "Y [R]",
                "Z [R]",
                "U_x [km/s]",
                "U_y [km/s]",
                "U_z [km/s]",
            ]
            if self.nReal >= 10:
                header_cols += ["B_x [nT]", "B_y [nT]", "B_z [nT]"]
            if self.nReal >= 13:
                header_cols += ["E_x [uV/m]", "E_y [uV/m]", "E_z [uV/m]"]
        elif self.unit == "SI":
            header_cols = [
                "time [s]",
                "X [m]",
                "Y [m]",
                "Z [m]",
                "U_x [m/s]",
                "U_y [m/s]",
                "U_z [m/s]",
            ]
            if self.nReal >= 10:
                header_cols += ["B_x [T]", "B_y [T]", "B_z [T]"]
            if self.nReal >= 13:
                header_cols += ["E_x [V/m]", "E_y [V/m]", "E_z [V/m]"]

        if self.nReal >= 22:
            header_cols += [
                "dBx_dx",
                "dBx_dy",
                "dBx_dz",
                "dBy_dx",
                "dBy_dy",
                "dBy_dz",
                "dBz_dx",
                "dBz_dy",
                "dBz_dz",
            ]

        if shiftTime:
            first_time = pData_lazy.select(pl.col("time").first()).collect().item()
            if first_time is not None:
                pData_lazy = pData_lazy.with_columns((pl.col("time") - first_time))
                if scaleTime:
                    last_time = (
                        pData_lazy.select(pl.col("time").last()).collect().item()
                    )
                    if last_time > 0:
                        pData_lazy = pData_lazy.with_columns(
                            (pl.col("time") / last_time)
                        )

        # Create a new LazyFrame with the desired header names
        pData_to_save = pData_lazy.select(
            [
                pl.col(original_name).alias(new_name)
                for original_name, new_name in zip(pData_lazy.columns, header_cols)
            ]
        )

        try:
            if format.lower() == "csv":
                pData_to_save.sink_csv(filename)
            elif format.lower() == "parquet":
                pData_to_save.sink_parquet(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except (IOError, pl.exceptions.PolarsError) as e:
            logger.error(f"Error saving trajectory to {format.upper()}: {e}")

    def save_trajectories(
        self,
        pIDs: Union[List[Tuple[int, int]], List[int]],
        filename: str = "trajectories.h5",
    ) -> None:
        """
        Save the trajectories of multiple particles to a single HDF5 file.

        Args:
            pIDs: A list of particle IDs to save. This can be a list of tuples
                  (cpu, id) or a list of integer indices.
            filename (str): The name of the HDF5 file to save the trajectories to.
        """
        with h5py.File(filename, "w") as f:
            if pIDs and isinstance(pIDs[0], int):
                # Handle list of integer indices
                for pID_index in pIDs:
                    pData_lazy = self[pID]
                    pData = pData_lazy.collect()
                    dataset_name = f"ID_{pID_index}"
                    dset = f.create_dataset(dataset_name, data=pData.to_numpy())
                    dset.attrs["columns"] = pData.columns
            else:
                # Handle list of tuples
                for pID in pIDs:
                    pData_lazy = self[pID]
                    pData = pData_lazy.collect()
                    dataset_name = f"ID_{pID[0]}_{pID[1]}"
                    dset = f.create_dataset(dataset_name, data=pData.to_numpy())
                    dset.attrs["columns"] = pData.columns

    def _get_particle_raw_data(self, pID: Tuple[int, int]) -> np.ndarray:
        """Reads all raw trajectory data for a particle across multiple files."""
        if pID not in self.particle_locations:
            return np.array([], dtype=np.float32)

        data_chunks = []
        record_format = "iiif"
        record_size = struct.calcsize(record_format)
        record_struct = struct.Struct(record_format)

        for filename, ploc in self.particle_locations[pID]:
            with open(filename, "rb") as f:
                f.seek(ploc)
                dataChunk = f.read(record_size)
                (_cpu, _idtmp, nRecord, _weight) = record_struct.unpack(dataChunk)
                if nRecord > 0:
                    binaryData = f.read(4 * self.nReal * nRecord)
                    data_chunks.append(np.frombuffer(binaryData, dtype=np.float32))
        if not data_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(data_chunks)

    def _read_particle_record(
        self, pID: Tuple[int, int], index: int = -1
    ) -> Union[list, None]:
        """Return a specific record of a test particle given its ID.

        Args:
            pID: particle ID
            index: The index of the record to be returned.
                   0: first record.
                   -1: last record (default).
        """
        if pID not in self.particle_locations:
            return None

        locations = self.particle_locations[pID]
        if not locations:
            return None

        record_format = "iiif"
        record_size = struct.calcsize(record_format)
        record_struct = struct.Struct(record_format)

        # Optimized path for the first record (index=0)
        if index == 0:
            for filename, ploc in locations:
                with open(filename, "rb") as f:
                    f.seek(ploc)
                    dataChunk = f.read(record_size)
                    (_cpu, _idtmp, nRecord, _weight) = record_struct.unpack(dataChunk)
                    if nRecord > 0:
                        # Found the first chunk with records, read the first one and return
                        binaryData = f.read(4 * self.nReal)
                        return list(struct.unpack("f" * self.nReal, binaryData))

        # Optimized path for the last record (index=-1)
        if index == -1:
            for filename, ploc in reversed(locations):
                with open(filename, "rb") as f:
                    f.seek(ploc)
                    dataChunk = f.read(record_size)
                    (_cpu, _idtmp, nRecord, _weight) = record_struct.unpack(dataChunk)
                    if nRecord > 0:
                        # This is the last chunk of data for this particle.
                        # Seek to the last record within this chunk.
                        offset = ploc + record_size + (nRecord - 1) * 4 * self.nReal
                        f.seek(offset)
                        binaryData = f.read(4 * self.nReal)
                        return list(struct.unpack("f" * self.nReal, binaryData))
        return None  # Only index 0 and -1 are supported

    def read_particle_trajectory(self, pID: Tuple[int, int]) -> pl.LazyFrame:
        """
        Return the trajectory of a test particle as a polars LazyFrame.
        """
        if pID not in self.particle_locations:
            raise KeyError(f"Particle ID {pID} not found.")

        data_array = self._get_particle_raw_data(pID)

        if data_array.size == 0:
            raise ValueError(f"No trajectory data found for particle ID {pID}.")

        nRecord = data_array.size // self.nReal
        trajectory_data = data_array.reshape(nRecord, self.nReal)

        # Use the Indices enum to create meaningful column names
        column_names = [i.name.lower() for i in islice(Indices, self.nReal)]
        lf = pl.from_numpy(data=trajectory_data, schema=column_names).lazy()
        return lf

    def read_initial_condition(self, pID: Tuple[int, int]) -> Union[list, None]:
        """
        Return the initial conditions of a test particle.
        """
        return self._read_particle_record(pID, index=0)

    def read_final_condition(self, pID: Tuple[int, int]) -> Union[list, None]:
        """
        Return the final conditions of a test particle.
        """
        return self._read_particle_record(pID, index=-1)

    def select_particles(self, f_select: Callable = None) -> List[Tuple[int, int]]:
        """
        Return the test particles whose initial conditions satisfy the requirement
        set by the user defined function f_select. The first argument of f_select is the
        particle ID, and the second argument is the ID of a particle.

        Examples:
        >>> from flekspy.tp import Indices
        >>> def f_select(tp, pid):
        >>>     pData = tp.read_initial_condition(pid)
        >>>     inTime = pData[Indices.TIME] < 3601
        >>>     inRegion = pData[Indices.X] > 20
        >>>     return inTime and inRegion
        >>>
        >>> pselected = tp.select_particles(f_select)
        >>> tp.plot_trajectory(list(pselected.keys())[1])
        """

        if f_select == None:

            def f_select(tp, pid):
                return True

        pSelected = list(filter(lambda pid: f_select(self, pid), self.IDs))

        return pSelected

    def get_kinetic_energy(self, vx, vy, vz):
        if self.unit == "planetary":
            ke = (
                0.5 * self.mass * (vx**2 + vy**2 + vz**2) * 1e6 / elementary_charge
            )  # [eV]
        elif self.unit == "SI":
            ke = 0.5 * self.mass * (vx**2 + vy**2 + vz**2) / elementary_charge  # [eV]

        return ke

    def get_kinetic_energy_change_rate(self, pt_lazy: pl.LazyFrame) -> pl.Series:
        """
        Calculates the rate of change of kinetic energy in [eV/s].
        """
        # Select only necessary columns before collecting to improve performance.
        collected = pt_lazy.select(["time", "vx", "vy", "vz"]).collect()
        time = collected["time"].to_numpy()
        vx = collected["vx"].to_numpy()
        vy = collected["vy"].to_numpy()
        vz = collected["vz"].to_numpy()

        ke = self.get_kinetic_energy(vx, vy, vz)
        dke_dt = np.gradient(ke, time)

        return pl.Series("dke_dt", dke_dt)

    def get_pitch_angle(self, pID):
        pt_lazy = self[pID]
        # Pitch Angle Calculation
        pitch_angle = self._get_pitch_angle_lazy(pt_lazy)

        return pitch_angle

    @staticmethod
    def _get_pitch_angle_lazy(lf: pl.LazyFrame) -> pl.Series:
        """
        Calculates the pitch angle from a LazyFrame.
        """
        # Pitch Angle Calculation
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )
        v_mag = (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).sqrt()
        b_mag = (pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()

        epsilon = 1e-15
        cos_alpha = v_dot_b / (v_mag * b_mag + epsilon)
        cos_alpha = cos_alpha.clip(-1.0, 1.0)
        pitch_angle_expr = (cos_alpha.arccos() * 180.0 / np.pi).alias("pitch_angle")

        return lf.select(pitch_angle_expr).collect().to_series()

    @staticmethod
    def get_pitch_angle_from_v_b(vx, vy, vz, bx, by, bz):
        # Pitch Angle Calculation
        v_vec = np.vstack((vx, vy, vz)).T
        b_vec = np.vstack((bx, by, bz)).T

        # Calculate magnitudes of velocity and B-field vectors
        v_mag = np.linalg.norm(v_vec, axis=1)
        b_mag = np.linalg.norm(b_vec, axis=1)

        # Calculate the dot product between V and B for each time step
        # Equivalent to (vx*bx + vy*by + vz*bz)
        v_dot_b = np.sum(v_vec * b_vec, axis=1)

        # To avoid division by zero if either vector magnitude is zero
        epsilon = 1e-15

        # Calculate the cosine of the pitch angle
        cos_alpha = v_dot_b / (v_mag * b_mag + epsilon)

        # Due to potential floating point inaccuracies, clip values to the valid range for arccos
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        # Calculate pitch angle and convert from radians to degrees
        pitch_angle = np.arccos(cos_alpha) * 180.0 / np.pi

        return pitch_angle

    def get_first_adiabatic_invariant(self, pt_lazy: pl.LazyFrame) -> pl.Series:
        """
        Calculates the 1st adiabatic invariant of a particle.
        The output units depend on the input data's units:
        - "planetary" (e.g., velocity in km/s, B-field in nT): result is in [1e9 J/T].
        - "SI" (e.g., velocity in m/s, B-field in T): result is in [J/T].
        """
        epsilon = 1e-15

        # Build the expression tree for the calculation
        v_mag_sq = pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2
        v_mag = v_mag_sq.sqrt()
        b_mag_expr = (pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )

        sin_alpha_sq = 1 - (v_dot_b / (v_mag * b_mag_expr + epsilon)) ** 2
        v_perp_sq = v_mag_sq * sin_alpha_sq
        if self.unit == "planetary":
            # Convert v_perp_sq from (km/s)^2 to (m/s)^2
            v_perp_sq = v_perp_sq * 1e6
        mu_expr = ((0.5 * self.mass * v_perp_sq) / (b_mag_expr + epsilon)).alias("mu")

        # Execute the expression and return
        return pt_lazy.select(mu_expr).collect()["mu"]

    @staticmethod
    def _calculate_bmag(
        df: Union[pl.DataFrame, pl.LazyFrame],
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Calculates the magnetic field magnitude.
        """
        df = df.with_columns(
            b_mag=(pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt(),
        )

        return df

    @staticmethod
    def _calculate_curvature(
        df: Union[pl.DataFrame, pl.LazyFrame],
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Calculates the magnetic field curvature vector and adds it to the DataFrame.
        κ = (b ⋅ ∇)b
        Depending on the selected units, output curvature may be
        - "planetary": [1/RE]
        - "SI": [1/m]
        """
        df = FLEKSTP._calculate_bmag(df)

        # Chain with_columns for better readability and performance
        df = df.with_columns(
            bx_u=pl.col("bx") / pl.col("b_mag"),
            by_u=pl.col("by") / pl.col("b_mag"),
            bz_u=pl.col("bz") / pl.col("b_mag"),
            dbx_u_dx=pl.col("dbxdx") / pl.col("b_mag"),
            dbx_u_dy=pl.col("dbxdy") / pl.col("b_mag"),
            dbx_u_dz=pl.col("dbxdz") / pl.col("b_mag"),
            dby_u_dx=pl.col("dbydx") / pl.col("b_mag"),
            dby_u_dy=pl.col("dbydy") / pl.col("b_mag"),
            dby_u_dz=pl.col("dbydz") / pl.col("b_mag"),
            dbz_u_dx=pl.col("dbzdx") / pl.col("b_mag"),
            dbz_u_dy=pl.col("dbzdy") / pl.col("b_mag"),
            dbz_u_dz=pl.col("dbzdz") / pl.col("b_mag"),
        )

        # Curvature vector: κ = (b ⋅ ∇)b
        kappa_x = (
            pl.col("bx_u") * pl.col("dbx_u_dx")
            + pl.col("by_u") * pl.col("dbx_u_dy")
            + pl.col("bz_u") * pl.col("dbx_u_dz")
        )
        kappa_y = (
            pl.col("bx_u") * pl.col("dby_u_dx")
            + pl.col("by_u") * pl.col("dby_u_dy")
            + pl.col("bz_u") * pl.col("dby_u_dz")
        )
        kappa_z = (
            pl.col("bx_u") * pl.col("dbz_u_dx")
            + pl.col("by_u") * pl.col("dbz_u_dy")
            + pl.col("bz_u") * pl.col("dbz_u_dz")
        )

        df = df.with_columns(kappa_x=kappa_x, kappa_y=kappa_y, kappa_z=kappa_z)

        return df

    def get_ExB_drift(self, pt_lazy: pl.LazyFrame) -> pl.DataFrame:
        """
        Calculates the convection drift velocity for a particle.
        v_exb = E x B / (B^2)
        Assuming Earth's planetary units, output drift velocity in [km/s].
        """
        lf = self._calculate_bmag(pt_lazy)

        # E x B expressions
        cross_x = pl.col("ey") * pl.col("bz") - pl.col("ez") * pl.col("by")
        cross_y = pl.col("ez") * pl.col("bx") - pl.col("ex") * pl.col("bz")
        cross_z = pl.col("ex") * pl.col("by") - pl.col("ey") * pl.col("bx")

        b_mag_sq = pl.col("b_mag") ** 2
        lf = lf.with_columns(
            vex=cross_x / b_mag_sq,
            vey=cross_y / b_mag_sq,
            vez=cross_z / b_mag_sq,
        )

        return lf.select(["vex", "vey", "vez"]).collect()

    def get_curvature_drift(
        self,
        pt_lazy: pl.LazyFrame,
    ) -> pl.DataFrame:
        """
        Calculates the curvature drift velocity for a particle.
        v_c = (m * v_parallel^2 / (q*B^2)) * (B x κ)
        Depending on the selected units, output drift velocity may be
        - "planetary": [km/s]
        - "SI": [m/s]
        """
        lf = self._calculate_bmag(pt_lazy)

        # Calculate v_parallel using expressions
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )
        v_parallel = v_dot_b / pl.col("b_mag")
        lf = lf.with_columns(v_parallel=v_parallel)

        # Calculate curvature
        lf = self._calculate_curvature(lf)

        # B x κ using expressions
        cross_x = pl.col("by") * pl.col("kappa_z") - pl.col("bz") * pl.col("kappa_y")
        cross_y = pl.col("bz") * pl.col("kappa_x") - pl.col("bx") * pl.col("kappa_z")
        cross_z = pl.col("bx") * pl.col("kappa_y") - pl.col("by") * pl.col("kappa_x")

        # Conversion factor expression
        v_parallel_sq = pl.col("v_parallel") ** 2
        b_mag_sq = pl.col("b_mag") ** 2
        if self.unit == "planetary":
            factor = (
                (self.mass * v_parallel_sq)
                / (self.charge * b_mag_sq)
                * 1e9
                / EARTH_RADIUS_KM
            )
        elif self.unit == "SI":
            factor = (self.mass * v_parallel_sq) / (self.charge * b_mag_sq)
        else:
            raise ValueError(
                f"Unknown unit: '{self.unit}'. Must be 'planetary' or 'SI'."
            )

        lf = lf.with_columns(
            vcx=factor * cross_x, vcy=factor * cross_y, vcz=factor * cross_z
        )

        return lf.select(["vcx", "vcy", "vcz"]).collect()

    def get_adiabaticity_parameter(
        self,
        pt_lazy: pl.LazyFrame,
    ) -> pl.Series:
        """
        Calculates the adiabaticity parameter, defined as the ratio of the
        magnetic field's radius of curvature to the particle's gyroradius.
        When this parameter is >> 1, the motion is adiabatic.
        """
        # Expression for v_perp
        v_mag_sq = pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2
        b_mag = (pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )
        sin_alpha_sq = 1 - (v_dot_b / (v_mag_sq.sqrt() * b_mag)) ** 2
        v_perp = (v_mag_sq * sin_alpha_sq).sqrt()

        # Expression for curvature radius
        lf_curv = self._calculate_curvature(pt_lazy)
        kappa_mag = (
            pl.col("kappa_x") ** 2 + pl.col("kappa_y") ** 2 + pl.col("kappa_z") ** 2
        ).sqrt()

        if self.unit == "planetary":
            # v_perp [km/s], b_mag [nT] -> r_g [km]
            r_g = (self.mass * v_perp) / (abs(self.charge) * b_mag) * 1e9
            # kappa_mag [1/RE] -> r_c [km]
            r_c_factor = EARTH_RADIUS_KM
        elif self.unit == "SI":
            # v_perp [m/s], b_mag [T] -> r_g [m]
            r_g = (self.mass * v_perp) / (abs(self.charge) * b_mag)
            # kappa_mag [1/m] -> r_c [m]
            r_c_factor = 1.0
        else:
            raise ValueError(
                f"Unknown unit: '{self.unit}'. Must be 'planetary' or 'SI'."
            )

        r_c = (1 / kappa_mag) * r_c_factor

        ratio_expr = (r_c / r_g).alias("adiabaticity")

        return lf_curv.select(ratio_expr).collect().to_series()

    @staticmethod
    def _calculate_gradient_b_magnitude(
        df: Union[pl.DataFrame, pl.LazyFrame],
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Calculates the gradient of the magnetic field magnitude.
        """
        # Gradient of B magnitude: ∇|B|
        grad_b_mag_x = (
            pl.col("bx") * pl.col("dbxdx")
            + pl.col("by") * pl.col("dbydx")
            + pl.col("bz") * pl.col("dbzdx")
        ) / pl.col("b_mag")
        grad_b_mag_y = (
            pl.col("bx") * pl.col("dbxdy")
            + pl.col("by") * pl.col("dbydy")
            + pl.col("bz") * pl.col("dbzdy")
        ) / pl.col("b_mag")
        grad_b_mag_z = (
            pl.col("bx") * pl.col("dbxdz")
            + pl.col("by") * pl.col("dbydz")
            + pl.col("bz") * pl.col("dbzdz")
        ) / pl.col("b_mag")

        df = df.with_columns(
            grad_b_mag_x=grad_b_mag_x,
            grad_b_mag_y=grad_b_mag_y,
            grad_b_mag_z=grad_b_mag_z,
        )
        return df

    def get_gradient_drift(
        self,
        pt_lazy: pl.LazyFrame,
    ) -> pl.DataFrame:
        """
        Calculates the gradient drift velocity for a particle.
        v_g = (μ / (q * B^2)) * (B x ∇|B|)
        Depending on the selected units, output drift velocity may be
        - "planetary": [km/s]
        - "SI": [m/s]
        """
        epsilon = 1e-15

        # Inlined expression for mu
        v_mag_sq = pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2
        v_mag = v_mag_sq.sqrt()
        b_mag = (pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )
        sin_alpha_sq = 1 - (v_dot_b / (v_mag * b_mag + epsilon)) ** 2
        v_perp_sq = v_mag_sq * sin_alpha_sq
        mu_expr = (0.5 * self.mass * v_perp_sq) / (b_mag + epsilon)

        lf = self._calculate_bmag(pt_lazy)

        # Gradient of B magnitude: ∇|B|
        lf = self._calculate_gradient_b_magnitude(lf)

        # B x ∇|B|
        cross_x = pl.col("by") * pl.col("grad_b_mag_z") - pl.col("bz") * pl.col(
            "grad_b_mag_y"
        )
        cross_y = pl.col("bz") * pl.col("grad_b_mag_x") - pl.col("bx") * pl.col(
            "grad_b_mag_z"
        )
        cross_z = pl.col("bx") * pl.col("grad_b_mag_y") - pl.col("by") * pl.col(
            "grad_b_mag_x"
        )

        b_mag_sq = pl.col("b_mag") ** 2
        # conversion factor
        if self.unit == "planetary":
            factor = mu_expr / (self.charge * b_mag_sq) * 1e9 / EARTH_RADIUS_KM
        elif self.unit == "SI":
            factor = mu_expr / (self.charge * b_mag_sq)
        else:
            raise ValueError(
                f"Unknown unit: '{self.unit}'. Must be 'planetary' or 'SI'."
            )

        lf = lf.with_columns(
            vgx=factor * cross_x, vgy=factor * cross_y, vgz=factor * cross_z
        )

        return lf.select(["vgx", "vgy", "vgz"]).collect()

    def get_polarization_drift(
        self,
        pt_lazy: pl.LazyFrame,
    ) -> pl.DataFrame:
        """
        Calculates the polarization drift velocity for a particle.
        v_p = (m / (q * B^2)) * (dE_perp / dt)
        Depending on the selected units, output drift velocity may be
        - "planetary": [km/s]
        - "SI": [m/s]
        """
        pt = pt_lazy.collect()
        time = pt["time"].to_numpy()

        # Calculate B magnitude and unit vector
        b_mag = (pt["bx"] ** 2 + pt["by"] ** 2 + pt["bz"] ** 2).sqrt()
        bx_u = pt["bx"] / b_mag
        by_u = pt["by"] / b_mag
        bz_u = pt["bz"] / b_mag

        # Calculate E_parallel = (E.b)b
        E_dot_b = pt["ex"] * bx_u + pt["ey"] * by_u + pt["ez"] * bz_u
        E_parallel_x = E_dot_b * bx_u
        E_parallel_y = E_dot_b * by_u
        E_parallel_z = E_dot_b * bz_u

        # Calculate E_perp = E - E_parallel
        E_perp_x = pt["ex"] - E_parallel_x
        E_perp_y = pt["ey"] - E_parallel_y
        E_perp_z = pt["ez"] - E_parallel_z

        # Calculate dE_perp/dt
        dE_perp_dt_x = pl.Series(np.gradient(E_perp_x, time))
        dE_perp_dt_y = pl.Series(np.gradient(E_perp_y, time))
        dE_perp_dt_z = pl.Series(np.gradient(E_perp_z, time))

        b_mag_sq = b_mag**2

        if self.unit == "planetary":
            # E is in [uV/m], B is in [nT]
            # Convert to SI: E[V/m] = E[uV/m]*1e-6, B[T] = B[nT]*1e-9
            # v_p [m/s] = (m[kg] / (q[C] * (B[nT]*1e-9)^2)) * (dE_perp/dt[uV/m/s] * 1e-6)
            # v_p [m/s] = (m/(q*B^2)) * (dE_perp/dt) * 1e12
            # To get km/s, we divide by 1e3
            factor = self.mass / (self.charge * b_mag_sq) * 1e9
        elif self.unit == "SI":
            factor = self.mass / (self.charge * b_mag_sq)

        vpx = factor * dE_perp_dt_x
        vpy = factor * dE_perp_dt_y
        vpz = factor * dE_perp_dt_z

        return pl.DataFrame({"vpx": vpx, "vpy": vpy, "vpz": vpz})

    def get_betatron_acceleration(self, pt, mu):
        """
        Calculates the Betatron acceleration term from particle trajectory data.

        The calculation follows the formula: dW/dt = μ * (∂B/∂t)
        where the partial derivative is found using: ∂B/∂t = dB/dt - v ⋅ ∇B

        Args:
            pt: A Polars LazyFrame containing the particle trajectory.
                     It must include columns for time, velocity (vx, vy, vz),
                     magnetic field (bx, by, bz), and the magnetic field
                     gradient tensor (e.g., 'dbxdx', 'dbydx', etc.).
            mu: A Polars Series containing the magnetic moment (first adiabatic invariant)
                of the particle.

        Returns:
            A new Polars LazyFrame with added intermediate columns and the
            final 'dW_betatron' column representing the rate of energy change in [eV/s].
        """

        # --- Step 1: Calculate the total derivative dB/dt ---
        pt = pt.with_columns(
            b_mag=(pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
        ).lazy()

        collected = pt.select("b_mag", "time").collect()
        B_mag = collected["b_mag"].to_numpy().flatten()
        time_steps = collected["time"].to_numpy().flatten()
        dB_dt = np.gradient(B_mag, time_steps)  # [nT/s]

        # --- Step 2: Define the rest of the calculations lazily ---
        pt_with_dbdt = pt.with_columns(pl.Series(name="dB_dt", values=dB_dt))

        # Gradient of B magnitude: ∇|B|
        pt_with_dbdt = self._calculate_gradient_b_magnitude(pt_with_dbdt)

        # Convective derivative: v ⋅ ∇|B| [nT/s]
        if self.unit == "planetary":
            v_dot_gradB = (
                pl.col("vx") * pl.col("grad_b_mag_x")
                + pl.col("vy") * pl.col("grad_b_mag_y")
                + pl.col("vz") * pl.col("grad_b_mag_z")
            ) / EARTH_RADIUS_KM
        elif self.unit == "SI":
            v_dot_gradB = (
                pl.col("vx") * pl.col("grad_b_mag_x")
                + pl.col("vy") * pl.col("grad_b_mag_y")
                + pl.col("vz") * pl.col("grad_b_mag_z")
            )

        # --- Step 3: Calculate the partial derivative ∂B/∂t ---
        partial_B_partial_t = pl.col("dB_dt") - v_dot_gradB

        # --- Step 4: Chain all calculations and compute the final Betatron term ---
        result = pt_with_dbdt.with_columns(
            v_dot_gradB=v_dot_gradB,
            partial_B_partial_t=partial_B_partial_t,
        )
        # Unit conversion to [eV/s].
        if self.unit == "planetary":
            # mu is in [1e9 J/T], partial_B_partial_t is in [nT/s].
            # The product mu * partial_B_partial_t is in [J/s] because
            # the nT -> T conversion (1e-9) and the mu unit (1e9) cancel out.
            result = result.with_columns(
                dW_betatron=mu * partial_B_partial_t / elementary_charge
            )
        elif self.unit == "SI":
            # mu is in [J/T], partial_B_partial_t is in [T/s].
            # The product is directly in [J/s].
            result = result.with_columns(
                dW_betatron=mu * partial_B_partial_t / elementary_charge
            )

        return result

    def get_energy_change_guiding_center(
        self,
        pID: Tuple[int, int],
    ) -> pl.DataFrame:
        """
        Computes the change of energy of a single particle based on the guiding center theory.

        The formula is given by:
        dW/dt = q*E_parallel*v_parallel + mu*(∂B/∂t + u_E.∇B) + m*v_parallel^2*(u_E.κ)

        where W is the particle energy, B is the magnetic field magnitude,
        u_E is the E cross B drift, and κ is the magnetic field curvature.
        The first term on the right hand side is the parallel acceleration,
        the second term is the Betatron acceleration, and the third term
        is one type of Fermi acceleration.

        Args:
            pID (Tuple[int, int]): The particle ID (cpu, id).

        Returns:
            A Polars DataFrame with the time, the three energy change components,
            and the total, in [eV/s].
        """
        pt_lazy = self[pID]

        # It's easier to work with collected data for this kind of calculation
        pt = pt_lazy.collect()
        time = pt["time"]

        # --- Common quantities ---
        mu = self.get_first_adiabatic_invariant(pt_lazy)  # returns Series
        u_E = self.get_ExB_drift(pt_lazy)  # returns DataFrame

        # B magnitude and unit vectors
        b_mag = (pt["bx"] ** 2 + pt["by"] ** 2 + pt["bz"] ** 2).sqrt()
        b_mag = b_mag.rename("b_mag")
        bx_u = (pt["bx"] / b_mag).rename("bx_u")
        by_u = (pt["by"] / b_mag).rename("by_u")
        bz_u = (pt["bz"] / b_mag).rename("bz_u")

        # Parallel E and v
        E_parallel = (pt["ex"] * bx_u + pt["ey"] * by_u + pt["ez"] * bz_u).rename(
            "E_parallel"
        )
        v_parallel = (pt["vx"] * bx_u + pt["vy"] * by_u + pt["vz"] * bz_u).rename(
            "v_parallel"
        )

        # --- 1. Parallel Acceleration Term ---
        if self.unit == "planetary":
            unit_factor_parallel = 1e-3  # (uV/m)*(km/s) -> 1e-3 J/C/s
        else:  # SI
            unit_factor_parallel = 1.0

        dW_parallel = (
            (self.charge / elementary_charge)
            * E_parallel
            * v_parallel
            * unit_factor_parallel
        )
        dW_parallel = dW_parallel.rename("dW_parallel")

        # --- 2. Betatron Acceleration Term ---
        # ∂B/∂t = dB/dt - v ⋅ ∇B
        dB_dt = pl.Series(np.gradient(b_mag, time.to_numpy())).rename("dB_dt")

        # Need nabla B
        pt_with_b_mag = pt.with_columns(b_mag)
        nabla_b_df = self._calculate_gradient_b_magnitude(
            pt_with_b_mag.lazy()
        ).collect()

        v_dot_gradB = (
            pt["vx"] * nabla_b_df["grad_b_mag_x"]
            + pt["vy"] * nabla_b_df["grad_b_mag_y"]
            + pt["vz"] * nabla_b_df["grad_b_mag_z"]
        )

        if self.unit == "planetary":
            v_dot_gradB = v_dot_gradB / EARTH_RADIUS_KM  # nT/s

        partial_B_partial_t = (dB_dt - v_dot_gradB).rename("partial_B_partial_t")

        # u_E . nabla B
        u_E_dot_nabla_B = (
            u_E["vex"] * nabla_b_df["grad_b_mag_x"]
            + u_E["vey"] * nabla_b_df["grad_b_mag_y"]
            + u_E["vez"] * nabla_b_df["grad_b_mag_z"]
        )
        if self.unit == "planetary":
            u_E_dot_nabla_B = u_E_dot_nabla_B / EARTH_RADIUS_KM  # nT/s

        betatron_term_in_paren = partial_B_partial_t + u_E_dot_nabla_B

        if self.unit == "planetary":
            # mu is in [J/nT], term is in [nT/s] -> J/s
            dW_betatron_J_s = mu * betatron_term_in_paren
        else:  # SI
            # mu is in [J/T], term is in [T/s] -> J/s
            dW_betatron_J_s = mu * betatron_term_in_paren

        dW_betatron = (dW_betatron_J_s / elementary_charge).rename("dW_betatron")

        # --- 3. Fermi Acceleration Term ---
        # m * v_parallel^2 * (u_E . kappa)
        kappa_df = self._calculate_curvature(pt_with_b_mag.lazy()).collect()

        u_E_dot_kappa = (
            u_E["vex"] * kappa_df["kappa_x"]
            + u_E["vey"] * kappa_df["kappa_y"]
            + u_E["vez"] * kappa_df["kappa_z"]
        )

        v_parallel_sq = v_parallel**2

        if self.unit == "planetary":
            # v_parallel is km/s -> m/s, so v^2 -> v^2*1e6
            v_parallel_sq_si = v_parallel_sq * 1e6  # (m/s)^2
            # u_E_dot_kappa is km/s/RE. To get 1/s, divide by RE_km
            u_E_dot_kappa_inv_s = u_E_dot_kappa / EARTH_RADIUS_KM  # 1/s
            dW_fermi_J_s = self.mass * v_parallel_sq_si * u_E_dot_kappa_inv_s
        else:  # SI
            # v_parallel is m/s, u_E_dot_kappa is 1/s
            dW_fermi_J_s = self.mass * v_parallel_sq * u_E_dot_kappa

        dW_fermi = (dW_fermi_J_s / elementary_charge).rename("dW_fermi")

        # --- Combine results ---
        df = pl.DataFrame(
            {
                "time": time,
                "dW_parallel": dW_parallel,
                "dW_betatron": dW_betatron,
                "dW_fermi": dW_fermi,
            }
        )

        df = df.with_columns(
            dW_total=pl.col("dW_parallel") + pl.col("dW_betatron") + pl.col("dW_fermi")
        )

        return df

    def integrate_drift_accelerations(
        self,
        pid: tuple[int, int],
    ):
        """
        Compute plasma drift velocities and the associated rate of energy change in [eV/s].
        """
        pt = self[pid]
        vc = self.get_curvature_drift(pt)
        vg = self.get_gradient_drift(pt)
        vp = self.get_polarization_drift(pt)
        mu = self.get_first_adiabatic_invariant(pt)
        pt = self.get_betatron_acceleration(pt, mu)

        if self.unit == "planetary":
            UNIT_FACTOR = 1e-3
        elif self.unit == "SI":
            UNIT_FACTOR = 1.0

        vx = pt.select("vx").collect().to_numpy().flatten()
        vy = pt.select("vy").collect().to_numpy().flatten()
        vz = pt.select("vz").collect().to_numpy().flatten()
        ke = self.get_kinetic_energy(vx, vy, vz)  # [eV]

        pt = (
            pt.with_columns(
                ke=ke,
                b_mag=(
                    pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2
                ).sqrt(),
            )
            .with_columns(
                bx_u=pl.col("bx") / pl.col("b_mag"),
                by_u=pl.col("by") / pl.col("b_mag"),
                bz_u=pl.col("bz") / pl.col("b_mag"),
            )
            .with_columns(
                E_parallel=(
                    pl.col("ex") * pl.col("bx_u")
                    + pl.col("ey") * pl.col("by_u")
                    + pl.col("ez") * pl.col("bz_u")
                ),
                v_parallel=(
                    pl.col("vx") * pl.col("bx_u")
                    + pl.col("vy") * pl.col("by_u")
                    + pl.col("vz") * pl.col("bz_u")
                ),
            )
        )

        # Calculate the dot product of E with each drift velocity [eV/s]
        pt = pt.with_columns(
            # Energy change from gradient drift
            dWg=(
                pl.col("ex") * vg["vgx"]
                + pl.col("ey") * vg["vgy"]
                + pl.col("ez") * vg["vgz"]
            )
            * UNIT_FACTOR,
            # Energy change from curvature drift
            dWc=(
                pl.col("ex") * vc["vcx"]
                + pl.col("ey") * vc["vcy"]
                + pl.col("ez") * vc["vcz"]
            )
            * UNIT_FACTOR,
            # Energy change from polarization drift
            dWp=(
                pl.col("ex") * vp["vpx"]
                + pl.col("ey") * vp["vpy"]
                + pl.col("ez") * vp["vpz"]
            )
            * UNIT_FACTOR,
            # Energy change from parallel acceleration
            dW_parallel=(pl.col("E_parallel") * pl.col("v_parallel")) * UNIT_FACTOR,
        ).collect()

        # 1. Calculate the time step 'dt' between each measurement
        dt = pl.col("time").diff().fill_null(0)

        # 2. Integrate each term using the trapezoidal rule and a cumulative sum
        pt = pt.with_columns(
            # Integrated energy from gradient drift
            Wg_integrated=((pl.col("dWg") + pl.col("dWg").shift(1)) / 2 * dt)
            .cum_sum()
            .fill_null(0),
            # Integrated energy from curvature drift
            Wc_integrated=((pl.col("dWc") + pl.col("dWc").shift(1)) / 2 * dt)
            .cum_sum()
            .fill_null(0),
            # Integrated energy from polarization drift
            Wp_integrated=((pl.col("dWp") + pl.col("dWp").shift(1)) / 2 * dt)
            .cum_sum()
            .fill_null(0),
            # Integrated energy from parallel acceleration
            W_parallel_integrated=(
                (pl.col("dW_parallel") + pl.col("dW_parallel").shift(1)) / 2 * dt
            )
            .cum_sum()
            .fill_null(0),
            # Also integrate the betatron term if it exists
            W_betatron_integrated=(
                (pl.col("dW_betatron") + pl.col("dW_betatron").shift(1)) / 2 * dt
            )
            .cum_sum()
            .fill_null(0),
        )

        # Let's also create a column for the total integrated energy change
        pt = pt.with_columns(
            W_total_integrated=(
                pl.col("Wg_integrated")
                + pl.col("Wc_integrated")
                + pl.col("Wp_integrated")
                + pl.col("W_parallel_integrated")
                + pl.col("W_betatron_integrated")
            )
        )

        df = pt.select(
            [
                "time",
                "ke",
                "Wg_integrated",
                "Wc_integrated",
                "Wp_integrated",
                "W_parallel_integrated",
                "W_betatron_integrated",
            ]
        )

        return df

    def analyze_drifts(
        self,
        pid: tuple[int, int],
        outname=None,
        switchYZ=False,
    ):
        """
        Compute plasma drift velocities and the associated rate of energy change in [eV/s].
        """
        pt = self[pid]
        ve = self.get_ExB_drift(pt)
        vc = self.get_curvature_drift(pt)
        vg = self.get_gradient_drift(pt)
        vp = self.get_polarization_drift(pt)
        adiabaticity = self.get_adiabaticity_parameter(pt)
        mu = self.get_first_adiabatic_invariant(pt)
        pt = self.get_betatron_acceleration(pt, mu)
        dke_dt = self.get_kinetic_energy_change_rate(pt)
        # Calculate the dot product of E with each drift velocity [eV/s]
        if self.unit == "planetary":
            UNIT_FACTOR = 1e-3
        elif self.unit == "SI":
            UNIT_FACTOR = 1.0

        pt = (
            pt.with_columns(
                b_mag=(pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
            )
            .with_columns(
                bx_u=pl.col("bx") / pl.col("b_mag"),
                by_u=pl.col("by") / pl.col("b_mag"),
                bz_u=pl.col("bz") / pl.col("b_mag"),
            )
            .with_columns(
                E_parallel=(
                    pl.col("ex") * pl.col("bx_u")
                    + pl.col("ey") * pl.col("by_u")
                    + pl.col("ez") * pl.col("bz_u")
                ),
                v_parallel=(
                    pl.col("vx") * pl.col("bx_u")
                    + pl.col("vy") * pl.col("by_u")
                    + pl.col("vz") * pl.col("bz_u")
                ),
            )
        )

        # Calculate the dot product of E with each drift velocity [eV/s]
        pt = pt.with_columns(
            # Energy change from gradient drift
            dWg=(
                pl.col("ex") * vg["vgx"]
                + pl.col("ey") * vg["vgy"]
                + pl.col("ez") * vg["vgz"]
            )
            * UNIT_FACTOR,
            # Energy change from curvature drift
            dWc=(
                pl.col("ex") * vc["vcx"]
                + pl.col("ey") * vc["vcy"]
                + pl.col("ez") * vc["vcz"]
            )
            * UNIT_FACTOR,
            # Energy change from polarization drift
            dWp=(
                pl.col("ex") * vp["vpx"]
                + pl.col("ey") * vp["vpy"]
                + pl.col("ez") * vp["vpz"]
            )
            * UNIT_FACTOR,
            # Energy change from parallel acceleration
            dW_parallel=(pl.col("E_parallel") * pl.col("v_parallel")) * UNIT_FACTOR,
        ).collect()

        fig, axes = plt.subplots(
            nrows=7, ncols=1, figsize=(12, 12), sharex=True, constrained_layout=True
        )

        def _plot_velocity_subplot(
            ax, time, vel_df, ylabel, switchYZ, x_col, y_col, z_col
        ):
            """Helper to plot a velocity subplot."""
            ax.plot(time, vel_df[x_col], label=x_col)
            if switchYZ:
                # Swap y and z data for plotting
                ax.plot(time, vel_df[z_col], label=y_col)
                ax.plot(time, vel_df[y_col], label=z_col)
            else:
                ax.plot(time, vel_df[y_col], label=y_col)
                ax.plot(time, vel_df[z_col], label=z_col)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.legend(ncol=3, fontsize="medium")
            ax.grid(True, linestyle="--", alpha=0.6)

        # --- 1. Raw Velocities ---
        _plot_velocity_subplot(
            axes[0], pt["time"], pt, "V [km/s]", switchYZ, "vx", "vy", "vz"
        )

        # --- 2. Plasma Convection Drift (vex, vey, vez) ---
        _plot_velocity_subplot(
            axes[1],
            pt["time"],
            ve,
            r"$V_{\mathbf{E}\times\mathbf{B}}$ [km/s]",
            switchYZ,
            "vex",
            "vey",
            "vez",
        )

        # --- 3. Plasma Gradient Drift (vgx, vgy, vgz) ---
        _plot_velocity_subplot(
            axes[2],
            pt["time"],
            vg,
            r"$V_{\nabla B}$ [km/s]",
            switchYZ,
            "vgx",
            "vgy",
            "vgz",
        )

        # --- 4. Plasma Curvature Drift (vcx, vcy, vcz) ---
        _plot_velocity_subplot(
            axes[3], pt["time"], vc, r"$V_c$ [km/s]", switchYZ, "vcx", "vcy", "vcz"
        )

        # --- 5. Plasma Polarization Drift (vpx, vpy, vpz) ---
        _plot_velocity_subplot(
            axes[4], pt["time"], vp, r"$V_p$ [km/s]", switchYZ, "vpx", "vpy", "vpz"
        )

        # --- 6. Rate of Energy Change (E dot V) ---
        axes[5].plot(
            pt["time"], pt["dWg"], label=r"$q \mathbf{E} \cdot \mathbf{V}_{\nabla B}$"
        )
        axes[5].plot(
            pt["time"], pt["dWc"], label=r"$q \mathbf{E} \cdot \mathbf{V}_{c}$"
        )
        axes[5].plot(
            pt["time"], pt["dWp"], label=r"$q \mathbf{E} \cdot \mathbf{V}_{p}$"
        )
        axes[5].plot(
            pt["time"], pt["dW_parallel"], label=r"$q E_{\|} v_{\|}$", linestyle="--"
        )
        axes[5].plot(
            pt["time"], pt["dW_betatron"], label="Betatron", linestyle="--", alpha=0.8
        )
        axes[5].plot(pt["time"], dke_dt, label="dKE/dt", linestyle="-.", alpha=0.8)
        axes[5].set_ylabel("Energy change rate\n [eV/s]", fontsize=14)
        axes[5].legend(ncol=3, fontsize="medium")
        axes[5].grid(True, linestyle="--", alpha=0.6)

        axes[-1].semilogy(pt["time"], adiabaticity)
        axes[-1].axhline(y=1.0, linestyle="--", color="tab:red")
        axes[-1].set_ylim(adiabaticity.quantile(0.001), adiabaticity.max())
        axes[-1].set_ylabel(r"$r_c / r_L$", fontsize=14)
        axes[-1].grid(True, linestyle="--", alpha=0.6)

        for ax in axes:
            ax.set_xlim(left=pt["time"][0], right=pt["time"][-1])

        axes[-1].set_xlabel("Time [s]", fontsize=14)

        if outname is not None:
            plt.savefig(outname, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def analyze_drifts_energy_change(
        self,
        pID: Tuple[int, int],
        outname=None,
    ):
        """
        Analyzes and plots the energy changes for each term in the guiding center
        approximation.

        This method computes the parallel, Betatron, and Fermi accelerations using
        `get_energy_change_guiding_center`. It also calculates the total kinetic
        energy change and treats the difference between the kinetic energy change
        and the sum of the guiding center terms (dW_total) as the non-adiabatic term.

        The method generates a plot with four subplots:
        1. dW_parallel: Energy change due to parallel electric fields.
        2. dW_betatron: Energy change due to the Betatron effect.
        3. dW_fermi: Energy change due to Fermi acceleration.
        4. dW_total and Non-adiabatic term: The sum of the above terms compared
           with the non-adiabatic heating component.

        Args:
            pID (Tuple[int, int]): The particle ID (cpu, id).
            outname (str, optional): If provided, the plot is saved to this
                                     filename instead of being shown. Defaults to None.
        """
        # --- 1. Get Guiding Center Energy Changes ---
        df_gc = self.get_energy_change_guiding_center(pID)
        time = df_gc["time"]
        dW_parallel = df_gc["dW_parallel"]
        dW_betatron = df_gc["dW_betatron"]
        dW_fermi = df_gc["dW_fermi"]
        dW_total = df_gc["dW_total"]

        # --- 2. Get Kinetic Energy Change Rate ---
        pt_lazy = self[pID]
        dke_dt = self.get_kinetic_energy_change_rate(pt_lazy)

        # --- 3. Calculate Non-adiabatic Term ---
        # Ensure dke_dt is aligned with the time from df_gc if lengths differ
        # (though they should be the same)
        if len(dke_dt) != len(dW_total):
            # This case should not happen if the underlying data is consistent.
            # Raise an error to prevent crashing later.
            msg = (
                f"Mismatch in length between kinetic energy and guiding center calculations: "
                f"len(dke_dt)={len(dke_dt)}, len(dW_total)={len(dW_total)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        non_adiabatic_term = dke_dt - dW_total

        # --- 4. Plotting ---
        fig, axes = plt.subplots(
            nrows=4, ncols=1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        fig.suptitle(f"Energy Change Analysis for Particle {pID}", fontsize=16)

        # Subplots for individual energy change terms
        plot_configs = [
            {
                "data": dW_parallel,
                "label": r"$dW_{\parallel}/dt$",
                "title": "Parallel Acceleration",
            },
            {
                "data": dW_betatron,
                "label": "$dW_{betatron}/dt$",
                "title": "Betatron Acceleration",
                "color": "tab:orange",
            },
            {
                "data": dW_fermi,
                "label": "$dW_{fermi}/dt$",
                "title": "Fermi Acceleration",
                "color": "tab:green",
            },
        ]

        for i, config in enumerate(plot_configs):
            ax = axes[i]
            ax.plot(
                time, config["data"], label=config["label"], color=config.get("color")
            )
            ax.set_ylabel("Rate [eV/s]")
            ax.set_title(config["title"])
            ax.grid(True, linestyle="--", alpha=0.6)

        # Subplot 4: dW_total and Non-adiabatic term
        axes[3].plot(time, dW_total, label="$dW_{total}/dt$ (GC)", color="tab:red")
        axes[3].plot(
            time,
            non_adiabatic_term,
            label="Non-adiabatic",
            color="tab:purple",
            linestyle="--",
        )
        axes[3].plot(
            time, dke_dt, label="d(KE)/dt", color="black", linestyle=":", alpha=0.7
        )
        axes[3].set_ylabel("Rate [eV/s]")
        axes[3].set_title("Total and Non-Adiabatic Energy Change")
        axes[3].grid(True, linestyle="--", alpha=0.6)
        axes[3].legend(fontsize="medium")

        axes[-1].set_xlabel("Time [s]", fontsize=14)
        for ax in axes:
            ax.set_xlim(left=time.min(), right=time.max())

        if outname is not None:
            plt.savefig(outname, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def analyze_drift(
        self,
        pID: tuple[int, int],
        drift_type: str,
        outname=None,
    ):
        """
        Analyzes a specific drift for a particle, plotting its velocity, the
        electric field, the energy change rate, and the integrated energy change.

        Args:
            pID (tuple[int, int]): The particle ID (cpu, id).
            drift_type (str): The type of drift to analyze. Supported options are:
                              'ExB', 'gradient', 'curvature', 'polarization'.
            outname (str, optional): If provided, the plot is saved to this
                                      filename instead of being shown. Defaults to None.
        """
        drift_getters = {
            "ExB": self.get_ExB_drift,
            "gradient": self.get_gradient_drift,
            "curvature": self.get_curvature_drift,
            "polarization": self.get_polarization_drift,
        }

        if drift_type not in drift_getters:
            raise ValueError(
                f"Unknown drift_type: '{drift_type}'. "
                f"Available options are {list(drift_getters.keys())}"
            )

        # 1. Get data
        pt_lazy = self[pID]
        v_drift_df = drift_getters[drift_type](pt_lazy)
        pt_df = pt_lazy.collect()
        time = pt_df["time"]

        # Rename columns of v_drift_df to be generic for plotting
        v_drift_df.columns = ["v_drift_x", "v_drift_y", "v_drift_z"]

        # 2. Calculate energy change rate
        if self.unit == "planetary":
            # E[uV/m] * v[km/s] -> (1e-6 V/m) * (1e3 m/s) = 1e-3 J/C/s
            # This is the rate of energy change in eV/s
            UNIT_FACTOR = 1e-3
        elif self.unit == "SI":
            UNIT_FACTOR = 1.0

        d_W = (
            pt_df["ex"] * v_drift_df["v_drift_x"]
            + pt_df["ey"] * v_drift_df["v_drift_y"]
            + pt_df["ez"] * v_drift_df["v_drift_z"]
        ) * UNIT_FACTOR
        d_W = d_W.rename("d_W")

        # 3. Integrate energy change using cumulative trapezoidal rule
        dt = time.diff().fill_null(0)
        W_integrated = ((d_W + d_W.shift(1)) / 2 * dt).cum_sum().fill_null(0)
        W_integrated = W_integrated.rename("W_integrated")

        # 4. Plotting
        fig, axes = plt.subplots(
            nrows=4, ncols=1, figsize=(12, 10), sharex=True, constrained_layout=True
        )

        if self.unit == "planetary":
            v_unit_label = "km/s"
            e_unit_label = "μV/m"
        else:  # self.unit == "SI"
            v_unit_label = "m/s"
            e_unit_label = "V/m"

        # Plot 1: Drift velocity
        axes[0].plot(time, v_drift_df["v_drift_x"], label="$V_x$")
        axes[0].plot(time, v_drift_df["v_drift_y"], label="$V_y$")
        axes[0].plot(time, v_drift_df["v_drift_z"], label="$V_z$")
        axes[0].set_ylabel(f"$V_{{{drift_type}}}$ [{v_unit_label}]", fontsize=14)
        axes[0].legend(ncol=3, fontsize="medium")
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Plot 2: Electric field
        axes[1].plot(time, pt_df["ex"], label="$E_x$")
        axes[1].plot(time, pt_df["ey"], label="$E_y$")
        axes[1].plot(time, pt_df["ez"], label="$E_z$")
        axes[1].set_ylabel(f"E [{e_unit_label}]", fontsize=14)
        axes[1].legend(ncol=3, fontsize="medium")
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # Plot 3: Energy change rate
        axes[2].plot(time, d_W)
        axes[2].set_ylabel("Energy Change Rate\n[eV/s]", fontsize=14)
        axes[2].grid(True, linestyle="--", alpha=0.6)

        # Plot 4: Integrated energy change
        axes[3].plot(time, W_integrated)
        axes[3].set_ylabel("Integrated Energy [eV]", fontsize=14)
        axes[3].set_xlabel("Time [s]", fontsize=14)
        axes[3].grid(True, linestyle="--", alpha=0.6)

        fig.suptitle(f"Analysis of {drift_type} Drift for Particle {pID}", fontsize=16)

        for ax in axes:
            ax.set_xlim(left=time.min(), right=time.max())

        if outname is not None:
            plt.savefig(outname, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_work_energy_verification(self, pID: Tuple[int, int], outname=None):
        """
        Verifies the work-energy theorem for a particle by plotting the rate of
        change of kinetic energy against the work rate done by the electric field.
        It also plots the integrated change in kinetic energy versus the total
        work done.

        Args:
            pID (Tuple[int, int]): The particle ID (cpu, id).
            outname (str, optional): If provided, the plot is saved to this
                                     filename. Defaults to None (displays plot).
        """
        pt_lazy = self[pID]
        # Collect necessary columns once to improve performance.
        required_cols = ["time", "vx", "vy", "vz", "ex", "ey", "ez"]
        pt_df = pt_lazy.select(required_cols).collect()
        time = pt_df["time"]

        # 1. Calculate the rate of change of kinetic energy (d(KE)/dt)
        ke = self.get_kinetic_energy(pt_df["vx"], pt_df["vy"], pt_df["vz"])
        dke_dt = pl.Series("dke_dt", np.gradient(ke.to_numpy(), time.to_numpy()))

        # 2. Calculate the rate of work done by the electric field (q * E.v)
        if self.unit == "planetary":
            # E[uV/m] * v[km/s] -> (1e-6 V/m) * (1e3 m/s) = 1e-3 J/C/s
            # To get eV/s, multiply by (1 / elementary_charge)
            unit_factor = 1e-3
        elif self.unit == "SI":
            unit_factor = 1.0

        work_rate = (
            (
                pt_df["ex"] * pt_df["vx"]
                + pt_df["ey"] * pt_df["vy"]
                + pt_df["ez"] * pt_df["vz"]
            )
            * unit_factor
            * self.charge
            / elementary_charge
        )
        work_rate = work_rate.rename("work_rate")

        # 3. Integrate both rates over time
        dt = time.diff().fill_null(0)

        # Integrated change in kinetic energy
        delta_ke_integrated = (
            ((dke_dt + dke_dt.shift(1)) / 2 * dt).cum_sum().fill_null(0)
        )

        # Integrated work done
        work_done_integrated = (
            ((work_rate + work_rate.shift(1)) / 2 * dt).cum_sum().fill_null(0)
        )

        # 4. Plotting
        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        fig.suptitle(f"Work-Energy Verification for Particle {pID}", fontsize=16)

        # Panel 1: Rates of change
        axes[0].plot(time, dke_dt, label="d(KE)/dt", linewidth=2)
        axes[0].plot(
            time, work_rate, label="q E⋅v (Work Rate)", linestyle="--", linewidth=2
        )
        axes[0].set_ylabel("Rate [eV/s]", fontsize=14)
        axes[0].set_title("Rate of Kinetic Energy Change vs. Work Rate", fontsize=14)
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Panel 2: Integrated quantities
        axes[1].plot(time, delta_ke_integrated, label="ΔKE (Integrated)", linewidth=2)
        axes[1].plot(
            time,
            work_done_integrated,
            label="Total Work Done",
            linestyle="--",
            linewidth=2,
        )
        axes[1].set_ylabel("Energy [eV]", fontsize=14)
        axes[1].set_title(
            "Integrated Kinetic Energy Change vs. Total Work Done", fontsize=14
        )
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        axes[1].set_xlabel("Time [s]", fontsize=14)

        for ax in axes:
            ax.set_xlim(left=time.min(), right=time.max())

        if outname is not None:
            plt.savefig(outname, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def find_shock_crossing_time(self, pid, b_threshold_factor=2.5, verbose=False):
        """
        Finds the shock crossing time for a single particle.

        The shock is identified by finding the first rate of change in the
        magnetic field magnitude that exceeds a threshold, which signifies a
        rapid transition between the upstream and downstream regions.

        Args:
            pid: particle index.
            b_threshold_factor (float): A multiplier for the standard deviation of
                                        the B-field derivative. A larger value makes
                                        the detection less sensitive to minor
                                        fluctuations. Defaults to 2.5.
            verbose (bool): If True, prints diagnostic information. Defaults to False.

        Returns:
            float or None: The time of the shock crossing in seconds. Returns None if
                        no significant crossing is detected based on the criteria.
        """
        # --- 1. Data Preparation ---
        pt = self[pid]
        t_and_b_mag = (
            pt.with_columns(
                b_mag=(pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
            )
            .select("time", "b_mag")
            .collect()
        )
        t = t_and_b_mag["time"].to_numpy()
        b_mag = t_and_b_mag["b_mag"].to_numpy()

        # Ensure there are enough data points for a derivative calculation
        if len(t) < 3:
            if verbose:
                logger.warning("Not enough data points to reliably find a shock.")
            return None

        # --- 2. Calculate the Rate of Change ---
        # Use np.gradient to find the time derivative of the B-field magnitude.
        # This correctly handles potentially uneven time steps.
        db_dt = np.gradient(b_mag, t)
        abs_db_dt = np.abs(db_dt)

        # --- 3. Dynamic Thresholding for Spike Detection ---
        # Set a threshold to distinguish significant spikes from noise.
        mean_db_dt = np.mean(abs_db_dt)
        std_db_dt = np.std(abs_db_dt)
        threshold = mean_db_dt + b_threshold_factor * std_db_dt

        # Find all time indices where the derivative exceeds this threshold
        candidate_indices = np.where(abs_db_dt > threshold)[0]

        # --- 4. Identify the Most Likely Crossing Time ---
        # If no points are above the threshold, no shock was detected.
        if candidate_indices.size == 0:
            if verbose:
                logger.info(
                    f"No B-field change above the threshold ({threshold:.2f} nT/s) was found."
                )
            return None

        shock_idx = int(candidate_indices[0])
        shock_time = t[shock_idx]

        if verbose:
            logger.info(f"Shock crossing detected at t = {shock_time:.2f} s")

        return shock_time

    def get_shock_up_down_states(
        self,
        pids,
        delta_t_up=20.0,
        delta_t_down=40.0,
        b_threshold_factor=2.5,
        verbose=False,
    ):
        """
        Analyzes particles to find their state upstream and downstream of a shock.

        This function iterates through a list of particle IDs. For each particle, it
        first identifies the shock crossing time. It then calculates specific upstream
        and downstream time points based on this crossing. Finally, it interpolates
        the particle's full state (position, velocity, fields) at these two points
        and collects the results.

        Args:
            pids (list): A list of particle IDs (e.g., [(0, 1), (0, 2), ...]) to process.
            delta_t_up (float): The time in seconds *before* the shock crossing to define
                                the upstream point. Defaults to 20.0.
            delta_t_down (float): The time in seconds *after* the shock crossing to define
                                  the downstream point. Defaults to 40.0.
            b_threshold_factor (float): The sensitivity factor for shock detection, passed to
                                        `find_shock_crossing_time`. Defaults to 2.5.
            verbose (bool): If True, prints progress and individual shock detection times.
                            Defaults to False.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: A tuple containing two Polars DataFrames:
                - The first DataFrame contains the states of all valid particles at their
                  respective upstream times.
                - The second DataFrame contains the states of all valid particles at their
                  respective downstream times.
            Each DataFrame includes the original particle ID (`pid_rank`, `pid_idx`), the
            shock crossing time (`t_cross`), and the interpolated physical quantities.
            Returns (None, None) if no particles with a valid shock crossing are found.
        """
        if verbose:
            logger.info(
                f"Starting upstream/downstream analysis for {len(pids)} particles..."
            )
        upstream_states = []
        downstream_states = []
        num_particles = len(pids)

        for i, pid in enumerate(pids):
            if verbose and ((i + 1) % 500 == 0 or i == num_particles - 1):
                logger.info(
                    f"  ...processing particle {i+1}/{num_particles} (ID: {pid})"
                )

            # 1. Find the shock crossing time for the current particle
            t_cross = self.find_shock_crossing_time(
                pid, b_threshold_factor=b_threshold_factor, verbose=False
            )

            # 2. Skip particle if no shock crossing is found
            if t_cross is None:
                if verbose:
                    logger.info(
                        f"  -> No shock crossing found for particle {pid}. Skipping."
                    )
                continue

            if verbose:
                logger.info(f"  -> Shock found for {pid} at t={t_cross:.2f}s.")

            # 3. Define the upstream and downstream time points
            t_upstream = t_cross - delta_t_up
            t_downstream = t_cross + delta_t_down

            try:
                # 4. Interpolate the particle's state at the specified times
                # The result is a 2-row Polars DataFrame
                interpolated_states = interpolate_at_times(
                    self[pid], times_to_interpolate=[t_upstream, t_downstream]
                )

                # Ensure we got two valid rows back
                if interpolated_states.height != 2:
                    if verbose:
                        logger.warning(
                            f"  -> Interpolation failed for {pid}. Skipping."
                        )
                    continue

                # 5. Separate and enrich the data for collection
                up_state = interpolated_states.slice(0, 1)
                down_state = interpolated_states.slice(1, 1)

                # Add metadata (pid and shock time) to each state DataFrame
                # This makes later analysis much easier
                up_state = up_state.with_columns(
                    pl.lit(pid[0]).alias("pid_rank"),
                    pl.lit(pid[1]).alias("pid_idx"),
                    pl.lit(t_cross).alias("t_cross"),
                )
                down_state = down_state.with_columns(
                    pl.lit(pid[0]).alias("pid_rank"),
                    pl.lit(pid[1]).alias("pid_idx"),
                    pl.lit(t_cross).alias("t_cross"),
                )

                upstream_states.append(up_state)
                downstream_states.append(down_state)

            except Exception as e:
                # Catch any other errors during interpolation (e.g., times out of bounds)
                if verbose:
                    logger.error(
                        f"  -> An error occurred for particle {pid}: {e}. Skipping."
                    )
                continue

        # 6. Finalize the results
        if not upstream_states:
            if verbose:
                logger.info(
                    "\nFinished processing. No valid shock-crossing particles found."
                )
            return None, None

        # Concatenate all the individual DataFrames into two final ones
        final_upstream_df = pl.concat(upstream_states)
        final_downstream_df = pl.concat(downstream_states)

        if verbose:
            logger.info(
                f"\nFinished processing. Found {final_upstream_df.height} valid particles."
            )
        return final_upstream_df, final_downstream_df

    def _get_HT_frame(
        self, upstream_df: pl.DataFrame, downstream_df: pl.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Finds the de Hoffmann-Teller frame velocity and the shock normal vector
        using the method from Sonnerup et al. [2006], which minimizes the
        residual electric field.

        Args:
            upstream_df (pl.DataFrame): DataFrame with upstream particle states.
            downstream_df (pl.DataFrame): DataFrame with downstream particle states.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: A tuple containing:
                - V_HT (np.ndarray | None): The de Hoffmann-Teller velocity vector
                  in [km/s] if successful, otherwise None.
                - shock_normal (np.ndarray | None): The estimated shock normal
                  vector if successful, otherwise None.
        """
        all_states = pl.concat([upstream_df, downstream_df])

        # Extract E and B fields, converting to SI units if necessary
        E = all_states.select(["ex", "ey", "ez"]).to_numpy()
        B = all_states.select(["bx", "by", "bz"]).to_numpy()

        if self.unit == "planetary":
            E = E * 1e-6  # uV/m to V/m
            B = B * 1e-9  # nT to T

        # --- Calculate V_HT by solving M * V_HT = C ---
        # M_ij = sum(B^2 * delta_ij - B_i * B_j)
        M = np.sum(B**2) * np.identity(3) - (B.T @ B)

        # C = sum(E x B)
        C = np.sum(np.cross(E, B), axis=0)

        try:
            V_HT = np.linalg.solve(M, C)  # Result is in m/s
        except np.linalg.LinAlgError:
            logger.warning(
                "Could not determine de Hoffmann-Teller velocity. "
                "The matrix M is singular, which can happen if all B vectors are parallel."
            )
            return None, None

        # Convert V_HT back to km/s if using planetary units
        if self.unit == "planetary":
            V_HT /= 1e3

        # --- Calculate shock normal using magnetic coplanarity ---
        B_up_avg = np.mean(upstream_df.select(["bx", "by", "bz"]).to_numpy(), axis=0)
        B_down_avg = np.mean(
            downstream_df.select(["bx", "by", "bz"]).to_numpy(), axis=0
        )

        db = B_up_avg - B_down_avg
        n_c = np.cross(B_up_avg, B_down_avg)
        shock_normal = np.cross(db, n_c)
        norm = np.linalg.norm(shock_normal)

        if norm > 1e-9:
            shock_normal /= norm
        else:
            logger.warning(
                "Could not determine shock normal via magnetic coplanarity. "
                "Upstream and downstream average B fields may be parallel."
            )
            return V_HT, None

        return V_HT, shock_normal

    def analyze_in_HT_frame(
        self, pID: Tuple[int, int], outname: str = None, verbose: bool = False
    ):
        """
        Analyzes a particle's trajectory in the de Hoffmann-Teller (HT) frame.

        This method performs the following steps:
        1. Finds the shock crossing and determines the upstream and downstream states.
        2. Calculates the de Hoffmann-Teller velocity (V_HT) and the shock normal.
        3. Transforms the particle's velocity and the electric/magnetic fields
           into the HT frame.
        4. In this frame, the energy gain is a direct measure of non-ideal
           acceleration. It calculates and plots this energy gain.
        5. Generates a summary plot of the analysis.

        Args:
            pID (Tuple[int, int]): The particle ID (cpu, id) to analyze.
            outname (str, optional): If provided, the plot is saved to this
                                     filename. Defaults to None (displays plot).
            verbose (bool, optional): If True, prints diagnostic information.
                                      Defaults to False.
        """
        # 1. Get upstream and downstream states for the particle
        upstream_df, downstream_df = self.get_shock_up_down_states(
            [pID], verbose=verbose
        )

        if (
            upstream_df is None
            or upstream_df.is_empty()
            or downstream_df is None
            or downstream_df.is_empty()
        ):
            logger.error(
                f"Could not find valid upstream/downstream states for particle {pID}."
            )
            return

        # 2. Calculate the HT frame velocity and shock normal
        V_HT, shock_normal = self._get_HT_frame(upstream_df, downstream_df)

        if V_HT is None:
            logger.error(
                f"Could not determine the de Hoffmann-Teller frame for particle {pID}."
            )
            return

        if verbose:
            logger.info(f"Determined V_HT = {V_HT} km/s")
            if shock_normal is not None:
                logger.info(f"Estimated shock normal n = {shock_normal}")

        # 3. Get the full particle trajectory
        pt_lazy = self[pID]
        pt = pt_lazy.collect()

        # 4. Transform trajectory data into the HT frame
        v_vec = pt.select(["vx", "vy", "vz"]).to_numpy()
        e_vec = pt.select(["ex", "ey", "ez"]).to_numpy()
        b_vec = pt.select(["bx", "by", "bz"]).to_numpy()

        # Velocity in HT frame
        v_prime = v_vec - V_HT

        # Electric field in HT frame: E' = E + V_HT x B
        V_HT_si = V_HT
        b_vec_si = b_vec
        e_vec_si = e_vec
        if self.unit == "planetary":
            V_HT_si = V_HT * 1e3  # km/s to m/s
            b_vec_si = b_vec * 1e-9  # nT to T
            e_vec_si = e_vec * 1e-6  # uV/m to V/m

        vht_cross_b = np.cross(V_HT_si, b_vec_si)
        e_prime_si = e_vec_si + vht_cross_b

        e_prime = e_prime_si
        if self.unit == "planetary":
            e_prime = e_prime_si * 1e6  # Convert back to uV/m for plotting

        # Add transformed fields to the DataFrame
        pt = pt.with_columns(
            pl.Series("vx_ht", v_prime[:, 0]),
            pl.Series("vy_ht", v_prime[:, 1]),
            pl.Series("vz_ht", v_prime[:, 2]),
            pl.Series("ex_ht", e_prime[:, 0]),
            pl.Series("ey_ht", e_prime[:, 1]),
            pl.Series("ez_ht", e_prime[:, 2]),
        )

        # 5. Calculate energy and its change rate in the HT frame
        ke_ht = self.get_kinetic_energy(v_prime[:, 0], v_prime[:, 1], v_prime[:, 2])
        time = pt["time"].to_numpy()
        dke_dt_ht = np.gradient(ke_ht, time)

        # Theoretical energy gain: (q/e) * E' . V'
        v_prime_si = v_prime
        if self.unit == "planetary":
            v_prime_si = v_prime * 1e3  # km/s to m/s

        e_dot_v_ht_j_per_c = np.sum(e_prime_si * v_prime_si, axis=1)
        e_dot_v_ht = (self.charge / elementary_charge) * e_dot_v_ht_j_per_c

        # 6. Plotting
        fig, axes = plt.subplots(
            nrows=4, ncols=1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        fig.suptitle(
            f"De Hoffmann-Teller Frame Analysis for Particle {pID}", fontsize=16
        )

        time_plot = pt["time"]
        v_unit = "km/s" if self.unit == "planetary" else "m/s"
        e_unit = "μV/m" if self.unit == "planetary" else "V/m"

        axes[0].plot(time_plot, pt["vx_ht"], label="$V'_x$")
        axes[0].plot(time_plot, pt["vy_ht"], label="$V'_y$")
        axes[0].plot(time_plot, pt["vz_ht"], label="$V'_z$")
        axes[0].set_ylabel(f"V' [{v_unit}]")
        axes[0].set_title("Velocity in HT Frame")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].plot(time_plot, pt["ex_ht"], label="$E'_x$")
        axes[1].plot(time_plot, pt["ey_ht"], label="$E'_y$")
        axes[1].plot(time_plot, pt["ez_ht"], label="$E'_z$")
        axes[1].set_ylabel(f"E' [{e_unit}]")
        axes[1].set_title("Electric Field in HT Frame")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        axes[2].plot(time_plot, ke_ht, label="KE'")
        axes[2].set_ylabel("KE' [eV]")
        axes[2].set_yscale("log")
        axes[2].set_title("Kinetic Energy in HT Frame")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.6)

        axes[3].plot(time_plot, dke_dt_ht, label="d(KE')/dt")
        axes[3].plot(time_plot, e_dot_v_ht, label="q E' ⋅ V'", linestyle="--")
        axes[3].set_ylabel("Rate [eV/s]")
        axes[3].set_title("Non-ideal Energy Gain Rate")
        axes[3].legend()
        axes[3].grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel("Time [s]")

        if outname:
            plt.savefig(outname, bbox_inches="tight")
            plt.close(fig)
            if verbose:
                logger.info(f"✅ Saved HT analysis plot to {outname}")
        else:
            plt.show()

    def plot_trajectory(
        self,
        pID: Tuple[int, int],
        *,
        fscaling=1,
        smoothing_window=None,
        t_start=None,
        t_end=None,
        dt=None,
        outname=None,
        shock_time=None,
        type="quick",
        xaxis="t",
        yaxis="x",
        switchYZ=False,
        splitYZ=False,
        ax=None,
        verbose=True,
        **kwargs,
    ):
        r"""
        Plots the trajectory and velocities of the particle pID.

        Example:
        >>> tp.plot_trajectory((3,15))
        """

        def plot_data(dd, label, irow, icol):
            ax[irow, icol].plot(t, dd, label=label)
            ax[irow, icol].scatter(
                t, dd, c=plt.cm.winter(tNorm), edgecolor="none", marker="o", s=10
            )
            ax[irow, icol].set_xlabel("time")
            ax[irow, icol].set_ylabel(label)

        def plot_vector(labels, irow):
            for i, label in enumerate(labels):
                plot_data(pt[label], label, irow, i, **kwargs)

        try:
            pt = self[pID].collect()
        except (KeyError, ValueError) as e:
            logger.error(f"Error plotting trajectory for {pID}: {e}")
            return

        t = pt["time"]
        tNorm = (t - t[0]) / (t[-1] - t[0])

        if type == "single":
            x = t if xaxis == "t" else pt[xaxis]
            y = pt[yaxis]

            if ax == None:
                f, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

            ax.plot(x, y, **kwargs)
            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
        elif type == "xv":
            if ax == None:
                f, ax = plt.subplots(
                    2, 1, figsize=(10, 6), constrained_layout=True, sharex=True
                )
            y1, y2, y3 = pt["x"], pt["y"], pt["z"]

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("location")
            ax[1].set_ylabel("velocity")
            ax[0].plot(t, y1, label="x")
            ax[0].plot(t, y2, label="y")
            ax[0].plot(t, y3, label="z")

            y1, y2, y3 = pt["vx"], pt["vy"], pt["vz"]

            ax[1].plot(t, y1, label="vx")
            ax[1].plot(t, y2, label="vy")
            ax[1].plot(t, y3, label="vz")

            for a in ax:
                a.legend()
                a.grid()

        elif type == "quick":
            ncol = 3
            nrow = 3  # Default for X, V
            if self.nReal == 10:  # additional B field
                nrow = 4
            elif self.nReal >= 13:  # additional B and E field
                nrow = 5

            f, ax = plt.subplots(nrow, ncol, figsize=(12, 6), constrained_layout=True)

            # Plot trajectories
            for i, a in enumerate(ax[0, :]):
                x_label = "x" if i < 2 else "y"
                y_label = "y" if i == 0 else "z"
                a.plot(pt[x_label], pt[y_label], "k")
                a.scatter(
                    pt[x_label],
                    pt[y_label],
                    c=plt.cm.winter(tNorm),
                    edgecolor="none",
                    marker="o",
                    s=10,
                )
                a.set_xlabel(x_label)
                a.set_ylabel(y_label)

            plot_vector(["x", "y", "z"], 1)
            plot_vector(
                ["vx", "vy", "vz"],
                2,
            )

            if self.nReal > Indices.BX:
                plot_vector(
                    ["bx", "by", "bz"],
                    3,
                )

            if self.nReal > Indices.EX:
                plot_vector(
                    ["ex", "ey", "ez"],
                    4,
                )
        elif type == "full":
            if verbose:
                logger.info(f"Analyzing particle ID: {pID}")
            if dt is not None:
                t = np.arange(
                    start=pt["time"].min(),
                    stop=pt["time"].max(),
                    step=dt,
                    dtype=np.float32,
                )
                pt = interpolate_at_times(pt, t)

            # --- Time Interval Selection using Polars ---
            if t_start is not None or t_end is not None:
                start_str = f"{t_start:.2f}" if t_start is not None else "beginning"
                end_str = f"{t_end:.2f}" if t_end is not None else "end"
                logger.info(f"Slicing data from t={start_str} s to t={end_str} s")

                # Build a filter expression for the given time range
                if t_start is not None and t_end is not None:
                    pt = pt.filter(
                        (pl.col("time") >= t_start) & (pl.col("time") <= t_end)
                    )
                elif t_start is not None:
                    pt = pt.filter(pl.col("time") >= t_start)
                else:  # t_end must be not None here
                    pt = pt.filter(pl.col("time") <= t_end)

            # --- Data Extraction ---
            if self.unit == "planetary":
                t = pt["time"].to_numpy()  # [s]
                x = pt["x"].to_numpy()  # [RE]
                vx = pt["vx"].to_numpy()  # [km/s]
                bx = pt["bx"].to_numpy()  # [nT]
                ex = pt["ex"].to_numpy() * 1e-3  # [mV/m]
                if switchYZ:
                    y = pt["z"].to_numpy()  # [RE]
                    z = pt["y"].to_numpy()  # [RE]
                    vy = pt["vz"].to_numpy()  # [km/s]
                    vz = pt["vy"].to_numpy()  # [km/s]
                    by = pt["bz"].to_numpy()  # [nT]
                    bz = pt["by"].to_numpy()  # [nT]
                    ey = pt["ez"].to_numpy() * 1e-3  # [mV/m]
                    ez = pt["ey"].to_numpy() * 1e-3  # [mV/m]
                else:
                    y = pt["y"].to_numpy()  # [RE]
                    z = pt["z"].to_numpy()  # [RE]
                    vy = pt["vy"].to_numpy()  # [km/s]
                    vz = pt["vz"].to_numpy()  # [km/s]
                    by = pt["by"].to_numpy()  # [nT]
                    bz = pt["bz"].to_numpy()  # [nT]
                    ey = pt["ey"].to_numpy() * 1e-3  # [mV/m]
                    ez = pt["ez"].to_numpy() * 1e-3  # [mV/m]
            elif self.unit == "SI":
                t = pt["time"].to_numpy()  # [s]
                x = pt["x"].to_numpy()  # [m]
                vx = pt["vx"].to_numpy()  # [m/s]
                bx = pt["bx"].to_numpy()  # [T]
                ex = pt["ex"].to_numpy()  # [V/m]
                if switchYZ:
                    y = pt["z"].to_numpy()  # [m]
                    z = pt["y"].to_numpy()  # [m]
                    vy = pt["vz"].to_numpy()  # [m/s]
                    vz = pt["vy"].to_numpy()  # [m/s]
                    by = pt["bz"].to_numpy()  # [T]
                    bz = pt["by"].to_numpy()  # [T]
                    ey = pt["ez"].to_numpy()  # [V/m]
                    ez = pt["ey"].to_numpy()  # [V/m]
                else:
                    y = pt["y"].to_numpy()  # [m]
                    z = pt["z"].to_numpy()  # [m]
                    vy = pt["vy"].to_numpy()  # [m/s]
                    vz = pt["vz"].to_numpy()  # [m/s]
                    by = pt["by"].to_numpy()  # [T]
                    bz = pt["bz"].to_numpy()  # [T]
                    ey = pt["ey"].to_numpy()  # [V/m]
                    ez = pt["ez"].to_numpy()  # [V/m]

            # --- Derived Quantities Calculation ---

            # Kinetic Energy
            ke = self.get_kinetic_energy(vx, vy, vz)  # [eV]

            # --- Velocity Smoothing and Envelope Calculation ---
            if (
                smoothing_window
                and isinstance(smoothing_window, int)
                and smoothing_window > 0
            ):
                if verbose:
                    logger.info(
                        f"Applying moving average with window size: {smoothing_window}"
                    )
                # Convert numpy arrays to polars Series for easy rolling calculations
                vx_s, vy_s, vz_s = pl.Series(vx), pl.Series(vy), pl.Series(vz)
                ke_s = pl.Series(ke)

                # Calculate moving average (the smoothed line)
                vx_smooth = vx_s.rolling_mean(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vy_smooth = vy_s.rolling_mean(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vz_smooth = vz_s.rolling_mean(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                ke_smooth = ke_s.rolling_mean(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                # Calculate min/max envelopes
                vx_min_env = vx_s.rolling_min(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vx_max_env = vx_s.rolling_max(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vy_min_env = vy_s.rolling_min(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vy_max_env = vy_s.rolling_max(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vz_min_env = vz_s.rolling_min(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                vz_max_env = vz_s.rolling_max(
                    window_size=smoothing_window, center=True, min_periods=1
                )

                ke_min_env = ke_s.rolling_min(
                    window_size=smoothing_window, center=True, min_periods=1
                )
                ke_max_env = ke_s.rolling_max(
                    window_size=smoothing_window, center=True, min_periods=1
                )

            v_vec = np.vstack((vx, vy, vz)).T
            b_vec = np.vstack((bx, by, bz)).T
            e_vec = np.vstack((ex, ey, ez)).T
            # Calculate magnitudes of vectors
            v_mag = np.linalg.norm(v_vec, axis=1)
            b_mag = np.linalg.norm(b_vec, axis=1)
            e_mag = np.linalg.norm(e_vec, axis=1)

            # Pitch Angle Calculation
            v_dot_b = np.sum(v_vec * b_vec, axis=1)
            epsilon = 1e-15
            cos_alpha = v_dot_b / (v_mag * b_mag + epsilon)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            pitch_angle_rad = np.arccos(cos_alpha)
            pitch_angle = pitch_angle_rad * 180.0 / np.pi

            if self.unit == "planetary":
                # Magnetic Field Energy Density Calculation
                U_B = (b_mag * 1e-9) ** 2 / (2 * mu_0 * elementary_charge)  # [eV/m^3]
                # Electric Field Energy Density Calculation
                U_E = (
                    0.5 * epsilon_0 * (e_mag * 1e-3) ** 2 / elementary_charge
                )  # [eV/m^3]
                # First Adiabatic Invariant (mu) Calculation
                # mu = mv_perp^2 / 2B.  v_perp = v * sin(alpha)
                # Ensure units are SI: v [m/s], B [T]
                # Perpendicular velocity in SI units [m/s]
                v_perp = v_mag * 1e3 * np.sin(pitch_angle_rad)
                # Calculate mu, handle potential division by zero in B
                mu = (0.5 * self.mass * v_perp**2) / (b_mag * 1e-9)  # [J/T]
                # Gyrofrequency in Hz
                gyro_freq = (
                    (elementary_charge * b_mag)
                    / (2 * np.pi * self.mass)
                    * 1e-9
                    / fscaling
                )
                # Gyroradius in km
                gyro_radius = (
                    (self.mass * v_perp) / (elementary_charge * b_mag) * 1e6 * fscaling
                )
            elif self.unit == "SI":
                # Magnetic Field Energy Density Calculation
                U_B = b_mag**2 / (2 * mu_0 * elementary_charge)  # [eV/m^3]
                # Electric Field Energy Density Calculation
                U_E = 0.5 * epsilon_0 * e_mag**2 / elementary_charge  # [eV/m^3]
                # First Adiabatic Invariant (mu) Calculation
                v_perp = v_mag * np.sin(pitch_angle_rad)
                # Calculate mu, handle potential division by zero in B
                mu = (0.5 * self.mass * v_perp**2) / b_mag  # [J/T]
                # Gyrofrequency in Hz
                gyro_freq = (
                    (elementary_charge * b_mag) / (2 * np.pi * self.mass) / fscaling
                )
                # Gyroradius in km
                gyro_radius = (
                    (self.mass * v_perp) / (elementary_charge * b_mag) * 1e3 * fscaling
                )

            # --- Plotting ---
            f, ax = plt.subplots(
                8, 1, figsize=(10, 12), constrained_layout=True, sharex=True
            )

            # Panel 0: Particle Location
            ax[0].plot(t, x, label="x")
            if splitYZ:
                ax0_twin = ax[0].twinx()
                ax0_twin.plot(t, z, label="z", color="tab:orange")
                ax0_twin.tick_params(axis="y", labelcolor="tab:orange")
                if self.unit == "planetary":
                    ax[0].set_ylabel(r"X [$R_E$]", fontsize=14)
                    ax0_twin.set_ylabel(r"Z [$R_E$]", fontsize=14, color="tab:orange")
                elif self.unit == "SI":
                    ax[0].set_ylabel("X [m]", fontsize=14)
                    ax0_twin.set_ylabel("Z [m]", fontsize=14, color="tab:orange")
            else:
                if self.unit == "planetary":
                    ax[0].set_ylabel(r"Location [$R_E$]", fontsize=14)
                elif self.unit == "SI":
                    ax[0].set_ylabel("Location [m]", fontsize=14)
                ax[0].plot(t, y, label="y")
                ax[0].plot(t, z, label="z")

            # Panel 1: Particle Velocity
            if self.unit == "planetary":
                ax[1].set_ylabel("V [km/s]", fontsize=14)
            elif self.unit == "SI":
                ax[1].set_ylabel("V [m/s]", fontsize=14)

            # If smoothing is enabled, plot the smoothed lines and envelopes
            if (
                smoothing_window
                and isinstance(smoothing_window, int)
                and smoothing_window > 0
            ):
                # Plot smoothed lines with a thicker, more prominent style
                ax[1].plot(t, vx_smooth, color="tab:blue", linewidth=1.5, label="$V_x$")
                ax[1].plot(
                    t, vy_smooth, color="tab:orange", linewidth=1.5, label="$V_y$"
                )
                ax[1].plot(
                    t, vz_smooth, color="tab:green", linewidth=1.5, label="$V_z$"
                )

                # Shade the area between the min and max envelopes
                ax[1].fill_between(
                    t, vx_min_env, vx_max_env, color="tab:blue", alpha=0.2
                )
                ax[1].fill_between(
                    t, vy_min_env, vy_max_env, color="tab:orange", alpha=0.2
                )
                ax[1].fill_between(
                    t, vz_min_env, vz_max_env, color="tab:green", alpha=0.2
                )
            else:
                ax[1].plot(t, vx, label="$V_x$", color="tab:blue", alpha=0.9)
                ax[1].plot(t, vy, label="$V_y$", color="tab:orange", alpha=0.9)
                ax[1].plot(t, vz, label="$V_z$", color="tab:green", alpha=0.9)

            # Panel 2: Kinetic Energy
            ax[2].set_ylabel("KE [eV]", fontsize=14)
            ax[2].set_yscale("log")
            if (
                smoothing_window
                and isinstance(smoothing_window, int)
                and smoothing_window > 0
            ):
                ax[2].plot(
                    t, ke_smooth, color="tab:brown", linewidth=1.5, label="KE (smooth)"
                )
                ax[2].fill_between(
                    t, ke_min_env, ke_max_env, color="tab:red", alpha=0.2
                )
            else:
                ax[2].plot(t, ke, label="KE", color="tab:brown")

            # Panel 3: Field Energy Densities (on twin axes)
            ax[3].plot(t, U_B, label=r"$U_B$", color="tab:red")
            ax[3].set_ylabel(r"$U_B$ [eV/m$^3$]", fontsize=14, color="tab:red")
            ax[3].tick_params(axis="y", labelcolor="tab:red")

            ax3_twin = ax[3].twinx()
            ax3_twin.plot(t, U_E, label=r"$U_E$", color="tab:purple")
            ax3_twin.set_ylabel(r"$U_E$ [eV/m$^3$]", fontsize=14, color="tab:purple")
            ax3_twin.tick_params(axis="y", labelcolor="tab:purple")

            # Panel 4: Magnetic Field
            ax[4].plot(t, bx, label="$B_x$")
            ax[4].plot(t, by, label="$B_y$")
            ax[4].plot(t, bz, label="$B_z$")
            ax[4].plot(t, b_mag, "k--", label="$B$")
            if self.unit == "planetary":
                ax[4].set_ylabel("B [nT]", fontsize=14)
            elif self.unit == "SI":
                ax[4].set_ylabel("B [T]", fontsize=14)

            # Panel 5: Electric Field
            ax[5].plot(t, ex, label="$E_x$")
            ax[5].plot(t, ey, label="$E_y$")
            ax[5].plot(t, ez, label="$E_z$")
            if self.unit == "planetary":
                ax[5].set_ylabel("E [mV/m]", fontsize=14)
            elif self.unit == "SI":
                ax[5].set_ylabel("E [V/m]", fontsize=14)

            # Panel 6: Pitch Angle
            ax[6].plot(t, pitch_angle, color="tab:brown")
            ax[6].set_ylabel(r"$\alpha$ [$^\circ$]", fontsize=14)
            ax[6].set_ylim(0, 180)
            ax[6].set_yticks([0, 45, 90, 135, 180])

            # Create segments for the line
            points = np.array([t, pitch_angle]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a LineCollection, coloring by gyroradius
            norm_rg = Normalize(gyro_radius.min(), gyro_radius.max())
            lc_rg = LineCollection(segments, cmap="cividis", norm=norm_rg)
            lc_rg.set_array(gyro_radius)
            lc_rg.set_linewidth(2)
            line_rg = ax[6].add_collection(lc_rg)
            # Add a color bar
            cbar_rg = f.colorbar(line_rg, ax=ax[6], pad=-0.051)
            cbar_rg.set_label(r"$r_L$ [km]", fontsize=12)

            # Panel 7: First Adiabatic Invariant
            ax[7].plot(t, mu, color="tab:brown")
            ax[7].set_ylabel(r"$\mu$ [J/T]", fontsize=14)
            ax[7].set_yscale("log")  # mu can vary, log scale is often useful
            # Create segments for the line
            points = np.array([t, mu]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a LineCollection, coloring by gyrofrequency
            norm_gf = Normalize(gyro_freq.min(), gyro_freq.max())
            lc_gf = LineCollection(segments, cmap="plasma", norm=norm_gf)
            lc_gf.set_array(gyro_freq)
            lc_gf.set_linewidth(2)
            line_gf = ax[7].add_collection(lc_gf)
            # Add a color bar
            cbar_gf = f.colorbar(line_gf, ax=ax[7], pad=-0.051)
            cbar_gf.set_label(r"$f_{ci}$ [Hz]", fontsize=12)

            # --- Add Shock Crossing Line if Provided ---
            if shock_time is not None:
                for a in ax:
                    a.axvline(
                        x=shock_time,
                        color="tab:cyan",
                        linestyle="--",
                        linewidth=1.5,
                    )

            # --- Decorations ---
            ax[-1].set_xlabel("t [s]", fontsize=14)
            for i, a in enumerate(ax):
                a.tick_params(axis="both", which="major", labelsize="medium")
                a.grid(True, which="both", linestyle="--", linewidth=0.5)
                a.set_xlim(left=t.min(), right=t.max())
                # Adjust legends
                if i in [0, 1, 5]:
                    a.legend(ncols=3, loc="best", framealpha=0.5, fontsize="large")
                elif i == 4:
                    a.legend(ncols=4, loc="best", framealpha=0.5, fontsize="large")

            f.suptitle(f"Test Particle ID: {pID}", fontsize=16)

        if outname:
            plt.savefig(outname, dpi=200, bbox_inches="tight")
            plt.close(f)
            if verbose:
                logger.info(f"✅ Saved figure to {outname}...")
        else:
            plt.show()
            return ax

    def plot_location(self, pData: np.ndarray):
        """
        Plot the location of particles pData.

        Examples:
        >>> ids, pData = tp.read_particles_at_time(3700, doSave=True)
        >>> f = tp.plot_location(pData)
        """

        px = pData[:, Indices.X]
        py = pData[:, Indices.Y]
        pz = pData[:, Indices.Z]

        # Create subplot mosaic with different keyword arguments
        skeys = ["A", "B", "C", "D"]
        f, ax = plt.subplot_mosaic(
            "AB;CD",
            per_subplot_kw={("D"): {"projection": "3d"}},
            gridspec_kw={"width_ratios": [1, 1], "wspace": 0.1, "hspace": 0.1},
            figsize=(10, 10),
            constrained_layout=True,
        )

        # Create 2D scatter plots
        for i, (x, y, labels) in enumerate(
            zip([px, px, py], [py, pz, pz], [("x", "y"), ("x", "z"), ("y", "z")])
        ):
            ax[skeys[i]].scatter(x, y, s=1)
            ax[skeys[i]].set_xlabel(labels[0])
            ax[skeys[i]].set_ylabel(labels[1])

        # Create 3D scatter plot
        ax[skeys[3]].scatter(px, py, pz, s=1)
        ax[skeys[3]].set_xlabel("x")
        ax[skeys[3]].set_ylabel("y")
        ax[skeys[3]].set_zlabel("z")

        return ax

    def _calculate_true_gc_trajectory(
        self,
        pt: pl.DataFrame,
        smoothing_gyro_periods: float,
    ) -> pl.DataFrame:
        """Calculates the 'true' guiding center trajectory by smoothing."""
        b_mag = (pt["bx"] ** 2 + pt["by"] ** 2 + pt["bz"] ** 2).sqrt()

        if self.unit == "planetary":
            b_mag_si = b_mag * 1e-9  # convert nT to T
        else:  # SI
            b_mag_si = b_mag

        epsilon = 1e-20
        omega_c = (abs(self.charge) * b_mag_si) / self.mass
        gyro_period = (2 * np.pi) / (omega_c + epsilon)

        time_steps = pt["time"].diff().mean()
        if time_steps is None or time_steps == 0:
            logger.warning(
                "Could not determine a valid time step. Cannot apply smoothing."
            )
            return pt.select(
                pl.col("x").alias("x_true"),
                pl.col("y").alias("y_true"),
                pl.col("z").alias("z_true"),
            )

        avg_gyro_period = gyro_period.mean()
        window_size = int(
            np.ceil((avg_gyro_period * smoothing_gyro_periods) / time_steps)
        )
        window_size = max(1, window_size if window_size % 2 != 0 else window_size + 1)

        logger.info(f"Average gyro-period: {avg_gyro_period:.4f} s")
        logger.info(f"Time step: {time_steps:.4f} s")
        logger.info(f"Using moving average window size: {window_size}")

        return pt.select(
            pl.col("x")
            .rolling_mean(window_size=window_size, center=True)
            .alias("x_true"),
            pl.col("y")
            .rolling_mean(window_size=window_size, center=True)
            .alias("y_true"),
            pl.col("z")
            .rolling_mean(window_size=window_size, center=True)
            .alias("z_true"),
        )

    @staticmethod
    def _integrate_velocity(
        v_series: pl.Series, initial_pos_series: pl.Series, dt_series: pl.Series
    ) -> pl.Series:
        """Integrates a velocity series using the trapezoidal rule."""
        initial_pos = initial_pos_series[0]
        integrated_pos = (
            initial_pos + (((v_series + v_series.shift(1)) / 2) * dt_series).cum_sum()
        )
        return integrated_pos.fill_null(initial_pos)

    def _calculate_predicted_gc_trajectory(
        self,
        pt: pl.DataFrame,
        pID: Tuple[int, int],
    ) -> Tuple[pl.Series, pl.Series, pl.Series]:
        """Calculates the predicted guiding center trajectory from theory."""
        pt_lazy = pt.lazy()
        ve = self.get_ExB_drift(pt_lazy)
        vg = self.get_gradient_drift(pt_lazy)
        vc = self.get_curvature_drift(pt_lazy)
        vp = self.get_polarization_drift(pt_lazy)

        epsilon = 1e-20
        b_mag = (pt["bx"] ** 2 + pt["by"] ** 2 + pt["bz"] ** 2).sqrt()
        b_mag_with_fallback = b_mag.fill_null(1.0).clip(lower_bound=epsilon)
        b_unit_x = pt["bx"] / b_mag_with_fallback
        b_unit_y = pt["by"] / b_mag_with_fallback
        b_unit_z = pt["bz"] / b_mag_with_fallback

        v_dot_b = pt["vx"] * b_unit_x + pt["vy"] * b_unit_y + pt["vz"] * b_unit_z

        v_parallel_x = v_dot_b * b_unit_x
        v_parallel_y = v_dot_b * b_unit_y
        v_parallel_z = v_dot_b * b_unit_z

        v_gc_x = v_parallel_x + ve["vex"] + vg["vgx"] + vc["vcx"] + vp["vpx"]
        v_gc_y = v_parallel_y + ve["vey"] + vg["vgy"] + vc["vcy"] + vp["vpy"]
        v_gc_z = v_parallel_z + ve["vez"] + vg["vgz"] + vc["vcz"] + vp["vpz"]

        dt = pt["time"].diff().fill_null(0)

        pos_gc_pred_x = self._integrate_velocity(v_gc_x, pt["x"], dt)
        pos_gc_pred_y = self._integrate_velocity(v_gc_y, pt["y"], dt)
        pos_gc_pred_z = self._integrate_velocity(v_gc_z, pt["z"], dt)

        return pos_gc_pred_x, pos_gc_pred_y, pos_gc_pred_z

    def _plot_gc_verification(
        self,
        pt: pl.DataFrame,
        pos_gc_true: pl.DataFrame,
        pos_gc_pred: Tuple[pl.Series, pl.Series, pl.Series],
        pID: Tuple[int, int],
    ):
        """Plots the verification results."""
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        fig.suptitle(
            f"Guiding Center Model Verification for Particle ID: {pID}", fontsize=16
        )
        time = pt["time"]

        coord_map = {0: ("x", "X"), 1: ("y", "Y"), 2: ("z", "Z")}

        for i, ax in enumerate(axes):
            coord, label = coord_map[i]

            ax.plot(
                time,
                pt[coord],
                label=f"Full Trajectory ({label})",
                color="gray",
                alpha=0.7,
            )
            ax.plot(
                time,
                pos_gc_true[f"{coord}_true"],
                label="'True' GC (Smoothed)",
                color="black",
                linestyle="--",
                linewidth=2,
            )
            ax.plot(
                time,
                pos_gc_pred[i],
                label="Predicted GC (Integrated)",
                color="red",
                linestyle=":",
                linewidth=2,
            )

            if self.unit == "planetary":
                ax.set_ylabel("Position [$R_E$]")
            else:
                ax.set_ylabel("Position [m]")

            ax.set_title(f"{label}-Component")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel("Time [s]")
        plt.show()

    def verify_guiding_center_model(
        self,
        pID: Tuple[int, int],
        smoothing_gyro_periods=1.0,
    ):
        """
        Verifies the guiding center model by comparing it against the full
        particle trajectory.

        This method performs the following steps:
        1. Calculates a "true" guiding center trajectory by applying a low-pass
           filter (moving average) to the full particle trajectory, smoothing
           out the gyromotion.
        2. Calculates a "predicted" guiding center trajectory by numerically
           integrating the guiding center velocity, which is the sum of the
           parallel velocity and all perpendicular drift velocities (E x B,
           gradient, curvature, and polarization).
        3. Generates a plot comparing the full trajectory, the "true" GC
           trajectory, and the "predicted" GC trajectory for each coordinate (X, Y, Z).

        A close overlap between the "true" and "predicted" trajectories
        validates the guiding center approximation and the drift calculations.

        Args:
            pID (Tuple[int, int]): The ID of the particle to analyze.
            smoothing_gyro_periods (float): The size of the moving average window
                                            in units of gyro-periods. Defaults to 1.0.
        """
        try:
            pt = self[pID].collect()
        except (KeyError, ValueError) as e:
            logger.error(f"Error getting trajectory for {pID}: {e}")
            return

        logger.info(f"Verifying guiding center model for particle ID: {pID}")

        # 1. Calculate "true" guiding center trajectory
        pos_gc_true = self._calculate_true_gc_trajectory(pt, smoothing_gyro_periods)

        # 2. Calculate predicted guiding center trajectory
        pos_gc_pred = self._calculate_predicted_gc_trajectory(pt, pID)

        # 3. Plot the comparison
        self._plot_gc_verification(pt, pos_gc_true, pos_gc_pred, pID)


def interpolate_at_times(
    df: Union[pl.DataFrame, pl.LazyFrame], times_to_interpolate: list[float]
) -> pl.DataFrame:
    """
    Interpolates multiple numeric columns of a DataFrame at specified time points.

    Args:
        df: The input Polars DataFrame or LazyFrame.
        times_to_interpolate: A list of time points (floats or ints) at which to interpolate.

    Returns:
        A new DataFrame containing the interpolated rows for each specified time.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Identify all numeric columns to be interpolated
    cols_to_interpolate = df.select(pl.col(pl.NUMERIC_DTYPES).exclude("time")).columns

    time_col_dtype = df.get_column("time").dtype

    null_rows_df = pl.DataFrame(
        {
            "time": times_to_interpolate,
            **{col: [None] * len(times_to_interpolate) for col in cols_to_interpolate},
        }
    ).with_columns(pl.col("time").cast(time_col_dtype))

    df_all = pl.concat([df, null_rows_df]).sort("time")

    # Create a Datetime Series to use for interpolation.
    time_dt_series = pl.from_epoch(
        (df_all.get_column("time") * 1_000_000).cast(pl.Int64), time_unit="us"
    )

    interpolated_df = df_all.with_columns(
        pl.col(cols_to_interpolate).interpolate_by(time_dt_series)
    ).filter(pl.col("time").is_in(times_to_interpolate))

    return interpolated_df


def plot_integrated_energy(df: pl.DataFrame, outname=None, **kwargs):
    """
    Plots integrated energy quantities as a function of time.

    Args:
        df (pl.DataFrame): A Polars DataFrame containing a time column and
                           one or more integrated energy columns.
        outname (str): If not None, save the plot to file.
    """
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    time_data = df["time"].to_numpy()
    energy_columns = [col for col in df.columns if col != "time"]

    for column_name in energy_columns:
        energy_data = df[column_name].to_numpy()

        if column_name == "W_parallel_integrated":
            label = r"$\text{W}_\parallel$"
        elif column_name == "W_betatron_integrated":
            label = r"$\text{W}_\text{betatron}$"
        elif column_name == "Wp_integrated":
            label = r"$\text{W}_p$"
        elif column_name == "ke":
            energy_data = energy_data - energy_data[0]
            label = r"$\Delta$KE"
        else:
            label = column_name.replace("_integrated", "")

        ax.plot(time_data, energy_data, label=label, linewidth=2.5)

    # Check if all required columns for the sum are present
    required_cols = [
        "Wg_integrated",
        "Wc_integrated",
        "Wp_integrated",
        "W_parallel_integrated",
        "W_betatron_integrated",
        "ke",
    ]
    if all(col in df.columns for col in required_cols):
        w_sum = (
            df["Wg_integrated"]
            + df["Wc_integrated"]
            + df["Wp_integrated"]
            + df["W_parallel_integrated"]
            + df["W_betatron_integrated"]
        ).to_numpy()
        ax.plot(
            time_data,
            w_sum,
            label=r"$\text{W}_\text{sum}$",
            linewidth=2.5,
            linestyle="--",
        )

        ke_data = df["ke"].to_numpy()
        delta_ke = ke_data - ke_data[0]
        non_adiabatic_heating = delta_ke - w_sum
        ax.plot(
            time_data,
            non_adiabatic_heating,
            label="Non-adiabatic",
            linewidth=0.7,
            linestyle=":",
        )

    # Customize the plot
    ax.grid(True)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Integrated Energy (eV)", fontsize=14)
    ax.set_title("Cumulative Energy Change Over Time", fontsize=16, fontweight="bold")
    ax.legend(title="Energy Source", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)

    if outname is not None:
        plt.savefig(outname, bbox_inches="tight")
    else:
        plt.show()
