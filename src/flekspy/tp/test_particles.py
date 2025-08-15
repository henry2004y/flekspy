from typing import List, Tuple, Dict, Union, Callable

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import polars as pl
import glob
import struct
from itertools import islice
from enum import IntEnum
from scipy.constants import proton_mass, elementary_charge, mu_0, epsilon_0


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
    >>> tp.save_trajectory_to_csv(tp.IDs[5])
    >>> ids, pData = tp.read_particles_at_time(0.0, doSave=False)
    >>> f = tp.plot_location(pData)
    """

    def __init__(
        self,
        dirs: Union[str, List[str]],
        iDomain: int = 0,
        iSpecies: int = 0,
        iListStart: int = 0,
        iListEnd: int = -1,
        readAllFiles: bool = False,
        use_cache: bool = False,
    ):
        self.use_cache = use_cache
        self._trajectory_cache = {}

        if isinstance(dirs, str):
            dirs = [dirs]

        header = Path(dirs[0] + "/Header")
        if header.exists():
            with open(header, "r") as f:
                self.nReal = int(f.readline())
        else:
            raise FileNotFoundError(f"Header file not found in {dirs[0]}")

        self.iSpecies = iSpecies
        plistfiles = list()
        self.pfiles = list()

        for outputDir in dirs:
            plistfiles.extend(
                glob.glob(
                    f"{outputDir}/FLEKS{iDomain}_particle_list_species_{iSpecies}_*"
                )
            )

            self.pfiles.extend(
                glob.glob(f"{outputDir}/FLEKS{iDomain}_particle_species_{iSpecies}_*")
            )

        plistfiles.sort()
        self.pfiles.sort()

        self.indextotime = []
        if readAllFiles:
            for filename in self.pfiles:
                record = self._read_the_first_record(filename)
                if record is None:
                    continue
                self.indextotime.append(record[Indices.TIME])

        if iListEnd == -1:
            iListEnd = len(plistfiles)
        plistfiles = plistfiles[iListStart:iListEnd]
        self.pfiles = self.pfiles[iListStart:iListEnd]

        self.particle_locations: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        for plist_filename, p_filename in zip(plistfiles, self.pfiles):
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
            f"Number of particles : {len(self.IDs)}\n"
            f"First time tag      : {self.filetime[0]}\n"
            f"Last  time tag      : {self.filetime[-1]}\n"
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

    def save_trajectory_to_csv(
        self,
        pID: Tuple[int, int],
        unit: str = "planetary",
        filename: str = None,
        shiftTime: bool = False,
        scaleTime: bool = False,
    ) -> None:
        """
        Save the trajectory of a particle to a csv file.

        Args:
            pID: particle ID.
            shiftTime (bool): If set to True, set the initial time to be 0.
            scaleTime (bool): If set to True, scale the time into [0,1] range.

        Example:
        >>> tp.save_trajectory_to_csv((3,15))
        """
        pData_lazy = self[pID]
        if filename is None:
            filename = f"trajectory_{pID[0]}_{pID[1]}.csv"

        if unit == "planetary":
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
        elif unit == "SI":
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
            pData_to_save.sink_csv(filename)
        except (IOError, pl.exceptions.PolarsError) as e:
            print(f"Error saving trajectory to CSV: {e}")

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

    @staticmethod
    def get_kinetic_energy(vx, vy, vz, mass=proton_mass):
        # Assume velocity in units of [m/s]
        ke = 0.5 * mass * (vx**2 + vy**2 + vz**2) / elementary_charge  # [eV]

        return ke

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

    def get_first_adiabatic_invariant(self, pID, mass=proton_mass):
        """
        Calculates the 1st adiabatic invariant of a particle.
        """
        pt_lazy = self[pID]
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
        mu_expr = ((0.5 * mass * v_perp_sq) / (b_mag_expr + epsilon)).alias("mu")

        # Execute the expression and return the result
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

    def get_ExB_drift(self, pID: Tuple[int, int]) -> pl.DataFrame:
        """
        Calculates the convection drift velocity for a particle.
        v_exb = E x B / (B^2)
        Assuming Earth's planetary units, output drift velocity in [km/s].
        """
        pt_lazy = self[pID]
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
        pID: Tuple[int, int],
        unit="planetary",
        mass=proton_mass,
        charge=elementary_charge,
    ) -> pl.DataFrame:
        """
        Calculates the curvature drift velocity for a particle.
        v_c = (m * v_parallel^2 / (q*B^2)) * (B x κ)
        Depending on the selected units, output drift velocity may be
        - "planetary": [km/s]
        - "SI": [m/s]
        """
        pt_lazy = self[pID]
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
        if unit == "planetary":
            factor = (mass * v_parallel_sq) / (charge * b_mag_sq) * 1e9 / 6378
        elif unit == "SI":
            factor = (mass * v_parallel_sq) / (charge * b_mag_sq)
        else:
            raise ValueError(f"Unknown unit: '{unit}'. Must be 'planetary' or 'SI'.")

        lf = lf.with_columns(
            vcx=factor * cross_x, vcy=factor * cross_y, vcz=factor * cross_z
        )

        return lf.select(["vcx", "vcy", "vcz"]).collect()

    def get_gyroradius_to_curvature_ratio(
        self,
        pID: Tuple[int, int],
        unit="planetary",
        mass=proton_mass,
        charge=elementary_charge,
    ) -> pl.Series:
        """
        Calculates the ratio of the particle's gyroradius to the magnetic
        field's radius of curvature.
        """
        pt_lazy = self[pID]
        epsilon = 1e-15

        # Expression for v_perp
        v_mag_sq = pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2
        b_mag = (pl.col("bx") ** 2 + pl.col("by") ** 2 + pl.col("bz") ** 2).sqrt()
        v_dot_b = (
            pl.col("vx") * pl.col("bx")
            + pl.col("vy") * pl.col("by")
            + pl.col("vz") * pl.col("bz")
        )
        sin_alpha_sq = 1 - (v_dot_b / (v_mag_sq.sqrt() * b_mag + epsilon)) ** 2
        v_perp = (v_mag_sq * sin_alpha_sq).sqrt()

        # Expression for gyroradius
        r_g = (mass * v_perp) / (abs(charge) * b_mag) * 1e9  # [km]

        # Expression for curvature radius
        lf_curv = self._calculate_curvature(pt_lazy)
        kappa_mag = (
            pl.col("kappa_x") ** 2 + pl.col("kappa_y") ** 2 + pl.col("kappa_z") ** 2
        ).sqrt()

        if unit == "planetary":
            factor = 6378  # conversion factor
        elif unit == "SI":
            factor = 1e-3  # conversion factor
        else:
            raise ValueError(f"Unknown unit: '{unit}'. Must be 'planetary' or 'SI'.")
        r_c = (1 / (kappa_mag + epsilon)) * factor  # [km]

        ratio_expr = (r_g / r_c).alias("ratio")

        return lf_curv.select(ratio_expr).collect().to_series()

    def get_gradient_drift(
        self,
        pID: Tuple[int, int],
        unit="planetary",
        mass=proton_mass,
        charge=elementary_charge,
    ) -> pl.DataFrame:
        """
        Calculates the gradient drift velocity for a particle.
        v_g = (μ / (q * B^2)) * (B x ∇|B|)
        Depending on the selected units, output drift velocity may be
        - "planetary": [km/s]
        - "SI": [m/s]
        """
        pt_lazy = self[pID]
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
        mu_expr = (0.5 * mass * v_perp_sq) / (b_mag + epsilon)

        lf = self._calculate_bmag(pt_lazy)

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

        lf = lf.with_columns(
            grad_b_mag_x=grad_b_mag_x,
            grad_b_mag_y=grad_b_mag_y,
            grad_b_mag_z=grad_b_mag_z,
        )

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
        if unit == "planetary":
            factor = mu_expr / (charge * b_mag_sq) * 1e9 / 6378
        elif unit == "SI":
            factor = mu_expr / (charge * b_mag_sq)
        else:
            raise ValueError(f"Unknown unit: '{unit}'. Must be 'planetary' or 'SI'.")

        lf = lf.with_columns(
            vgx=factor * cross_x, vgy=factor * cross_y, vgz=factor * cross_z
        )

        return lf.select(["vgx", "vgy", "vgz"]).collect()

    def plot_trajectory(
        self,
        pID: Tuple[int, int],
        *,
        mass=proton_mass,
        type="quick",
        xaxis="t",
        yaxis="x",
        ax=None,
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
            print(f"Error plotting trajectory for {pID}: {e}")
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
            print(f"Analyzing particle ID: {pID}")
            # TODO Proper unit handling
            # --- Data Extraction ---
            t = pt["time"]  # [s]
            x, y, z = pt["x"], pt["y"], pt["z"]  # [RE]
            vx, vy, vz = pt["vx"], pt["vy"], pt["vz"]  # [km/s]
            bx, by, bz = pt["bx"], pt["by"], pt["bz"]  # [nT]
            ex, ey, ez = pt["ex"], pt["ey"], pt["ez"]  # [muV/m]

            # --- Derived Quantities Calculation ---

            # Kinetic Energy
            ke = self.get_kinetic_energy(vx, vy, vz) * 1e6  # [eV]

            # Vectorize B and V fields for easier calculations
            b_vec = pt.select(["bx", "by", "bz"]).to_numpy()
            e_vec = pt.select(["ex", "ey", "ez"]).to_numpy()

            # Calculate magnitudes of vectors
            b_mag = np.linalg.norm(b_vec, axis=1)  # [nT]
            e_mag = np.linalg.norm(e_vec, axis=1) * 1e-3  # [mV/m]

            # Magnetic Field Energy Density Calculation
            U_B = (b_mag * 1e-9) ** 2 / (2 * mu_0)  # [J/m^3]

            # Electric Field Energy Density Calculation
            U_E = 0.5 * epsilon_0 * (e_mag * 1e-3) ** 2  # [J/m^3]

            # Pitch Angle Calculation
            pitch_angle = self.get_pitch_angle_from_v_b(vx, vy, vz, bx, by, bz)

            # --- First Adiabatic Invariant (mu) Calculation ---
            # mu = mv_perp^2 / 2B.  v_perp = v * sin(alpha)
            mu = self.get_first_adiabatic_invariant(pID, mass=mass) * 1e9  # [J/T]

            # --- Plotting ---
            # Create 7 subplots, sharing the x-axis
            f, ax = plt.subplots(
                8, 1, figsize=(10, 12), constrained_layout=True, sharex=True
            )

            # Panel 0: Particle Location
            ax[0].set_ylabel(r"Location [$R_E$]", fontsize=14)
            ax[0].plot(t, x, label="x")
            ax[0].plot(t, y, label="y")
            ax[0].plot(t, z, label="z")

            # Panel 1: Particle Velocity
            ax[1].set_ylabel("V [km/s]", fontsize=14)
            ax[1].plot(t, vx, label="$v_x$")
            ax[1].plot(t, vy, label="$v_y$")
            ax[1].plot(t, vz, label="$v_z$")

            # Panel 2: Kinetic Energy
            ax[2].plot(t, ke, label="KE", color="tab:brown")
            ax[2].set_ylabel("KE [eV]", fontsize=14)
            ax[2].set_yscale("log")

            # Panel 3: Field Energy Densities (on twin axes)
            ax[3].plot(t, U_B, label=r"$U_B$", color="tab:red")
            ax[3].set_ylabel(r"$U_B$ [J/m$^3$]", fontsize=14, color="tab:red")
            # ax[3].set_yscale("log")
            ax[3].tick_params(axis="y", labelcolor="tab:red")

            ax3_twin = ax[3].twinx()
            ax3_twin.plot(t, U_E, label=r"$U_E$", color="tab:purple")
            ax3_twin.set_ylabel(r"$U_E$ [J/m$^3$]", fontsize=14, color="tab:purple")
            # ax3_twin.set_yscale("log")
            ax3_twin.tick_params(axis="y", labelcolor="tab:purple")

            # Panel 4: Magnetic Field
            ax[4].plot(t, bx, label="$B_x$")
            ax[4].plot(t, by, label="$B_y$")
            ax[4].plot(t, bz, label="$B_z$")
            ax[4].set_ylabel("B [nT]", fontsize=14)

            # Panel 5: Electric Field
            ax[5].plot(t, ex, label="$E_x$")
            ax[5].plot(t, ey, label="$E_y$")
            ax[5].plot(t, ez, label="$E_z$")
            ax[5].set_ylabel("E [mV/m]", fontsize=14)

            # Panel 6: Pitch Angle
            ax[6].plot(t, pitch_angle, color="tab:brown")
            ax[6].set_ylabel(r"Pitch Angle [$^\circ$]", fontsize=14)
            ax[6].set_ylim(0, 180)
            ax[6].set_yticks([0, 45, 90, 135, 180])

            # Panel 7: First Adiabatic Invariant
            ax[7].plot(t, mu, color="tab:brown")
            ax[7].set_ylabel(r"$\mu$ [J/T]", fontsize=14)
            ax[7].set_yscale("log")  # mu can vary, log scale is often useful

            # --- Decorations ---
            ax[-1].set_xlabel("t [s]", fontsize=14)

            # Add legends and grid to all plots
            for i, a in enumerate(ax):
                # Set tick label size for all axes
                a.tick_params(axis="both", which="major", labelsize="medium")

                # Skip legend for the last panel (pitch angle) as it only has one line
                if i == 6:
                    a.grid(True, which="both", linestyle="--", linewidth=0.5)
                    a.set_xlim(left=t.min(), right=t.max())
                    continue

                # For panels with 3 items, arrange legend in 3 columns
                if i in [0, 1, 4, 5]:
                    a.legend(ncols=3, loc="best", fontsize="large")
                else:
                    pass

                a.grid(True, which="both", linestyle="--", linewidth=0.5)
                a.set_xlim(left=t.min(), right=t.max())

            f.suptitle(f"Test Particle ID: {pID}", fontsize=16)

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
