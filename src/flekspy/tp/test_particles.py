from typing import List, Tuple, Dict, Union, Callable

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import glob
import struct
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


class ParticleTrajectory:
    """
    A class to store particle trajectory data.
    Data can be accessed directly using dictionary-like keys.

    Examples:
    >>> pt = tp.read_particle_trajectory((66,888))
    >>> time = pt["t"]
    >>> x_position = pt["x"]
    >>> x_vel, y_vel, z_vel = pt["velocity"]
    """

    def __init__(self, particle_id: Tuple[int, int], trajectory_data: np.ndarray):
        """
        Args:
            particle_id (Tuple[int, int]): The ID of the particle.
            trajectory_data (np.ndarray): A 2D numpy array of the trajectory data.
        """
        self.particle_id = particle_id
        self.trajectory = trajectory_data

    def __getitem__(self, key: str):
        """
        Gets trajectory data using dictionary-like keys for components or vectors.
        """
        # Mapping for single components (and common aliases)
        component_map = {
            "t": Indices.TIME,
            "x": Indices.X,
            "y": Indices.Y,
            "z": Indices.Z,
            "u": Indices.VX,
            "v": Indices.VY,
            "w": Indices.VZ,
            "vx": Indices.VX,
            "vy": Indices.VY,
            "vz": Indices.VZ,
            "ux": Indices.VX,
            "uy": Indices.VY,
            "uz": Indices.VZ,
            "bx": Indices.BX,
            "by": Indices.BY,
            "bz": Indices.BZ,
            "ex": Indices.EX,
            "ey": Indices.EY,
            "ez": Indices.EZ,
        }

        # Mapping for vector quantities
        vector_map = {
            "position": (Indices.X, Indices.Y, Indices.Z),
            "velocity": (Indices.VX, Indices.VY, Indices.VZ),
            "b": (Indices.BX, Indices.BY, Indices.BZ),
            "e": (Indices.EX, Indices.EY, Indices.EZ),
        }

        if key.lower() in component_map:
            idx = component_map[key.lower()]
            if self.trajectory.shape[1] > idx:
                return self.trajectory[:, idx]
            else:
                raise KeyError(
                    f"Component '{key}' not available in this trajectory data."
                )

        elif key.lower() in vector_map:
            indices = vector_map[key.lower()]
            if self.trajectory.shape[1] > max(indices):
                return (
                    self.trajectory[:, indices[0]],
                    self.trajectory[:, indices[1]],
                    self.trajectory[:, indices[2]],
                )
            else:
                raise KeyError(f"Vector '{key}' not available in this trajectory data.")

        else:
            raise KeyError(
                f"Unknown key: '{key}'. Valid keys include {list(component_map.keys()) + list(vector_map.keys())}"
            )


class FLEKSTP(object):
    """
    A class that is used to read and plot test particles. Each particle ID consists of
    a CPU index, a particle index on each CPU, and a location index.
    By default, 7 real numbers saved for each step: time + position + velocity.

    Args:
        dirs (str): the path to the test particle dataset.

    Examples:
    >>> tp = FLEKSTP("res/run1/PC/test_particles", iSpecies=1)
    >>> pIDs = list(tp.IDs())
    >>> tp.plot_trajectory(pIDs[3])
    >>> tp.save_trajectory_to_csv(pIDs[5])
    >>> ids, pData = tp.read_particles_at_time(6500.8, doSave=True)
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
    ):
        if type(dirs) == str:
            dirs = [dirs]

        header = Path(dirs[0] + "/Header")
        if header.exists():
            with open(header, "r") as f:
                self.nReal = int(f.readline())
        else:
            raise FileNotFoundError(f"Header file not found in {dirs[0]}")

        self.iSpecies = iSpecies
        self.plistfiles = list()
        self.pfiles = list()

        for outputDir in dirs:
            self.plistfiles = self.plistfiles + glob.glob(
                f"{outputDir}/FLEKS{iDomain}_particle_list_species_{iSpecies}_*"
            )

            self.pfiles = self.pfiles + glob.glob(
                f"{outputDir}/FLEKS{iDomain}_particle_species_{iSpecies}_*"
            )

        self.plistfiles.sort()
        self.pfiles.sort()

        self.indextotime = []
        if readAllFiles:
            for filename in self.pfiles:
                record = self._read_the_first_record(filename)
                if record == None:
                    continue
                self.indextotime.append(record[Indices.TIME])

        if iListEnd == -1:
            iListEnd = len(self.plistfiles)
        self.plistfiles = self.plistfiles[iListStart:iListEnd]
        self.pfiles = self.pfiles[iListStart:iListEnd]

        self.plists: List[Dict[Tuple[int, int], int]] = []
        for filename in self.plistfiles:
            self.plists.append(self.read_particle_list(filename))

        self.IDs = set()
        for plist in self.plists:
            self.IDs.update(plist.keys())

        self.filetime = []
        for filename in self.pfiles:
            record = self._read_the_first_record(filename)
            if record == None:
                continue
            self.filetime.append(record[Indices.TIME])

    def __repr__(self):
        str = (
            f"Particles species ID: {self.iSpecies}\n"
            f"Number of particles : {len(self.IDs)}\n"
            f"First time tag      : {self.filetime[0]}\n"
            f"Last  time tag      : {self.filetime[-1]}\n"
        )
        return str

    def getIDs(self):
        return list(sorted(self.IDs))

    def get_index_to_time(self) -> List:
        """
        Getter method for accessing indextotime.
        """
        if len(self.indextotime) == 0:
            print("Index to time mapping was not initialized")
        return self.indextotime

    def read_particle_list(self, filename: str) -> Dict[Tuple[int, int], int]:
        """
        Read and return a list of the particle IDs.
        """
        record_format = "iiQ"  # 2 integers + 1 unsigned long long
        record_size = struct.calcsize(record_format)
        record_struct = struct.Struct(record_format)
        nByte = Path(filename).stat().st_size
        nPart = int(nByte / record_size)
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
        pData = self.read_particle_trajectory(pID).trajectory
        if filename == None:
            filename = "trajectory_" + str(pID[0]) + "_" + str(pID[1]) + ".csv"
        header = "time [s], X [R], Y [R], Z [R], U_x [km/s], U_y [km/s], U_z [km/s]"
        if self.nReal == 10:
            header += ", B_x [nT], B_y [nT], B_z [nT]"
        if self.nReal == 13:
            header += (
                ", B_x [nT], B_y [nT], B_z [nT], E_x [uV/m], E_y [uV/m], E_z [uV/m]"
            )
        if shiftTime:
            pData[:, 0] -= pData[0, 0]
            if scaleTime:
                pData[:, 0] /= pData[-1, 0]
        np.savetxt(filename, pData, delimiter=",", header=header, comments="")

    def read_particle_trajectory(self, pID: Tuple[int, int]):
        """
        Return the trajectory of a test particle.

        Args:
            pID: particle ID

        Examples:
        >>> trajectory = tp.read_particle_trajectory((66,888))
        """
        dataList = list()
        record_format = "iiif"
        record_size = struct.calcsize(record_format)
        record_struct = struct.Struct(record_format)

        for filename, plist in zip(self.pfiles, self.plists):
            if pID in plist:
                ploc = plist[pID]
                with open(filename, "rb") as f:
                    f.seek(ploc)
                    dataChunk = f.read(record_size)
                    (cpu, idtmp, nRecord, weight) = record_struct.unpack(dataChunk)
                    binaryData = f.read(4 * self.nReal * nRecord)
                    dataList = dataList + list(
                        struct.unpack("f" * nRecord * self.nReal, binaryData)
                    )

        nRecord = int(len(dataList) / self.nReal)
        trajectory_data = np.array(dataList).reshape(nRecord, self.nReal)

        return ParticleTrajectory(pID, trajectory_data)

    def read_initial_location(self, pID):
        """
        Return the initial location of a test particle.
        """

        for filename, plist in zip(self.pfiles, self.plists):
            if pID in plist:
                ploc = plist[pID]
                with open(filename, "rb") as f:
                    f.seek(ploc)
                    binaryData = f.read(4 * 4)
                    (cpu, idtmp, nRecord, weight) = struct.unpack("iiif", binaryData)
                    nRead = 1
                    binaryData = f.read(4 * self.nReal * nRead)
                    dataList = list(struct.unpack("f" * nRead * self.nReal, binaryData))
                return dataList

    def select_particles(self, f_select: Callable = None) -> List[Tuple[int, int]]:
        """
        Return the test particles whose initial conditions satisfy the requirement
        set by the user defined function f_select. The first argument of f_select is the
        particle ID, and the second argument is the ID of a particle.

        Examples:
        >>> def f_select(tp, pid):
        >>>     pData = tp.read_initial_location(pid)
        >>>     inTime = pData[tp.Indices.TIME] < 3601
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
        pt = self.read_particle_trajectory(pID)
        vx, vy, vz = pt["velocity"]
        bx, by, bz = pt["b"]
        # Pitch Angle Calculation
        pitch_angle = self.get_pitch_angle_from_v_b(vx, vy, vz, bx, by, bz)

        return pitch_angle

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
        pt = self.read_particle_trajectory(pID)
        vx, vy, vz = pt["velocity"]
        bx, by, bz = pt["b"]

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

        # Calculate the sine square of the pitch angle
        sin_alpha_sq = 1 - (v_dot_b / (v_mag * b_mag + epsilon)) ** 2

        v_perp_sq = v_mag * v_mag * sin_alpha_sq

        epsilon = 1e-15  # Avoid division by zero
        mu = (0.5 * proton_mass * v_perp_sq) / (b_mag + epsilon)  # [J/T]

        return mu

    def plot_trajectory(
        self,
        pID: Tuple[int, int],
        *,
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

        def plot_vector(idx, labels, irow):
            for i, (id, label) in enumerate(zip(idx, labels)):
                plot_data(pt.trajectory[:, id], label, irow, i, **kwargs)

        pt = self.read_particle_trajectory(pID)
        t = pt.trajectory[:, Indices.TIME]
        tNorm = (t - t[0]) / (t[-1] - t[0])

        if type == "single":
            if xaxis == "t":
                x = t
            else:
                x = pt[xaxis]
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
            y1, y2, y3 = pt["position"]

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("location")
            ax[1].set_ylabel("velocity")
            ax[0].plot(t, y1, label="x")
            ax[0].plot(t, y2, label="y")
            ax[0].plot(t, y3, label="z")

            y1, y2, y3 = pt["velocity"]

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
            elif self.nReal == 13:  # additional B and E field
                nrow = 5

            f, ax = plt.subplots(nrow, ncol, figsize=(12, 6), constrained_layout=True)

            # Plot trajectories
            for i, a in enumerate(ax[0, :]):
                x_id = Indices.X if i < 2 else Indices.Y
                y_id = Indices.Y if i == 0 else Indices.Z
                a.plot(pt.trajectory[:, x_id], pt.trajectory[:, y_id], "k")
                a.scatter(
                    pt.trajectory[:, x_id],
                    pt.trajectory[:, y_id],
                    c=plt.cm.winter(tNorm),
                    edgecolor="none",
                    marker="o",
                    s=10,
                )
                a.set_xlabel("x" if i < 2 else "y")
                a.set_ylabel("y" if i == 0 else "z")

            plot_vector([Indices.X, Indices.Y, Indices.Z], ["x", "y", "z"], 1)
            plot_vector(
                [Indices.VX, Indices.VY, Indices.VZ],
                ["Vx", "Vy", "Vz"],
                2,
            )

            if self.nReal > Indices.BX:
                plot_vector(
                    [Indices.BX, Indices.BY, Indices.BZ],
                    ["Bx", "By", "Bz"],
                    3,
                )

            if self.nReal > Indices.EX:
                plot_vector(
                    [Indices.EX, Indices.EY, Indices.EZ],
                    ["Ex", "Ey", "Ez"],
                    4,
                )
        elif type == "full":
            print(f"Analyzing particle ID: {pt.pid}")
            #TODO Proper unit handling
            # --- Data Extraction ---
            t = pt.trajectory[:, Indices.TIME]  # [s]
            x, y, z = pt["position"]  # [RE]
            vx, vy, vz = pt["velocity"]  # [km/s]
            bx, by, bz = pt["b"]  # [nT]
            ex, ey, ez = pt["e"]  # [muV/m]

            # --- Derived Quantities Calculation ---

            # Kinetic Energy
            ke = self.get_kinetic_energy(vx, vy, vz) * 1e6  # [eV]

            # Vectorize B and V fields for easier calculations
            v_vec = np.vstack((vx, vy, vz)).T
            b_vec = np.vstack((bx, by, bz)).T
            e_vec = np.vstack((ex, ey, ez)).T

            # Calculate magnitudes of vectors
            v_mag = np.linalg.norm(v_vec, axis=1)  # [km/s]
            b_mag = np.linalg.norm(b_vec, axis=1)  # [nT]
            e_mag = np.linalg.norm(e_vec, axis=1) * 1e-3  # [mV/m]

            # Magnetic Field Energy Density Calculation
            U_B = (b_mag * 1e-9) ** 2 / (2 * mu_0)  # [J/m^3]

            # Electric Field Energy Density Calculation
            U_E = 0.5 * epsilon_0 * (e_mag * 1e-3) ** 2  # [J/m^3]

            # Pitch Angle Calculation
            v_dot_b = np.sum(v_vec * b_vec, axis=1)
            epsilon = 1e-15
            cos_alpha = v_dot_b / (v_mag * b_mag + epsilon)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            pitch_angle_rad = np.arccos(cos_alpha)
            pitch_angle = pitch_angle_rad * 180.0 / np.pi

            # --- First Adiabatic Invariant (mu) Calculation ---
            # mu = mv_perp^2 / 2B.  v_perp = v * sin(alpha)
            # Ensure units are SI: v [m/s], B [T]
            # Calculate perpendicular velocity squared
            v_perp_sq = (v_mag * 1e3 * np.sin(pitch_angle_rad)) ** 2

            # Calculate mu, handle potential division by zero in B
            mu = (0.5 * proton_mass * v_perp_sq) / (b_mag * 1e-9)  # [J/T]

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

            f.suptitle(f"Test Particle ID: {pt.pid}", fontsize=16)

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
