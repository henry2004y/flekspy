from typing import List, Tuple, Dict, Union, Callable

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import glob
import struct
from enum import IntEnum


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
                self.indextotime.append(record[self.Indices.TIME])

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
            self.filetime.append(record[self.Indices.TIME])

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
        record_format = "iiQ" # 2 integers + 1 unsigned long long
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
                    if allRecords[self.nReal * i + self.Indices.TIME] >= time:
                        dataList.append(
                            allRecords[self.nReal * i : self.nReal * (i + 1)]
                        )
                        idList.append((cpu, idtmp))
                        break
                    elif (
                        i == nRecord - 1
                        and allRecords[self.nReal * i + self.Indices.TIME] < time
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
        pData = self.read_particle_trajectory(pID)
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
        return np.array(dataList).reshape(nRecord, self.nReal)

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
        >>>     inTime = pData[tp.indices.TIME] < 3601
        >>>     inRegion = pData[self.Indices.X] > 20
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

    def get_data(self, data, name: str):
        match name:
            case "t":
                x = data[:, self.Indices.TIME]
            case "x":
                x = data[:, self.Indices.X]
            case "y":
                x = data[:, self.Indices.Y]
            case "z":
                x = data[:, self.Indices.Z]
            case "u" | "vx" | "ux":
                x = data[:, self.Indices.VX]
            case "v" | "vy" | "uy":
                x = data[:, self.Indices.VY]
            case "w" | "vz" | "uz":
                x = data[:, self.Indices.VZ]
            case "Bx" | "bx":
                x = data[:, self.Indices.BX]
            case "By" | "by":
                x = data[:, self.Indices.BY]
            case "Bz" | "bz":
                x = data[:, self.Indices.BZ]
            case "Ex" | "ex":
                x = data[:, self.Indices.EX]
            case "Ey" | "ey":
                x = data[:, self.Indices.EY]
            case "Ez" | "ez":
                x = data[:, self.Indices.EZ]
            case _:
                raise Exception(f"Unknown plot variable {name}")

        return x

    def get_pitch_angle(self, pID):
        data = self.read_particle_trajectory(pID)
        vx = self.get_data(data, "u")
        vy = self.get_data(data, "v")
        vz = self.get_data(data, "w")
        bx = self.get_data(data, "bx")
        by = self.get_data(data, "by")
        bz = self.get_data(data, "bz")
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

    def plot_trajectory(
        self,
        pID: Tuple[int, int],
        *,
        type="all",
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
                plot_data(data[:, id], label, irow, i, **kwargs)

        data = self.read_particle_trajectory(pID)
        t = data[:, self.Indices.TIME]
        tNorm = (t - t[0]) / (t[-1] - t[0])

        if type == "single":
            if xaxis == "t":
                x = t
            else:
                x = self.get_data(data, xaxis)
            y = self.get_data(data, yaxis)

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
            y1 = self.get_data(data, "x")
            y2 = self.get_data(data, "y")
            y3 = self.get_data(data, "z")

            ax[0].set_xlabel("t")
            ax[0].set_ylabel("location")
            ax[1].set_ylabel("velocity")
            ax[0].plot(t, y1, label="x")
            ax[0].plot(t, y2, label="y")
            ax[0].plot(t, y3, label="z")

            y1 = self.get_data(data, "vx")
            y2 = self.get_data(data, "vy")
            y3 = self.get_data(data, "vz")
            
            ax[1].plot(t, y1, label="vx")
            ax[1].plot(t, y2, label="vy")
            ax[1].plot(t, y3, label="vz")
            
            for a in ax:
                a.legend()
                a.grid()

        elif type == "all":
            ncol = 3
            nrow = 3  # Default for X, V
            if self.nReal == 10:  # additional B field
                nrow = 4
            elif self.nReal == 13:  # additional B and E field
                nrow = 5

            f, ax = plt.subplots(nrow, ncol, figsize=(12, 6), constrained_layout=True)

            # Plot trajectories
            for i, a in enumerate(ax[0, :]):
                x_id = self.Indices.X if i < 2 else self.Indices.Y
                y_id = self.Indices.Y if i == 0 else self.Indices.Z
                a.plot(data[:, x_id], data[:, y_id], "k")
                a.scatter(
                    data[:, x_id],
                    data[:, y_id],
                    c=plt.cm.winter(tNorm),
                    edgecolor="none",
                    marker="o",
                    s=10,
                )
                a.set_xlabel("x" if i < 2 else "y")
                a.set_ylabel("y" if i == 0 else "z")

            plot_vector([self.Indices.X, self.Indices.Y, self.Indices.Z], ["x", "y", "z"], 1)
            plot_vector([self.Indices.VX, self.Indices.VY, self.Indices.VZ], ["Vx", "Vy", "Vz"], 2)

            if self.nReal > self.Indices.BX:
                plot_vector(
                    [self.Indices.BX, self.Indices.BY, self.Indices.BZ], ["Bx", "By", "Bz"], 3
                )

            if self.nReal > self.Indices.EX:
                plot_vector(
                    [self.Indices.EX, self.Indices.EY, self.Indices.EZ], ["Ex", "Ey", "Ez"], 4
                )

        return ax

    def plot_location(self, pData: np.ndarray):
        """
        Plot the location of particles pData.

        Examples:
        >>> ids, pData = tp.read_particles_at_time(3700, doSave=True)
        >>> f = tp.plot_location(pData)
        """

        px = pData[:, self.Indices.X]
        py = pData[:, self.Indices.Y]
        pz = pData[:, self.Indices.Z]

        # create subplot mosaic with different keyword arguments
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
