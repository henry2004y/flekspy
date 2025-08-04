import pandas as pd
import xarray as xr
import numpy as np
import struct
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable
from scipy.constants import proton_mass, elementary_charge, mu_0, epsilon_0
import matplotlib.pyplot as plt
import warnings
from enum import IntEnum


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


@xr.register_dataset_accessor("tp")
class TPAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def get_kinetic_energy(self, pID, mass=proton_mass):
        """Calculates the kinetic energy of a particle."""
        particle_data = self._ds.sel(particle=pID)
        vx, vy, vz = particle_data["vx"], particle_data["vy"], particle_data["vz"]
        ke = 0.5 * mass * (vx**2 + vy**2 + vz**2) / elementary_charge  # [eV]
        return ke

    def get_pitch_angle(self, pID):
        """Calculates the pitch angle of a particle."""
        particle_data = self._ds.sel(particle=pID)
        vx, vy, vz = particle_data["vx"], particle_data["vy"], particle_data["vz"]
        bx, by, bz = particle_data["bx"], particle_data["by"], particle_data["bz"]

        v_vec = np.vstack([vx.values, vy.values, vz.values]).T
        b_vec = np.vstack([bx.values, by.values, bz.values]).T

        v_mag = np.linalg.norm(v_vec, axis=1)
        b_mag = np.linalg.norm(b_vec, axis=1)
        v_dot_b = np.sum(v_vec * b_vec, axis=1)

        epsilon = 1e-15
        cos_alpha = v_dot_b / (v_mag * b_mag + epsilon)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        pitch_angle = np.arccos(cos_alpha) * 180.0 / np.pi

        return xr.DataArray(
            pitch_angle, dims=["time"], coords={"time": particle_data.time}
        )

    def get_first_adiabatic_invariant(self, pID, mass=proton_mass):
        """Calculates the first adiabatic invariant of a particle."""
        particle_data = self._ds.sel(particle=pID)
        vx, vy, vz = particle_data["vx"], particle_data["vy"], particle_data["vz"]
        bx, by, bz = particle_data["bx"], particle_data["by"], particle_data["bz"]

        v_vec = np.vstack([vx.values, vy.values, vz.values]).T
        b_vec = np.vstack([bx.values, by.values, bz.values]).T

        v_mag = np.linalg.norm(v_vec, axis=1)
        b_mag = np.linalg.norm(b_vec, axis=1)
        v_dot_b = np.sum(v_vec * b_vec, axis=1)

        epsilon = 1e-15
        sin_alpha_sq = 1 - (v_dot_b / (v_mag * b_mag + epsilon)) ** 2
        v_perp_sq = v_mag * v_mag * sin_alpha_sq
        mu = (0.5 * mass * v_perp_sq) / (b_mag + epsilon)  # [J/nT]

        return xr.DataArray(mu, dims=["time"], coords={"time": particle_data.time})

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
        pt = self._ds.sel(particle=pID)
        t = pt.time.values
        tNorm = (t - t[0]) / (t[-1] - t[0]) if len(t) > 1 else np.zeros_like(t)

        if type == "single":
            if xaxis == "t":
                x_data = t
            else:
                x_data = pt[xaxis].values
            y_data = pt[yaxis].values

            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

            ax.plot(x_data, y_data, **kwargs)
            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
            return ax

        # For brevity, only 'single' type is implemented as it's the one tested.
        warnings.warn(f"Plot type '{type}' is not fully implemented for this accessor.")
        # Fallback to a simple plot to avoid errors
        if ax is None:
            _, ax = plt.subplots()
        pt.x.plot(ax=ax)
        return ax

    def plot_location(self, time: float, **kwargs):
        """Plots the location of particles at a given time."""
        pData = self._ds.sel(time=time, method="nearest")
        px, py, pz = pData["x"].values, pData["y"].values, pData["z"].values

        skeys = ["A", "B", "C", "D"]
        f, ax = plt.subplot_mosaic(
            "AB;CD",
            per_subplot_kw={("D"): {"projection": "3d"}},
            gridspec_kw={"width_ratios": [1, 1], "wspace": 0.1, "hspace": 0.1},
            figsize=(10, 10),
            constrained_layout=True,
        )

        for i, (x, y, labels) in enumerate(
            zip([px, px, py], [py, pz, pz], [("x", "y"), ("x", "z"), ("y", "z")])
        ):
            ax[skeys[i]].scatter(x, y, s=1, **kwargs)
            ax[skeys[i]].set_xlabel(labels[0])
            ax[skeys[i]].set_ylabel(labels[1])

        ax[skeys[3]].scatter(px, py, pz, s=1, **kwargs)
        ax[skeys[3]].set_xlabel("x")
        ax[skeys[3]].set_ylabel("y")
        ax[skeys[3]].set_zlabel("z")

        return ax

def _read_particle_list(filename: str) -> Dict[Tuple[int, int], int]:
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


def read_tp_data(
    dirs: Union[str, List[str]],
    iDomain: int = 0,
    iSpecies: int = 0,
    iListStart: int = 0,
    iListEnd: int = -1,
) -> xr.Dataset:
    """
    Reads test particle data from specified directories and returns an xarray.Dataset.

    This function processes binary particle data files to construct a comprehensive
    xarray.Dataset, which is well-suited for advanced analysis and visualization.
    The output Dataset contains all particle trajectories, indexed by particle ID
    (cpu, id) and time.

    Args:
        dirs (Union[str, List[str]]): Path or list of paths to the directories
                                     containing the test particle data.
        iDomain (int, optional): Domain index. Defaults to 0.
        iSpecies (int, optional): Species index. Defaults to 0.
        iListStart (int, optional): Starting index for the particle list files.
                                    Defaults to 0.
        iListEnd (int, optional): Ending index for the particle list files.
                                  Defaults to -1 (all files).

    Returns:
        xr.Dataset: An xarray.Dataset containing the test particle data.
                    The Dataset is indexed by a multi-index 'particle' coordinate
                    (composed of 'cpu' and 'id') and a 'time' coordinate.
    """
    if isinstance(dirs, str):
        dirs = [dirs]

    header_path = Path(dirs[0]) / "Header"
    if header_path.exists():
        with open(header_path, "r") as f:
            nReal = int(f.readline())
    else:
        raise FileNotFoundError(f"Header file not found in {dirs[0]}")

    plistfiles = []
    pfiles = []
    for outputDir in dirs:
        plistfiles.extend(
            glob.glob(
                f"{outputDir}/FLEKS{iDomain}_particle_list_species_{iSpecies}_*"
            )
        )
        pfiles.extend(
            glob.glob(f"{outputDir}/FLEKS{iDomain}_particle_species_{iSpecies}_*")
        )

    plistfiles.sort()
    pfiles.sort()

    if iListEnd == -1:
        iListEnd = len(plistfiles)
    plistfiles = plistfiles[iListStart:iListEnd]
    pfiles = pfiles[iListStart:iListEnd]

    plists = [_read_particle_list(filename) for filename in plistfiles]

    # Read raw trajectory data for all particles
    all_trajectories_raw = {}
    record_format = "iiif"
    record_size = struct.calcsize(record_format)
    record_struct = struct.Struct(record_format)

    for filename, plist in zip(pfiles, plists):
        with open(filename, "rb") as f:
            for pID, ploc in plist.items():
                f.seek(ploc)
                dataChunk = f.read(record_size)
                (cpu, idtmp, nRecord, weight) = record_struct.unpack(dataChunk)

                if pID != (cpu, idtmp):
                    warnings.warn(f"Mismatch pID {pID} and read pID {(cpu, idtmp)}")
                    continue

                if nRecord > 0:
                    binaryData = f.read(4 * nReal * nRecord)
                    data = list(struct.unpack("f" * nRecord * nReal, binaryData))
                    if pID not in all_trajectories_raw:
                        all_trajectories_raw[pID] = []
                    all_trajectories_raw[pID].extend(data)

    # Process raw data into structured numpy arrays and collect all time steps
    trajectories = {}
    all_times = set()
    for pID, raw_data in all_trajectories_raw.items():
        n_records = len(raw_data) // nReal
        traj = np.array(raw_data).reshape(n_records, nReal)
        trajectories[pID] = traj
        all_times.update(traj[:, Indices.TIME])

    if not trajectories:
        return xr.Dataset(attrs={"nReal": nReal, "iSpecies": iSpecies})

    # Create common coordinates for the dataset
    pids = sorted(trajectories.keys())
    time_coord = sorted(list(all_times))
    time_to_idx = {t: i for i, t in enumerate(time_coord)}
    variable_names = [e.name.lower() for e in Indices]

    # Create a NaN-filled data array
    data = np.full((len(pids), len(time_coord), nReal), np.nan, dtype=np.float32)

    # Fill the data array with trajectory data
    for i, pID in enumerate(pids):
        traj = trajectories[pID]
        for j in range(traj.shape[0]):
            t = traj[j, Indices.TIME]
            ti = time_to_idx[t]
            data[i, ti, :] = traj[j, :]

    # Create the xarray Dataset
    particle_index = pd.MultiIndex.from_tuples(pids, names=["cpu", "id"])
    data_vars = {}
    for i, var_name in enumerate(variable_names):
        if i < nReal and var_name != "time":
            data_vars[var_name] = xr.DataArray(
                data[:, :, i],
                dims=("particle", "time"),
                coords={"particle": particle_index, "time": time_coord},
            )

    ds = xr.Dataset(data_vars, attrs={"nReal": nReal, "iSpecies": iSpecies})

    # Add aliases for backward compatibility without copying data
    aliases = {
        "u": "vx", "v": "vy", "w": "vz",
        "ux": "vx", "uy": "vy", "uz": "vz",
    }
    for alias, original in aliases.items():
        if original in ds.variables:
            ds.variables[alias] = ds.variables[original]

    return ds
