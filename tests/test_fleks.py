import pytest
import os
import itertools
import numpy as np
import xarray as xr
import polars as pl

from flekspy.tp import interpolate_at_times
import flekspy as fs
from flekspy.util import download_testfile

import matplotlib

matplotlib.use("agg")


filedir = os.path.dirname(__file__)

if not os.path.isfile(filedir + "/data/3d_raw.out"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/batsrus_data.tar.gz"
    download_testfile(url, "tests/data")

if not os.path.isdir(filedir + "/data/test_particles"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles.tar.gz"
    download_testfile(url, "tests/data")

if not os.path.isdir(filedir + "/data/test_particles_PBEG"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles_PBEG.tar.gz"
    download_testfile(url, "tests/data")

if not os.path.isdir(filedir + "/data/3d_particle_region0_1_t00000002_n00000007_amrex"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/3d_particle.tar.gz"
    download_testfile(url, "tests/data")


class TestIDL:
    filenames = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
        "3d_raw.out",
    )
    files = [os.path.join("tests/data/", file) for file in filenames]

    def test_load(self):
        ds = fs.load(self.files[0])
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["time"] == 25.6
        assert ds.coords["x"][1] == -126.5
        assert np.isclose(ds["Bx"][2].item(), 0.22360679775)

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            ds = fs.load("None")

    def test_extract(self):
        ds = fs.load(self.files[1])
        d = ds.interp(x=-28000.0, y=0.0)
        assert len(d) == 28

    def test_slice(self):
        ds = fs.load(self.files[2])
        slice_data = ds.idl.get_slice("z", 0.0)
        assert isinstance(slice_data, xr.Dataset)
        assert len(slice_data) == 14
        assert slice_data.sizes["x"] == 8
        assert slice_data.sizes["y"] == 8
        assert slice_data["absdivB"].shape == (8, 8)

    def test_plot(self):
        ds = fs.load(self.files[0])
        ds.p.plot()
        ds = fs.load(self.files[1])
        ds.rhoS0.plot.pcolormesh(x="x", y="y")
        ds["Bx"].plot.pcolormesh(x="x", y="y")
        ds.plot.streamplot(x="x", y="y", u="Bx", v="By", color="w")
        assert True


class TestAMReX:
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", "3d*amrex")
    files = [os.path.join("tests/data/", file) for file in files]

    def test_load(self):
        ds = fs.load(self.files[0])
        assert ds["uxS0"][2, 1] == np.float32(-131.71918)
        assert ds["uxS1"].shape == (601, 2)

    def test_pic(self):
        ds = fs.load(self.files[1])
        assert ds.domain_left_edge[0].v == -0.016
        dc = ds.get_slice("z", 0.5)
        assert dc.data["particle_id"][0].value == 216050.0
        assert dc.__repr__().startswith("variables")

    def test_phase(self):
        ds = fs.load(self.files[1])
        x_field = "p_uy"
        y_field = "p_uz"
        z_field = "p_w"
        xleft = [-0.016, -0.01, ds.domain_left_edge[2]]
        xright = [0.016, 0.01, ds.domain_right_edge[2]]

        ## Select and plot the particles inside a box defined by xleft and xright
        region = ds.box(xleft, xright)
        x, y, w = ds.get_phase(
            x_field,
            y_field,
            z_field,
            region=region,
            domain_size=(xleft[0], xright[0], xleft[1], xright[1]),
        )
        assert x.shape == (128,) and w.max() == 2.8024863240162035e23
        pp = ds.plot_phase(
            x_field,
            y_field,
            z_field,
            region=region,
            unit_type="si",
            x_bins=100,
            y_bins=32,
            domain_size=(xleft[0], xright[0], xleft[1], xright[1]),
        )
        ## Plot inside a sphere
        center = [0, 0, 0]
        radius = 1
        # Object sphere is defined in yt/data_objects/selection_objects/spheroids.py
        sp = ds.sphere(center, radius)
        pp = ds.plot_particles(
            "p_x", "p_y", "p_w", region=sp, unit_type="planet", x_bins=32, y_bins=32
        )
        pp = ds.plot_phase(
            "p_uy", "p_uz", "p_w", region=sp, unit_type="si", x_bins=16, y_bins=16
        )
        f = fs.extract_phase(pp)
        assert f[0].size == 16 and f[2].shape == (16, 16)


class TestParticles:
    dirs = (
        "tests/data/test_particles",
        "tests/data/test_particles_PBEG",
    )
    from flekspy import FLEKSTP

    tp = FLEKSTP(dirs[0], iSpecies=1)

    def test_particles(self):
        tp = self.tp
        pIDs = tp.getIDs()
        assert tp.__repr__().startswith("Particles")
        assert pIDs[0] == (0, 5121)
        pt = tp[10].collect()
        assert pt["x"][0] == -0.031386006623506546
        assert pt["vx"][3] == 5.870406312169507e-05
        assert pt["vy"][5] == 4.103916944586672e-05
        assert pt["vz"].shape == (8,)
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            pt.select("unknown")
        x = tp.read_initial_condition(pIDs[10])
        assert x[1] == pt["x"][0]
        x = tp.read_final_condition(tp.IDs[10])
        assert x[1] == pt["x"][-1]
        ids, pData = tp.read_particles_at_time(0.0, doSave=False)
        assert ids[1][1] == 5129
        ax = tp.plot_location(pData[0:2, :])
        assert ax["A"].get_xlim()[1] == -0.03136133626103401
        with pytest.raises(Exception):
            ids, pData = tp.read_particles_at_time(-10.0, doSave=False)
        with pytest.raises(Exception):
            ids, pData = tp.read_particles_at_time(10.0, doSave=False)

    def test_particle_select(self):
        from flekspy.tp import Indices

        def f_select(tp, pid):
            pData = tp.read_initial_condition(pid)
            inRegion = pData[Indices.X] > 0 and pData[Indices.Y] > 0
            return inRegion

        pSelected = self.tp.select_particles(f_select)
        assert len(pSelected) == 2560

    def test_trajectory(self):
        tp = self.tp
        pIDs = tp.getIDs()
        ax = tp.plot_trajectory(pIDs[0], type="single")
        assert ax.get_xlim()[1] == 2.140599811077118
        ax = tp.plot_trajectory(pIDs[0], type="single", xaxis="y", yaxis="z", ax=ax)
        assert ax.get_xlim()[0] == -0.12292625373229385
        ax = tp.plot_trajectory(pIDs[0], type="xv")
        assert ax[1].get_xlim()[1] == 2.140599811077118
        ax = tp.plot_trajectory(pIDs[0])
        assert ax[1][0].get_xlim()[1] == 2.140599811077118

    def test_particle_cache(self):
        tp = self.FLEKSTP(self.dirs[0], iSpecies=1, use_cache=True)
        pID = tp.getIDs()[0]

        # First access, should be read from file
        trajectory1 = tp[pID]

        # Second access, should be from cache
        trajectory2 = tp[pID]

        # Check if they are the same object
        assert trajectory1 is trajectory2

    def test_read_particle_trajectory_key_error(self):
        with pytest.raises(KeyError):
            self.tp.read_particle_trajectory((-1, -1))

    def test_read_particle_trajectory_value_error(self, monkeypatch):
        pID = self.tp.IDs[0]
        # Ensure the cache is clean for this test
        if pID in self.tp._trajectory_cache:
            del self.tp._trajectory_cache[pID]

        monkeypatch.setattr(
            self.tp,
            "_get_particle_raw_data",
            lambda pID: np.array([], dtype=np.float32),
        )
        with pytest.raises(ValueError):
            self.tp.read_particle_trajectory(pID)

    def test_interpolate_at_times_float32(self):
        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x": [0, 10, 20, 30, 40],
                "y": [40, 30, 20, 10, 0],
                "z": [0, 5, 10, 5, 0],
            }
        ).with_columns(pl.col("time").cast(pl.Float32))

        times_to_interpolate = [0.5, 1.5, 2.5]

        interpolated_df = interpolate_at_times(df, times_to_interpolate)

        assert interpolated_df.shape == (3, 4)
        assert interpolated_df["time"].dtype == pl.Float32
        assert np.all(
            np.isclose(interpolated_df["time"].to_list(), times_to_interpolate)
        )
        assert np.all(np.isclose(interpolated_df["x"].to_list(), [5.0, 15.0, 25.0]))
        assert np.all(np.isclose(interpolated_df["y"].to_list(), [35.0, 25.0, 15.0]))
        assert np.all(np.isclose(interpolated_df["z"].to_list(), [2.5, 7.5, 7.5]))

    def test_EBG(self):
        from flekspy.tp import plot_integrated_energy

        tp = self.FLEKSTP(self.dirs[1], iSpecies=1, use_cache=True)
        p0_collected = tp[0].collect()
        assert p0_collected.item(0, 7) == 224199.65625  # bx
        assert p0_collected.item(0, 16) == 2194893.75  # dbydx
        pid = tp.getIDs()[0]
        assert tp.get_pitch_angle(pid)[0] == np.float32(57.661438)
        vx, vy, vz = (
            p0_collected.item(0, 4),
            p0_collected.item(0, 5),
            p0_collected.item(0, 6),
        )
        ke = tp.get_kinetic_energy(vx, vy, vz, unit="SI")
        assert np.isclose(ke, 3.361357097373841e-17)
        pt_lazy = tp[pid]
        assert np.isclose(tp.get_ExB_drift(pid).item(0, 1), 3.9656504668528214e-05)
        # kappa z, not y
        assert np.isclose(
            tp._calculate_curvature(pt_lazy).collect().item(0, -1), -0.4797530472278595
        )
        assert np.isclose(
            tp.get_curvature_drift(pid).item(0, 0), -6.5444069202873204e-18
        )
        assert np.isclose(tp.get_gradient_drift(pid).item(0, 1), -9.73609190874673e-21)

        df_drifts = tp.integrate_drift_accelerations(pid)
        plot_integrated_energy(df_drifts)
        tp.analyze_drifts(pid)

        rg2rc = tp.get_gyroradius_to_curvature_ratio(pid)[0]
        assert np.isclose(rg2rc, 4.83376226572133e-12)

def load(files):
    """
    Benchmarking flekspy loading.
    """
    ds = fs.load(files[0])
    ds = fs.load(files[1])
    return ds

def test_load_idl(benchmark):
    filenames = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
    )
    files = [os.path.join("tests/data/", file) for file in filenames]

    result = benchmark(load, files)

    assert isinstance(result, xr.Dataset)

def load_test_particle_trajectories(tp, pIDs):
    """
    Load all particle trajectories.
    """
    for pID in itertools.islice(pIDs, 100):
        tp.read_particle_trajectory(pID)

def test_load_tp(benchmark):
    """
    Benchmark loading all particle trajectories.
    """
    from flekspy.tp import FLEKSTP

    dirs = (os.path.join(filedir, "data", "test_particles"),)
    tp = FLEKSTP(dirs, iSpecies=1)
    pIDs = tp.getIDs()

    benchmark(load_test_particle_trajectories, tp, pIDs)

def get_drifts(tp, pid):
    tp.get_curvature_drift(pid)
    tp.get_gradient_drift(pid)
    tp.get_ExB_drift(pid)

def test_drift_tp(benchmark):
    """
    Benchmark particle drift calculations.
    """
    from flekspy.tp import FLEKSTP

    dirs = (os.path.join(filedir, "data", "test_particles_PBEG"),)
    tp = FLEKSTP(dirs, iSpecies=1)
    pid = tp.getIDs()[0]

    benchmark(get_drifts, tp, pid)
