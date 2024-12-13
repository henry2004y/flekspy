import pytest
import os
import numpy as np

import flekspy
from flekspy.util.utilities import download_testfile

import matplotlib

matplotlib.use("agg")


filedir = os.path.dirname(__file__)

if not os.path.isfile(filedir + "/data/3d_raw.out"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/batsrus_data.tar.gz"
    download_testfile(url, "tests/data")

if not os.path.isdir(filedir + "/data/test_particles"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles.tar.gz"
    download_testfile(url, "tests/data")

if not os.path.isdir(filedir + "/data/3d_particle_region0_1_t00000002_n00000007_amrex"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/3d_particle.tar.gz"
    download_testfile(url, "data")


class TestIDL:
    files = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
    )
    files = [os.path.join("tests/data/", file) for file in files]

    def test_load(self):
        ds = flekspy.load(self.files[0])
        assert ds.__repr__().startswith("filename")
        assert ds.time == 25.6
        assert ds.data["x"][1] == -126.5
        assert ds.data["Bx"][2] == 0.22360679775

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            ds = flekspy.load("None")

    def test_extract(self):
        ds = flekspy.load(self.files[1])
        sat = np.array([[-28000.0, 0.0], [9000.0, 0.0]])
        d = ds.extract_data(sat)
        assert d[0][1] == 0.0

    def test_plot(self):
        ds = flekspy.load(self.files[0])
        ds.plot("p")
        ds = flekspy.load(self.files[1])
        ds.pcolormesh("x")
        ds.pcolormesh("Bx", "By", "Bz")
        assert True


class TestAMReX:
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", "3d*amrex")
    files = [os.path.join("tests/data/", file) for file in files]

    def test_load(self):
        ds = flekspy.load(self.files[0])
        assert ds.data["uxS0"][2, 1] == np.float32(-131.71918)
        assert ds.data["uxS1"].shape == (601, 2)

    def test_pic(self):
        ds = flekspy.load(self.files[1])
        assert ds.domain_left_edge[0].v == -0.016

    def test_phase(self):
        ds = flekspy.load(self.files[1])
        x_field = "p_uy"
        y_field = "p_uz"
        z_field = "p_w"
        xleft = [-0.016, -0.01, ds.domain_left_edge[2]]
        xright = [0.016, 0.01, ds.domain_right_edge[2]]

        ## Select and plot the particles inside a box defined by xleft and xright
        pp = ds.plot_phase(
            xleft,
            xright,
            x_field,
            y_field,
            z_field,
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
        pp = ds.plot_particles_region(
            sp, "p_x", "p_y", "p_w", unit_type="si", x_bins=32, y_bins=32
        )
        pp = ds.plot_phase_region(
            sp, "p_uy", "p_uz", "p_w", unit_type="si", x_bins=64, y_bins=64
        )
        assert True


class TestParticles:
    dirs = ("tests/data/test_particles",)
    from flekspy import FLEKSTP

    tp = FLEKSTP(dirs, iSpecies=1)
    pIDs = tp.getIDs()
    assert pIDs[0] == (0, 5121)
    traj = tp.read_particle_trajectory(pIDs[10])
    assert traj[0, 1] == -0.031386006623506546
    x = tp.read_initial_location(pIDs[10])
    assert x[1] == traj[0, 1]
    ids, pData = tp.read_particles_at_time(0.0, doSave=False)
    assert ids[1][1] == 5129


def load(files):
    """
    Benchmarking flekspy loading.
    """
    ds = flekspy.load(files[0])
    ds = flekspy.load(files[1])
    return ds


def test_load(benchmark):
    path = "tests/data/"
    files = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
    )
    files = [os.path.join(path, file) for file in files]

    result = benchmark(load, files)

    assert type(result) == flekspy.IDLData
