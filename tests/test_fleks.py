import pytest
import os
import numpy as np

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
        assert ds.__repr__().startswith("filename")
        assert ds.time == 25.6
        assert ds.data["x"][1] == -126.5
        assert ds.data["Bx"][2] == 0.22360679775

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            ds = fs.load("None")

    def test_extract(self):
        ds = fs.load(self.files[1])
        sat = np.array([[-28000.0, 0.0], [9000.0, 0.0]])
        d = ds.extract_data(sat)
        assert d[0][1] == 0.0

    def test_slice(self):
        ds = fs.load(self.files[2])
        slice = ds.get_slice("z", 0.0)
        assert slice.dimensions == (8, 8)
        assert slice.data["absdivB"][2, 3].value == np.float32(3.3033288e-05)

    def test_plot(self):
        ds = fs.load(self.files[0])
        ds.plot("p")
        ds = fs.load(self.files[1])
        ds.pcolormesh("x")
        ds.pcolormesh("Bx", "By", "Bz")
        assert True


class TestAMReX:
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", "3d*amrex")
    files = [os.path.join("tests/data/", file) for file in files]

    def test_load(self):
        ds = fs.load(self.files[0])
        assert ds.data["uxS0"][2, 1] == np.float32(-131.71918)
        assert ds.data["uxS1"].shape == (601, 2)

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
    dirs = ("tests/data/test_particles",)
    from flekspy import FLEKSTP

    tp = FLEKSTP(dirs, iSpecies=1)
    pIDs = tp.getIDs()

    def test_particles(self):
        tp = self.tp
        pIDs = self.pIDs
        assert tp.__repr__().startswith("Particles")
        assert pIDs[0] == (0, 5121)
        pt = tp.read_particle_trajectory(pIDs[10])
        assert pt.trajectory[0, 1] == -0.031386006623506546
        assert pt["u"][3] == 5.870406312169507e-05
        assert pt["v"][5] == 4.103916944586672e-05
        assert pt["w"].shape == (8,)
        assert pt.get_vector("x")[0].shape == (8,)
        with pytest.raises(Exception):
            pt["unknown"]
        x = tp.read_initial_location(pIDs[10])
        assert x[1] == pt.trajectory[0, 1]
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
            pData = tp.read_initial_location(pid)
            inRegion = pData[Indices.X] > 0 and pData[Indices.Y] > 0
            return inRegion

        pSelected = self.tp.select_particles(f_select)
        assert len(pSelected) == 2560

    def test_trajectory(self):
        tp = self.tp
        pIDs = self.pIDs
        ax = tp.plot_trajectory(pIDs[0], type="single")
        assert ax.get_xlim()[1] == 2.140599811077118
        ax = tp.plot_trajectory(pIDs[0], type="single", xaxis="y", yaxis="z", ax=ax)
        assert ax.get_xlim()[0] == -0.12292625373229385
        ax = tp.plot_trajectory(pIDs[0], type="xv")
        assert ax[1].get_xlim()[1] == 2.140599811077118
        ax = tp.plot_trajectory(pIDs[0])
        assert ax[1][0].get_xlim()[1] == 2.140599811077118


def load(files):
    """
    Benchmarking flekspy loading.
    """
    ds = fs.load(files[0])
    ds = fs.load(files[1])
    return ds


def test_load(benchmark):
    filenames = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
    )
    files = [os.path.join("tests/data/", file) for file in filenames]

    result = benchmark(load, files)

    assert type(result) == fs.IDLData
