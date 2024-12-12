import pytest
import os
import numpy as np

import flekspy
from flekspy.util.utilities import download_testfile


filedir = os.path.dirname(__file__)

if not os.path.isfile(filedir + "/data/bulk.1d.vlsv"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/batsrus_data.tar.gz"
    download_testfile(url, "data")
elif not os.path.isdir(filedir + "/data/test_particles"):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles.tar.gz"
    download_testfile(url, "data")

class TestIDL:
    files = ("1d__raw_2_t25.60000_n00000258.out",)
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

class TestAMReX:
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", )
    files = [os.path.join("tests/data/", file) for file in files]

    def test_load(self):
        ds = flekspy.load(self.files[0])
        assert ds.data["uxS0"][2,1] == np.float32(-131.71918)
        assert ds.data["uxS1"].shape == (601, 2)

class TestParticles:
    dirs = ("tests/data/test_particles", )
    from flekspy import FLEKSTP

    tp = FLEKSTP(dirs, iSpecies=1)
    pIDs = tp.getIDs()
    assert pIDs[0] == (0, 5121)
    traj = tp.read_particle_trajectory(pIDs[10])
    assert traj[0,1] == -0.031386006623506546
    x = tp.read_initial_location(pIDs[10])
    assert x[1] == traj[0,1]
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
    files = ("1d__raw_2_t25.60000_n00000258.out", "z=0_fluid_region0_0_t00001640_n00010142.out")
    files = [os.path.join(path, file) for file in files]

    result = benchmark(load, files)

    assert type(result) == flekspy.IDLData