import pytest
import requests
import tarfile
import os

import flekspy
import numpy as np
import matplotlib

filedir = os.path.dirname(__file__)

if os.path.isfile(filedir + "/data/bulk.1d.vlsv"):
    pass
else:
    url = (
        "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/batsrus_data.tar.gz"
    )
    testfiles = url.rsplit("/", 1)[1]
    r = requests.get(url, allow_redirects=True)
    open(testfiles, "wb").write(r.content)

    path = filedir + "/data"

    if not os.path.exists(path):
        os.makedirs(path)

    with tarfile.open(testfiles) as file:
        file.extractall(path)

class TestIDL:
    path = "tests/data/"
    files = ("1d__raw_2_t25.60000_n00000258.out",)
    files = [os.path.join(path, file) for file in files]

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
    path = "tests/data/"
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", )
    files = [os.path.join(path, file) for file in files]

    def test_load(self):
        ds = flekspy.load(self.files[0])
        assert ds.data["uxS0"][2,1] == np.float32(-131.71918)
        assert ds.data["uxS1"].shape == (601, 2)
    

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