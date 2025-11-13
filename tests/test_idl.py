import pytest
import flekspy as fs
import numpy as np


def test_get_pressure_anisotropy(idl_data_files):
    ds = fs.load(idl_data_files[1])
    anisotropy = ds.idl.get_pressure_anisotropy(species=1)
    assert anisotropy.name == "pressure_anisotropy_S1"
    assert anisotropy.shape == (601, 2)
    assert np.isclose(anisotropy.isel(x=0, y=0), 1.2906302, atol=1e-5)
