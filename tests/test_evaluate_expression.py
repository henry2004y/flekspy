import pytest
import os
import numpy as np

import flekspy as fs
from flekspy.util import download_testfile

import matplotlib

matplotlib.use("agg")


filedir = os.path.dirname(__file__)

if not os.path.isdir(
    os.path.join(filedir, "data", "3d_particle_region0_1_t00000002_n00000007_amrex")
):
    url = "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/3d_particle.tar.gz"
    download_testfile(url, "tests/data")


class TestEvaluateExpression:
    files = ("3d_region*amrex",)
    files = [os.path.join("tests/data/", file) for file in files]

    def test_evaluate_expression(self):
        ds = fs.load(self.files[0])
        dc = ds.get_slice("z", 0.5)

        # Test a simple expression
        result = dc.evaluate_expression("{Bx} + {By}")

        import yt

        assert isinstance(result, yt.units.yt_array.YTArray)
        assert result.shape == dc.data["Bx"].shape
        expected = dc.data["Bx"] + dc.data["By"]
        np.testing.assert_allclose(result.value, expected.value)

        # Test an expression with a function call
        result = dc.evaluate_expression("np.sqrt({Bx}**2+{By}**2)")

        assert isinstance(result, yt.units.yt_array.YTArray)
        assert result.shape == dc.data["Bx"].shape
        expected = np.sqrt(dc.data["Bx"] ** 2 + dc.data["By"] ** 2)
        np.testing.assert_allclose(result.value, expected.value)

        # Test with a non-existent variable
        with pytest.raises(KeyError):
            dc.evaluate_expression("{non_existent_var}")
