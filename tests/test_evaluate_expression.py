import pytest
import os
import numpy as np
import yt

import flekspy as fs
from flekspy.util import download_testfile

import matplotlib

matplotlib.use("agg")


@pytest.fixture(scope="module")
def amrex_dataset(setup_test_data):
    """Fixture to load the AMReX dataset for expression evaluation."""
    file_path = os.path.join(setup_test_data, "3d*amrex")
    ds = fs.load(file_path, use_yt_loader=True)
    return ds.get_slice("z", 0.5)


class TestEvaluateExpression:
    def test_simple_expression(self, amrex_dataset):
        """Test a simple arithmetic expression."""
        result = amrex_dataset.evaluate_expression("{Bx} + {By}")
        assert isinstance(result, yt.units.yt_array.YTArray)
        assert result.shape == amrex_dataset.data["Bx"].shape
        expected = amrex_dataset.data["Bx"] + amrex_dataset.data["By"]
        np.testing.assert_allclose(result.value, expected.value)

    def test_function_call_expression(self, amrex_dataset):
        """Test an expression involving a NumPy function."""
        result = amrex_dataset.evaluate_expression("np.sqrt({Bx}**2+{By}**2)")
        assert isinstance(result, yt.units.yt_array.YTArray)
        assert result.shape == amrex_dataset.data["Bx"].shape
        expected = np.sqrt(amrex_dataset.data["Bx"] ** 2 + amrex_dataset.data["By"] ** 2)
        np.testing.assert_allclose(result.value, expected.value)

    def test_non_existent_variable(self, amrex_dataset):
        """Test that a KeyError is raised for a non-existent variable."""
        with pytest.raises(KeyError):
            amrex_dataset.evaluate_expression("{non_existent_var}")
