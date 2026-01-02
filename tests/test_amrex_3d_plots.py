import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt
from flekspy.amrex import AMReXParticle
from pathlib import Path

@pytest.mark.parametrize(
    "data_file",
    ["3d_particle_region0_1_t00000002_n00000007_amrex"],
)
@pytest.mark.parametrize("plot_method_name", [
    "plot_phase_3d",
    "plot_intersecting_planes",
])
def test_3d_plots(setup_test_data, data_file, plot_method_name):
    """
    Test 3D plotting methods to ensure they run without errors and return
    a matplotlib figure and axes.
    """
    data_dir = Path(setup_test_data)
    ds = AMReXParticle(data_dir / data_file)
    plot_func = getattr(ds, plot_method_name)

    with patch("matplotlib.pyplot.show"):
        fig, ax = plot_func("x", "y", "velocity_z")
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)

    # Test with normalization
    with patch("matplotlib.pyplot.show"):
        fig, ax = plot_func("x", "y", "velocity_z", normalize=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)

    # Test with different variables
    with patch("matplotlib.pyplot.show"):
        fig, ax = plot_func("velocity_x", "velocity_y", "velocity_z")
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)
