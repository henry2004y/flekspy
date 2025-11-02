import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt
from flekspy.amrex import AMReXParticleData

@pytest.mark.parametrize(
    "data_file",
    ["3d_particle_region0_1_t00000002_n00000007_amrex"],
)
def test_plot_intersecting_planes(setup_test_data, data_file):
    """
    Test the plot_intersecting_planes method to ensure it runs without errors and returns
    a matplotlib figure and axes.
    """
    from pathlib import Path

    data_dir = Path(setup_test_data)
    ds = AMReXParticleData(data_dir / data_file)

    with patch("matplotlib.pyplot.show"):
        fig, ax = ds.plot_intersecting_planes("x", "y", "velocity_z")
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)

    # Test with normalization
    with patch("matplotlib.pyplot.show"):
        fig, ax = ds.plot_intersecting_planes("x", "y", "velocity_z", normalize=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)

    # Test with different variables
    with patch("matplotlib.pyplot.show"):
        fig, ax = ds.plot_intersecting_planes("velocity_x", "velocity_y", "velocity_z")
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        plt.close(fig)
