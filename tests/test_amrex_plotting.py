import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

# Since AMReXPlottingMixin is in a private module, we import it like this
from flekspy.amrex.plotting import AMReXPlottingMixin

# To test the mixin, we need a class that uses it
class MockAMReXData(AMReXPlottingMixin):
    def __init__(self, rdata, header):
        self.rdata = rdata
        self.header = header

    def select_particles_in_region(self, x_range=None, y_range=None, z_range=None):
        return self.rdata

@pytest.fixture
def mock_amrex_data():
    """Creates a mock AMReXParticleData object for testing."""
    # Mock header
    header = MagicMock()
    header.real_component_names = ["x", "y", "z", "velocity_x", "velocity_y", "velocity_z", "weight"]

    # Mock particle data
    # Create 100 particles with random data
    rdata = np.random.rand(100, 7)

    return MockAMReXData(rdata, header)

@patch("matplotlib.pyplot.show")
def test_pairplot_corner(mock_show, mock_amrex_data):
    """
    Tests that the pairplot method with corner=True only plots the lower
    triangle of the plot matrix.
    """
    fig, axes = mock_amrex_data.pairplot(
        variables=["velocity_x", "velocity_y", "velocity_z"],
        corner=True
    )

    # The axes that should be visible
    visible_axes = [
        (0, 0),
        (1, 0), (1, 1),
        (2, 0), (2, 1), (2, 2)
    ]

    nvar = len(axes)
    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]
            if (i, j) in visible_axes:
                assert ax.get_visible(), f"Axis ({i}, {j}) should be visible"
            else:
                assert not ax.get_visible(), f"Axis ({i}, {j}) should NOT be visible"

    plt.close(fig)
