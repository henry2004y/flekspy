import numpy as np
import pytest
from unittest.mock import MagicMock
from flekspy.amrex.particle_data import AMReXParticle

class MockAMReXParticle(AMReXParticle):
    def __init__(self, rdata, header):
        self._rdata = rdata
        self.header = header
        self.output_dir = "mock_dir" # Dummy path
        # Mock other attributes needed by fit_gmm
        self._idata = np.empty((0, 0)) # Prevent load trigger if checked

    @property
    def rdata(self):
        return self._rdata

    def _extract_variable_columns(self, rdata, variables, component_names=None):
        # Determine column indices
        if component_names is None:
            component_names = self.header.real_component_names

        indices = [component_names.index(var) for var in variables]
        return rdata[:, indices]

    def select_particles_in_region(self, x_range=None, y_range=None, z_range=None):
        # For this test, we assume no region selection is actually performed or needed
        # fit_gmm calls this if ranges are provided.
        # If fit_gmm calls this, we just return all data for simplicity unless
        # we specifically test range selection (which we aren't here).
        return self._rdata


@pytest.fixture
def mock_weighted_data():
    """
    Creates a MockAMReXParticle with weighted data.
    Two populations:
    1. Center 0, Weight 1
    2. Center 10, Weight 100
    Equal number of particles.
    """
    rng = np.random.default_rng(42)
    n_per_group = 1000

    # Group 1: Center 0, weight 1
    g1_x = rng.normal(0, 0.1, n_per_group)
    g1_w = np.ones(n_per_group)

    # Group 2: Center 10, weight 100
    g2_x = rng.normal(10, 0.1, n_per_group)
    g2_w = np.full(n_per_group, 100.0)

    x = np.concatenate([g1_x, g2_x])
    w = np.concatenate([g1_w, g2_w])

    # Dummy y for 2D requirement of fit_gmm if needed, though we fit 1D "x" mostly
    y = np.zeros_like(x)

    # Create rdata: columns [x, y, weight]
    rdata = np.column_stack([x, y, w])

    header = MagicMock()
    header.real_component_names = ["x", "y", "weight"]

    return MockAMReXParticle(rdata, header)

def test_fit_gmm_weighted(mock_weighted_data):
    """
    Tests that fit_gmm respects particle weights.
    Without weighting: Mean should be ~5 (average of 0 and 10).
    With weighting (1 vs 100): Mean should be close to 10.
    """
    # Fit GMM on 'x'
    # We pass variables=['x'] to fit 1D
    gmm = mock_weighted_data.fit_gmm(n_components=1, variables=['x'])

    mean = gmm.means_[0][0]

    # If weights are ignored, mean is (0 + 10) / 2 = 5
    # If weights are respected, mean is (1*0 + 100*10) / 101 ~= 9.9

    print(f"GMM Mean: {mean}")

    # Assert that the mean is significantly higher than 5, indicating weights were used.
    # We use a loose bound to account for randomness, but 5 vs 9.9 is huge.
    assert mean > 8.0, f"Mean {mean} is too low, weights likely ignored."
