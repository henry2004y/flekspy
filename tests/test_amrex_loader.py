import pytest
from flekspy.amrex import AMReXParticleData
import numpy as np

@pytest.fixture(scope="module")
def particle_data():
    """Fixture to load the test particle data."""
    plotfile_directory = "tests/data/3d_particle_region0_1_t00000002_n00000007_amrex"
    return AMReXParticleData(plotfile_directory)

def test_header_properties(particle_data):
    """Tests that the header properties are read correctly."""
    assert particle_data.header.dim == 2
    # 2 (positions) + 4 (extra real components) = 6
    assert particle_data.header.num_real == 6

def test_lazy_loading(particle_data):
    """Tests that the data is loaded lazily."""
    # Initially, data should not be loaded
    assert particle_data._idata is None
    assert particle_data._rdata is None

    # Accessing the data should trigger loading
    idata = particle_data.idata
    rdata = particle_data.rdata

    # Now, data should be loaded
    assert particle_data._idata is not None
    assert particle_data._rdata is not None
    assert isinstance(idata, np.ndarray)
    assert isinstance(rdata, np.ndarray)
    assert idata.shape[0] == rdata.shape[0]
    assert idata.shape[0] == 256423 # Check number of particles from header

def test_select_particles_in_region(particle_data):
    """Tests the regional particle selection."""
    x_range = (-0.008, 0.008)
    y_range = (-0.005, 0.005)

    rdata = particle_data.select_particles_in_region(x_range=x_range, y_range=y_range)

    # Check that the returned data is within the specified region
    assert np.all(rdata[:, 0] >= x_range[0])
    assert np.all(rdata[:, 0] <= x_range[1])
    assert np.all(rdata[:, 1] >= y_range[0])
    assert np.all(rdata[:, 1] <= y_range[1])

    # Check that some particles were selected
    assert rdata.shape[0] == 64019

    # Ensure that not all particles were selected (i.e., filtering happened)
    assert rdata.shape[0] < particle_data.header.num_particles