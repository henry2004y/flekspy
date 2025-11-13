import pytest
import os
from flekspy.util import download_testfile


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Fixture to download and extract test data once per session.
    """
    data_dir = "tests/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Dictionary of test data files and their URLs
    test_data_urls = {
        "batsrus_data.tar.gz": "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/batsrus_data.tar.gz",
        "test_particles.tar.gz": "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles.tar.gz",
        "test_particles_PBEG.tar.gz": "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/test_particles_PBEG.tar.gz",
        "3d_particle.tar.gz": "https://raw.githubusercontent.com/henry2004y/batsrus_data/master/3d_particle.tar.gz",
    }

    # Check for the existence of an expected file from each archive
    expected_files = {
        "batsrus_data.tar.gz": "3d_raw.out",
        "test_particles.tar.gz": "test_particles",
        "test_particles_PBEG.tar.gz": "test_particles_PBEG",
        "3d_particle.tar.gz": "3d_particle_region0_1_t00000002_n00000007_amrex",
    }

    for archive, url in test_data_urls.items():
        # Check if the data is already extracted
        expected_path = os.path.join(data_dir, expected_files[archive])
        if not os.path.exists(expected_path):
            download_testfile(url, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def idl_data_files(setup_test_data):
    """Fixture to provide paths to the IDL test data files."""
    filenames = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
        "3d_raw.out",
        "bx0_mhd_6_t00000100_n00000352.out",
    )
    return [os.path.join(setup_test_data, file) for file in filenames]
