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


@pytest.fixture
def mock_3d_amrex_data(tmp_path):
    """
    Generates a mock 3D AMReX particle dataset directory structure.
    Returns the path to the dataset directory.
    """
    import numpy as np
    import numpy as np

    output_dir = tmp_path / "mock_3d_amrex"
    output_dir.mkdir()
    
    # 1. Create Main Header
    # flekspy checks: version, num_fields, dim, time, prob_refine, left/right edge, domain definition
    header_path = output_dir / "Header"
    with open(header_path, "w") as f:
        f.write("HyperCLaw-V1.1\n")
        f.write("0\n") # num_fields
        f.write("3\n") # dim
        f.write("0.0\n") # time
        f.write("0\n") # prob_refine_ratio (ignored)
        f.write("-1.0 -1.0 -1.0\n") # left_edge
        f.write("1.0 1.0 1.0\n") # right_edge
        f.write("0\n") # ignored line
        # The line causing issues earlier: 3D domain definition
        # x1, y1, z1, x2, y2, z2, _, _, _
        f.write("((0,0,0) (10,10,10) (0,0,0))\n") 

    # 2. Create Particle Directory Structure
    particles_dir = output_dir / "particles"
    particles_dir.mkdir()
    
    level_0_dir = particles_dir / "Level_0"
    level_0_dir.mkdir()

    # 3. Create Particle Data
    num_particles = 100
    num_int = 2 # id, cpu
    num_real = 7 # x, y, z, vx, vy, vz, w
    
    # Random data
    idata = np.zeros((num_particles, num_int), dtype=np.int32)
    idata[:, 0] = np.arange(num_particles) # IDs
    
    rdata = np.random.rand(num_particles, num_real).astype(np.float64)
    # Ensure positions are within bounds (-1 to 1)
    rdata[:, 0:3] = rdata[:, 0:3] * 2 - 1
    
    # Write binary data file
    data_filename = "DATA_00000"
    data_path = level_0_dir / data_filename
    
    # We need to know where we write the data to update the header
    # header.grids stores: (which, count, where)
    # which is the suffix of DATA_nnnnn (0 here)
    # count is number of particles
    # where is offset.
    
    # AMReX binary particle format:
    # If checkpoint: ints then floats
    # If not checkpoint: just floats? 
    # Let's check particle_data.py read_amrex_binary_particle_file:
    # if header.is_checkpoint: ints... floats...
    # else: floats... (and num_int forced to 0 in header init)
    
    # Let's verify header.is_checkpoint parsing. 
    # It reads line after components.
    
    is_checkpoint = True
    
    with open(data_path, "wb") as f:
        # Write Ints
        f.write(idata.tobytes())
        # Write Reals
        f.write(rdata.tobytes())
        
    file_size = data_path.stat().st_size
    
    # 4. Create Particle Header
    p_header_path = particles_dir / "Header"
    with open(p_header_path, "w") as f:
        f.write("Version_double\n") # version indicating double precision
        f.write("3\n") # dim
        f.write("4\n") # num_real_extra (total 3 base + 4 extra = 7)
        f.write("velocity_x\n")
        f.write("velocity_y\n")
        f.write("velocity_z\n")
        f.write("weight\n")
        f.write("0\n") # num_int_extra (total 2 base + 0 extra = 2)
        f.write(f"{int(is_checkpoint)}\n") # is_checkpoint
        f.write(f"{num_particles}\n") # num_particles
        f.write(f"{num_particles + 1}\n") # max_next_id
        f.write("0\n") # finest_level
        f.write("1\n") # grids_per_level (Level 0)
        # Grid info for Level 0: which count where
        # which=0, count=num_particles, where=0
        f.write(f"0 {num_particles} 0\n") 
        
    # 5. Create Particle_H (Bounding boxes)
    p_h_path = level_0_dir / "Particle_H"
    with open(p_h_path, "w") as f:
        f.write("(1 0\n") # num_boxes level
        f.write("((0,0,0) (10,10,10) (0,0,0))\n") # The box
        f.write(")\n")

    return output_dir

