
import pytest
from flekspy.amrex import AMReXParticle

def test_load_mock_3d_data(mock_3d_amrex_data):
    # Use the mock data fixture
    data_path = mock_3d_amrex_data

    ds = AMReXParticle(data_path)
    assert ds.dim == 3
    assert len(ds.domain_dimensions) == 3
    # Our mock header defined: ((0,0,0) (10,10,10) (0,0,0))
    # x1=0, x2=10 -> dim = 10 - 0 + 1 = 11
    # Similarly for y and z
    assert ds.domain_dimensions == [11, 11, 11] 
    
    # Load data to ensure it works
    assert ds.rdata is not None
    assert ds.rdata.shape == (100, 7) # 100 particles, 7 real components
