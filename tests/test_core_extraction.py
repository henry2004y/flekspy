
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from flekspy.amrex.particle_data import AMReXParticleData
from sklearn.mixture import GaussianMixture

def generate_synthetic_data(n_core, n_supra, v_th_core, v_th_supra, v_shift_supra, dim=3):
    rng = np.random.default_rng(42)

    # Core: Maxwellian (Normal distribution) centered at 0
    core_data = rng.normal(0, v_th_core, (n_core, dim))

    # Suprathermal: Hotter and shifted
    supra_data = rng.normal(v_shift_supra, v_th_supra, (n_supra, dim))

    data = np.vstack([core_data, supra_data])

    # Shuffle only if needed, but for validation we want to know truth
    # For now, we return data and the true mask
    true_core_mask = np.concatenate([np.ones(n_core, dtype=bool), np.zeros(n_supra, dtype=bool)])

    return data, true_core_mask

@pytest.fixture
def mock_amrex_data():
    # Create a mock AMReXParticleData object
    # We need to mock .rdata and .header
    mock_obj = MagicMock(spec=AMReXParticleData)

    # Copy methods from the real class to the mock instance for testing logic
    mock_obj.fit_gmm = AMReXParticleData.fit_gmm.__get__(mock_obj, AMReXParticleData)
    mock_obj.extract_core_population = AMReXParticleData.extract_core_population.__get__(mock_obj, AMReXParticleData)
    # Also need _resolve_alias for variable name checking
    # Assuming _resolve_alias is simple, let's mock it to identity
    mock_obj._resolve_alias = lambda x: x

    return mock_obj

def test_fit_gmm_backward_compatibility(mock_amrex_data):
    """Test fit_gmm with legacy arguments."""
    n_samples = 100
    mock_amrex_data.rdata = np.random.rand(n_samples, 3)
    mock_amrex_data.header = MagicMock()
    mock_amrex_data.header.real_component_names = ["ux", "uy", "uz"]

    # Test with keyword arguments
    gmm = mock_amrex_data.fit_gmm(n_components=1, x_variable="ux", y_variable="uy")
    assert gmm.means_.shape == (1, 2)

    # Test with positional arguments (CRITICAL for backward compatibility)
    gmm_pos = mock_amrex_data.fit_gmm(1, "ux", "uy")
    assert gmm_pos.means_.shape == (1, 2)

def test_extract_core_population_validation(mock_amrex_data):
    """Test input validation for extract_core_population."""
    with pytest.raises(ValueError, match="Length of 'velocity_columns'"):
        mock_amrex_data.extract_core_population(
            velocity_columns=["ux", "uy"],
            v_dim=3
        )

def test_extract_core_population_1d(mock_amrex_data):
    """Test extract_core_population with 1D data."""
    v_th_core = 1.0
    v_th_supra = 5.0
    v_shift = 0.0
    n_core = 1000
    n_supra = 200
    dim = 1

    data, true_mask = generate_synthetic_data(n_core, n_supra, v_th_core, v_th_supra, v_shift, dim)

    # Set mock data (need to have enough columns to pick from, let's say 3 exist but we pick 1)
    full_data = np.zeros((len(data), 3))
    full_data[:, 0] = data[:, 0] # Put 1D data in first col

    mock_amrex_data.rdata = full_data
    mock_amrex_data.header = MagicMock()
    mock_amrex_data.header.real_component_names = ["ux", "uy", "uz"]

    core_mask, supra_mask = mock_amrex_data.extract_core_population(
        velocity_columns=["ux"],
        v_dim=1,
        cutoff=3.0
    )

    # Verify separation
    overlap = np.sum(core_mask & true_mask) / np.sum(true_mask)
    assert overlap > 0.95

def test_extract_core_population_2d(mock_amrex_data):
    """Test extract_core_population with 2D data."""
    v_th_core = 1.0
    v_th_supra = 5.0
    v_shift = 0.0
    n_core = 1000
    n_supra = 200
    dim = 2

    data, true_mask = generate_synthetic_data(n_core, n_supra, v_th_core, v_th_supra, v_shift, dim)

    full_data = np.zeros((len(data), 3))
    full_data[:, 0] = data[:, 0]
    full_data[:, 1] = data[:, 1]

    mock_amrex_data.rdata = full_data
    mock_amrex_data.header = MagicMock()
    mock_amrex_data.header.real_component_names = ["ux", "uy", "uz"]

    core_mask, supra_mask = mock_amrex_data.extract_core_population(
        velocity_columns=["ux", "uy"],
        v_dim=2,
        cutoff=3.0
    )

    overlap = np.sum(core_mask & true_mask) / np.sum(true_mask)
    assert overlap > 0.95

def test_extract_core_population_3d(mock_amrex_data):
    """Test extract_core_population with 3D data."""
    v_th_core = 1.0
    v_th_supra = 5.0
    v_shift = 0.0
    n_core = 1000
    n_supra = 200
    dim = 3

    data, true_mask = generate_synthetic_data(n_core, n_supra, v_th_core, v_th_supra, v_shift, dim)

    mock_amrex_data.rdata = data
    mock_amrex_data.header = MagicMock()
    mock_amrex_data.header.real_component_names = ["ux", "uy", "uz"]

    core_mask, supra_mask = mock_amrex_data.extract_core_population(
        velocity_columns=["ux", "uy", "uz"],
        v_dim=3,
        cutoff=3.0
    )

    overlap = np.sum(core_mask & true_mask) / np.sum(true_mask)
    assert overlap > 0.95
