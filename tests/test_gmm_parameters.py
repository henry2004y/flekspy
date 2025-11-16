import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from unittest.mock import MagicMock

from flekspy.amrex.particle_data import AMReXParticleData


@pytest.fixture
def fitted_gmm():
    """Provides a mock fitted GaussianMixture object."""
    gmm = MagicMock(spec=GaussianMixture)
    gmm.n_components = 2
    gmm.means_ = np.array([[0.0, 1.0], [2.0, 3.0]])
    gmm.covariances_ = np.array([[[1.0, 0.5], [0.5, 2.0]], [[3.0, -0.2], [-0.2, 4.0]]])
    return gmm


def test_get_gmm_parameters_isotropic(fitted_gmm):
    """
    Tests the extraction of isotropic temperature from a GMM.
    """
    from scipy.constants import m_u, k

    # Test with explicit particle_mass in amu
    parameters = AMReXParticleData.get_gmm_parameters(
        fitted_gmm, particle_mass=1.0, isotropic=True
    )

    assert len(parameters) == 2
    # v_th_sq_1 = (1.0 + 2.0) / 2.0 = 1.5
    # T_1 = (1.0 * m_u) * v_th_sq_1 / k
    expected_temp_1 = 1.0 * m_u * 1.5 / k
    assert parameters[0]["temperature"] == pytest.approx(expected_temp_1)

    # v_th_sq_2 = (3.0 + 4.0) / 2.0 = 3.5
    # T_2 = (1.0 * m_u) * v_th_sq_2 / k
    expected_temp_2 = 1.0 * m_u * 3.5 / k
    assert parameters[1]["temperature"] == pytest.approx(expected_temp_2)

    # Test with default particle_mass (should be 1.0 amu)
    parameters_default = AMReXParticleData.get_gmm_parameters(
        fitted_gmm, isotropic=True
    )
    assert parameters_default[0]["temperature"] == pytest.approx(expected_temp_1)
    assert parameters_default[1]["temperature"] == pytest.approx(expected_temp_2)


def test_get_gmm_parameters_bi_maxwellian(fitted_gmm):
    """
    Tests the extraction of Bi-Maxwellian temperatures from a GM.
    """
    from scipy.constants import m_u, k

    parameters = AMReXParticleData.get_gmm_parameters(
        fitted_gmm, particle_mass=1.0, isotropic=False
    )

    assert len(parameters) == 2

    # T_par_1 = (1.0 * m_u) * 1.0 / k
    expected_t_par_1 = 1.0 * m_u * 1.0 / k
    # T_perp_1 = (1.0 * m_u) * 2.0 / k
    expected_t_perp_1 = 1.0 * m_u * 2.0 / k
    assert parameters[0]["T_parallel"] == pytest.approx(expected_t_par_1)
    assert parameters[0]["T_perpendicular"] == pytest.approx(expected_t_perp_1)

    # T_par_2 = (1.0 * m_u) * 3.0 / k
    expected_t_par_2 = 1.0 * m_u * 3.0 / k
    # T_perp_2 = (1.0 * m_u) * 4.0 / k
    expected_t_perp_2 = 1.0 * m_u * 4.0 / k
    assert parameters[1]["T_parallel"] == pytest.approx(expected_t_par_2)
    assert parameters[1]["T_perpendicular"] == pytest.approx(expected_t_perp_2)
