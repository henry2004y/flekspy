import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from unittest.mock import MagicMock

from flekspy.util.gmm import get_gmm_parameters
from flekspy.amrex.particle_data import AMReXParticle


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
    Tests the extraction of isotropic squared thermal velocities from a GMM.
    """
    parameters = get_gmm_parameters(fitted_gmm, isotropic=True)

    assert len(parameters) == 2
    # v_th_sq_1 = (1.0 + 2.0) / 2.0 = 1.5
    assert parameters[0]["v_th_sq"] == pytest.approx(1.5)
    # v_th_sq_2 = (3.0 + 4.0) / 2.0 = 3.5
    assert parameters[1]["v_th_sq"] == pytest.approx(3.5)


def test_get_gmm_parameters_bi_maxwellian(fitted_gmm):
    """
    Tests the extraction of Bi-Maxwellian squared thermal velocities from a GMM.
    """
    parameters = get_gmm_parameters(fitted_gmm, isotropic=False)

    assert len(parameters) == 2
    # v_parallel_sq_1 = 1.0
    assert parameters[0]["v_parallel_sq"] == pytest.approx(1.0)
    # v_perp_sq_1 = 2.0
    assert parameters[0]["v_perp_sq"] == pytest.approx(2.0)
    # v_parallel_sq_2 = 3.0
    assert parameters[1]["v_parallel_sq"] == pytest.approx(3.0)
    # v_perp_sq_2 = 4.0
    assert parameters[1]["v_perp_sq"] == pytest.approx(4.0)


def test_get_gmm_temperatures_isotropic(fitted_gmm):
    """
    Tests the conversion of isotropic thermal velocities to temperatures.
    """
    from scipy.constants import m_u, k

    # Test with explicit particle_mass in amu
    parameters = AMReXParticle.get_gmm_temperatures(
        fitted_gmm, particle_mass=2.0, isotropic=True
    )

    assert len(parameters) == 2
    # T_1 = (2.0 * m_u) * 1.5 / k
    expected_temp_1 = 2.0 * m_u * 1.5 / k
    assert parameters[0]["temperature"] == pytest.approx(expected_temp_1)

    # T_2 = (2.0 * m_u) * 3.5 / k
    expected_temp_2 = 2.0 * m_u * 3.5 / k
    assert parameters[1]["temperature"] == pytest.approx(expected_temp_2)

    # Test with default particle_mass (should be 1.0 amu)
    parameters_default = AMReXParticle.get_gmm_temperatures(
        fitted_gmm, isotropic=True
    )
    expected_temp_1_default = 1.0 * m_u * 1.5 / k
    expected_temp_2_default = 1.0 * m_u * 3.5 / k
    assert parameters_default[0]["temperature"] == pytest.approx(
        expected_temp_1_default
    )
    assert parameters_default[1]["temperature"] == pytest.approx(
        expected_temp_2_default
    )


def test_get_gmm_temperatures_bi_maxwellian(fitted_gmm):
    """
    Tests the conversion of Bi-Maxwellian thermal velocities to temperatures.
    """
    from scipy.constants import m_u, k

    parameters = AMReXParticle.get_gmm_temperatures(
        fitted_gmm, particle_mass=2.0, isotropic=False
    )

    assert len(parameters) == 2

    # T_par_1 = (2.0 * m_u) * 1.0 / k
    expected_t_par_1 = 2.0 * m_u * 1.0 / k
    # T_perp_1 = (2.0 * m_u) * 2.0 / k
    expected_t_perp_1 = 2.0 * m_u * 2.0 / k
    assert parameters[0]["T_parallel"] == pytest.approx(expected_t_par_1)
    assert parameters[0]["T_perpendicular"] == pytest.approx(expected_t_perp_1)

    # T_par_2 = (2.0 * m_u) * 3.0 / k
    expected_t_par_2 = 2.0 * m_u * 3.0 / k
    # T_perp_2 = (2.0 * m_u) * 4.0 / k
    expected_t_perp_2 = 2.0 * m_u * 4.0 / k
    assert parameters[1]["T_parallel"] == pytest.approx(expected_t_par_2)
    assert parameters[1]["T_perpendicular"] == pytest.approx(expected_t_perp_2)
