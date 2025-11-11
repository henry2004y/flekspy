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
    parameters = AMReXParticleData.get_gmm_parameters(fitted_gmm, isotropic=True)

    assert len(parameters) == 2

    # Component 1
    assert "center" in parameters[0]
    assert "temperature" in parameters[0]
    np.testing.assert_almost_equal(parameters[0]["center"], [0.0, 1.0])
    # T = (1.0 + 2.0) / 2.0 = 1.5
    assert parameters[0]["temperature"] == pytest.approx(1.5)

    # Component 2
    assert "center" in parameters[1]
    assert "temperature" in parameters[1]
    np.testing.assert_almost_equal(parameters[1]["center"], [2.0, 3.0])
    # T = (3.0 + 4.0) / 2.0 = 3.5
    assert parameters[1]["temperature"] == pytest.approx(3.5)


def test_get_gmm_parameters_bi_maxwellian(fitted_gmm):
    """
    Tests the extraction of Bi-Maxwellian temperatures from a GMM.
    """
    parameters = AMReXParticleData.get_gmm_parameters(fitted_gmm, isotropic=False)

    assert len(parameters) == 2

    # Component 1
    assert "center" in parameters[0]
    assert "T_parallel" in parameters[0]
    assert "T_perpendicular" in parameters[0]
    np.testing.assert_almost_equal(parameters[0]["center"], [0.0, 1.0])
    assert parameters[0]["T_parallel"] == pytest.approx(1.0)
    assert parameters[0]["T_perpendicular"] == pytest.approx(2.0)

    # Component 2
    assert "center" in parameters[1]
    assert "T_parallel" in parameters[1]
    assert "T_perpendicular" in parameters[1]
    np.testing.assert_almost_equal(parameters[1]["center"], [2.0, 3.0])
    assert parameters[1]["T_parallel"] == pytest.approx(3.0)
    assert parameters[1]["T_perpendicular"] == pytest.approx(4.0)
