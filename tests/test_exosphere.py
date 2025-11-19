import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from flekspy.util.exosphere import Exosphere
from scipy.constants import G, k as k_B


def test_exosphere_initialization():
    """Test the initialization of the Exosphere class."""
    exo = Exosphere()
    assert exo.neutral_profile == "exponential"
    assert exo.n0 == 1.0e10
    assert exo.H0 == 100.0e3
    assert exo.T0 == 1000.0
    assert exo.k0 == 2.0
    assert exo.exobase_radius == 6371.0e3


@pytest.mark.parametrize(
    "profile, kwargs, expected_func",
    [
        (
            "exponential",
            {"n0": 1e10, "H0": 100e3},
            lambda r, exo: exo.n0 * np.exp(-(r - exo.exobase_radius) / exo.H0),
        ),
        (
            "power_law",
            {"n0": 1e10, "k0": 2},
            lambda r, exo: exo.n0 * (exo.exobase_radius / r) ** exo.k0,
        ),
        (
            "chamberlain",
            {"n0": 1e10, "T0": 1000},
            lambda r, exo: exo.n0
            * np.exp(
                (G * exo.M_planet * exo.m_neutral / (k_B * exo.T0 * exo.exobase_radius))
                * (exo.exobase_radius / r - 1)
            ),
        ),
    ],
)
def test_get_neutral_density(profile, kwargs, expected_func):
    """Test the neutral density profiles."""
    exo = Exosphere(neutral_profile=profile, exobase_radius=6371e3, **kwargs)
    r = np.array([6371e3, 6471e3, 6571e3])
    densities = exo.get_neutral_density(r)
    expected_densities = expected_func(r, exo)
    expected_densities[r < exo.exobase_radius] = 0
    assert np.allclose(densities, expected_densities)


def test_get_neutral_density_invalid_profile():
    """Test that an invalid profile raises a ValueError."""
    exo = Exosphere()
    exo.neutral_profile = "invalid_profile"
    with pytest.raises(ValueError):
        exo.get_neutral_density(np.array([6371e3]))


@patch("matplotlib.pyplot.subplots")
def test_plot_neutral_profile(mock_subplots):
    """Test the plotting function."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    exo = Exosphere()
    fig, ax = exo.plot_neutral_profile()

    assert fig is mock_fig
    assert ax is mock_ax
    mock_ax.plot.assert_called_once()
    mock_ax.set_xlabel.assert_called_with("Altitude [km]")
    mock_ax.set_ylabel.assert_called_with("Neutral Density [m⁻³]")
    mock_ax.set_title.assert_called_with(f"Neutral Density Profile: {exo.neutral_profile}")
    mock_ax.grid.assert_called_with(True)
    mock_ax.set_yscale.assert_called_with("log")
