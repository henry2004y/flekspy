import pytest
from flekspy.amrex import AMReXParticleData
import numpy as np
from unittest.mock import patch, MagicMock
import os


@pytest.fixture(scope="module")
def particle_data(setup_test_data):
    """Fixture to load the test particle data."""
    plotfile_directory = os.path.join(
        setup_test_data, "3d_particle_region0_1_t00000002_n00000007_amrex"
    )
    return AMReXParticleData(plotfile_directory)


@pytest.fixture
def mock_plot_components():
    """Fixture to mock matplotlib components for plotting tests."""
    with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
        "flekspy.amrex.make_axes_locatable"
    ) as mock_make_axes_locatable:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_divider = MagicMock()
        mock_cax = MagicMock()
        mock_make_axes_locatable.return_value = mock_divider
        mock_divider.append_axes.return_value = mock_cax

        yield {
            "subplots": mock_subplots,
            "make_axes_locatable": mock_make_axes_locatable,
            "fig": mock_fig,
            "ax": mock_ax,
            "divider": mock_divider,
            "cax": mock_cax,
        }


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
    assert idata.shape[0] == 256423  # Check number of particles from header


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


def test_plot_phase(mock_plot_components):
    """
    Tests the plot_phase function by mocking the plotting backend.
    """
    mock_fig = mock_plot_components["fig"]
    mock_ax = mock_plot_components["ax"]

    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y", "vx", "vy", "weight"]
    mock_pdata.rdata = np.random.rand(100, 5)

    result_fig, result_ax = AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="x",
        y_variable="vy",
        normalize=True,
        title="Test Title",
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
        cmap="viridis",
    )

    assert result_fig is mock_fig
    assert result_ax is mock_ax
    mock_plot_components["subplots"].assert_called_once_with(figsize=(8, 6))
    assert mock_ax.imshow.called
    imshow_kwargs = mock_ax.imshow.call_args.kwargs
    assert imshow_kwargs["cmap"] == "viridis"
    mock_ax.set_title.assert_called_once_with("Test Title", fontsize="x-large")
    mock_ax.set_xlabel.assert_called_once_with("Custom X Label", fontsize="x-large")
    mock_ax.set_ylabel.assert_called_once_with("Custom Y Label", fontsize="x-large")
    mock_plot_components["make_axes_locatable"].assert_called_once_with(mock_ax)
    mock_plot_components["divider"].append_axes.assert_called_once_with(
        "right", size="3%", pad=0.05
    )
    mock_fig.colorbar.assert_called_once()
    cbar_instance = mock_fig.colorbar.return_value
    cbar_instance.set_label.assert_called_once_with("Normalized Weighted Density")


@patch("flekspy.amrex.logger")
def test_plot_phase_no_particles(mock_logger, mock_plot_components):
    """
    Tests that plot_phase logs a warning and returns early
    when there are no particles to plot.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.rdata = np.empty((0, 5))  # No particles

    AMReXParticleData.plot_phase(mock_pdata, x_variable="x", y_variable="y")

    mock_logger.warning.assert_called_once_with("No particles to plot.")
    mock_plot_components["subplots"].assert_not_called()


@patch("numpy.histogram2d")
def test_plot_phase_with_hist_range(mock_histogram2d, mock_plot_components):
    """
    Tests that the hist_range parameter is correctly passed to numpy.histogram2d.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.random.rand(100, 2)
    mock_histogram2d.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
    )

    custom_range = [[0.1, 0.9], [0.2, 0.8]]
    AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", hist_range=custom_range
    )

    mock_histogram2d.assert_called_once()
    _, _, kwargs = mock_histogram2d.mock_calls[0]
    assert "range" in kwargs
    assert kwargs["range"] == custom_range