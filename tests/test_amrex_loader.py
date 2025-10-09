import pytest
from flekspy.amrex import AMReXParticleData
import numpy as np
from unittest.mock import patch, MagicMock

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


@patch('flekspy.amrex.make_axes_locatable')
@patch('matplotlib.pyplot.subplots')
def test_plot_phase(mock_subplots, mock_make_axes_locatable):
    """
    Tests the plot_phase function by mocking the plotting backend.
    This test verifies that the correct matplotlib functions are called
    with the expected arguments, and that the figure and axes are returned.
    """
    # --- 1. Setup Mocks ---
    # Mock the figure and axes objects that subplots() would return
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # Mock the axes divider and colorbar axes
    mock_divider = MagicMock()
    mock_cax = MagicMock()
    mock_make_axes_locatable.return_value = mock_divider
    mock_divider.append_axes.return_value = mock_cax

    # --- 2. Create a mock AMReXParticleData instance ---
    # We create a mock object and manually attach the attributes
    # that `plot_phase` will access.
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ['x', 'y', 'vx', 'vy', 'weight']
    # Create some random data for the plot to process
    mock_pdata.rdata = np.random.rand(100, 5)

    # --- 3. Call the real plot_phase method and capture the return value ---
    result_fig, result_ax = AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable='x',
        y_variable='vy',
        normalize=True,
        title="Test Title",
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
        cmap='viridis'
    )

    # --- 4. Assertions ---
    # Verify that the correct figure and axes are returned
    assert result_fig is mock_fig
    assert result_ax is mock_ax

    # Verify that subplots was called correctly
    mock_subplots.assert_called_once_with(figsize=(8, 6))

    # Verify that imshow was called and received the custom colormap
    assert mock_ax.imshow.called
    imshow_kwargs = mock_ax.imshow.call_args.kwargs
    assert imshow_kwargs['cmap'] == 'viridis'

    # Verify titles and labels were set with the correct text and font size
    mock_ax.set_title.assert_called_once_with("Test Title", fontsize="x-large")
    mock_ax.set_xlabel.assert_called_once_with("Custom X Label", fontsize="x-large")
    mock_ax.set_ylabel.assert_called_once_with("Custom Y Label", fontsize="x-large")

    # Verify colorbar creation and labeling
    mock_make_axes_locatable.assert_called_once_with(mock_ax)
    mock_divider.append_axes.assert_called_once_with("right", size="3%", pad=0.05)
    mock_fig.colorbar.assert_called_once()
    # Check the label reflects that the data was normalized and weighted
    cbar_instance = mock_fig.colorbar.return_value
    cbar_instance.set_label.assert_called_once_with("Normalized Weighted Density")


@patch('flekspy.amrex.logger')
@patch('matplotlib.pyplot.subplots')
def test_plot_phase_no_particles(mock_subplots, mock_logger):
    """
    Tests that plot_phase logs a warning and returns early
    when there are no particles to plot.
    """
    # --- 1. Create a mock AMReXParticleData instance with empty data ---
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.rdata = np.empty((0, 5)) # No particles

    # --- 2. Call the plot_phase method ---
    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable='x',
        y_variable='y'
    )

    # --- 3. Assertions ---
    # Verify that a warning was logged
    mock_logger.warning.assert_called_once_with("No particles to plot.")

    # Verify that no plot was created
    mock_subplots.assert_not_called()