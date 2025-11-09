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
    with (
        patch("flekspy.amrex.plotting.plt.subplots") as mock_subplots,
        patch("flekspy.amrex.plotting.make_axes_locatable") as mock_make_axes_locatable,
    ):
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
    assert imshow_kwargs["cmap"].name == "viridis"
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


def test_plot_phase_with_existing_axes(mock_plot_components):
    """
    Tests that plot_phase can draw on an existing matplotlib axes.
    """
    mock_fig = mock_plot_components["fig"]
    mock_ax = mock_plot_components["ax"]
    mock_ax.figure = mock_fig

    # This time, we pass the existing ax to the function
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.random.rand(100, 2)

    result_fig, result_ax = AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", ax=mock_ax
    )

    # Assert that the returned objects are the same ones we passed in
    assert result_fig is mock_fig
    assert result_ax is mock_ax

    # Assert that no new subplots were created
    mock_plot_components["subplots"].assert_not_called()

    # Assert that the plotting was done on the provided axes
    assert mock_ax.imshow.called


def test_plot_phase_no_colorbar(mock_plot_components):
    """
    Tests that the colorbar is not created when add_colorbar=False.
    """
    mock_fig = mock_plot_components["fig"]
    mock_ax = mock_plot_components["ax"]

    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.random.rand(100, 2)

    AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", add_colorbar=False
    )

    # Assert that the colorbar creation logic was not called
    mock_plot_components["make_axes_locatable"].assert_not_called()
    mock_fig.colorbar.assert_not_called()


@patch("flekspy.amrex.plotting.logger")
def test_plot_phase_no_particles(mock_logger, mock_plot_components):
    """
    Tests that plot_phase logs a warning and returns early
    when there are no particles to plot.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.rdata = np.empty((0, 5))  # No particles
    mock_pdata.select_particles_in_region.return_value = np.empty((0, 5))

    AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", x_range=(0, 1)
    )

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


def test_plot_phase_log_scale_with_vmin_vmax(mock_plot_components):
    """
    Tests that vmin and vmax are correctly used in log scale.
    """
    mock_ax = mock_plot_components["ax"]
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.random.rand(100, 2) + 0.1  # Ensure data is > 0 for log

    with patch("flekspy.amrex.plotting.colors.LogNorm") as mock_log_norm:
        AMReXParticleData.plot_phase(
            mock_pdata, x_variable="x", y_variable="y", log_scale=True, vmin=1, vmax=10
        )
        mock_log_norm.assert_called_once_with(vmin=1, vmax=10)


def test_plot_phase_subplots():
    """
    Tests the plot_phase_subplots function by mocking the plotting backend.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y", "vx", "vy", "weight"]
    mock_pdata.select_particles_in_region.return_value = np.random.rand(50, 5)

    x_ranges = [(-1, 1), (-2, 2)]
    y_ranges = [(-1, 1), (-2, 2)]

    fig_mock = MagicMock()
    axes_mock = np.empty((1, 2), dtype=object)
    axes_mock[0, 0] = MagicMock()
    axes_mock[0, 1] = MagicMock()

    with (
        patch(
            "flekspy.amrex.plotting.plt.subplots", return_value=(fig_mock, axes_mock)
        ) as mock_subplots,
        patch(
            "numpy.histogram2d",
            return_value=(
                np.random.rand(10, 10),
                np.linspace(0, 1, 11),
                np.linspace(0, 1, 11),
            ),
        ) as mock_hist,
    ):
        result_fig, result_axes = AMReXParticleData.plot_phase_subplots(
            mock_pdata,
            x_variable="x",
            y_variable="vy",
            x_ranges=x_ranges,
            y_ranges=y_ranges,
            suptitle="Test Subplots",
        )

        assert result_fig is fig_mock
        mock_subplots.assert_called_once()
        assert mock_pdata.select_particles_in_region.call_count == 2
        assert mock_hist.call_count == 2

        # Check that 'range' is not in kwargs for histogram2d
        _, _, kwargs = mock_hist.mock_calls[0]
        assert "range" not in kwargs

        # Check that imshow was called on the axes that were used
        for i in range(len(x_ranges)):
            ax = result_axes.flatten()[i]
            assert ax.imshow.called

        result_fig.colorbar.assert_called_once()
        cbar_instance = result_fig.colorbar.return_value
        cbar_instance.set_label.assert_called_once_with("Weighted Particle Density")
        result_fig.suptitle.assert_called_once_with("Test Subplots", fontsize="x-large")


def test_plot_phase_subplots_empty_region():
    """
    Tests that plot_phase_subplots handles an empty region without crashing.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y", "vx", "vy", "weight"]

    # One region with data, one without
    mock_pdata.select_particles_in_region.side_effect = [
        np.random.rand(50, 5),
        np.empty((0, 5)),
    ]

    x_ranges = [(-1, 1), (-2, 2)]
    y_ranges = [(-1, 1), (-2, 2)]

    fig_mock = MagicMock()
    axes_mock = np.empty((1, 2), dtype=object)
    axes_mock[0, 0] = MagicMock()
    axes_mock[0, 1] = MagicMock()

    with patch(
        "flekspy.amrex.plotting.plt.subplots", return_value=(fig_mock, axes_mock)
    ):
        # This should execute without raising a ValueError
        AMReXParticleData.plot_phase_subplots(
            mock_pdata,
            x_variable="x",
            y_variable="vy",
            x_ranges=x_ranges,
            y_ranges=y_ranges,
        )


def test_pairplot():
    """
    Tests the pairplot function.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = [
        "x",
        "y",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "weight",
    ]
    mock_pdata.rdata = np.random.rand(100, 6)

    fig_mock = MagicMock()
    axes_mock = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes_mock[i, j] = MagicMock()

    with patch(
        "flekspy.amrex.plotting.plt.subplots", return_value=(fig_mock, axes_mock)
    ) as mock_subplots:
        result_fig, result_axes = AMReXParticleData.pairplot(mock_pdata)

        assert result_fig is fig_mock
        assert np.array_equal(result_axes, axes_mock)
        mock_subplots.assert_called_once_with(
            3, 3, figsize=(10, 10), constrained_layout=True
        )

        # Verify histograms were called
        for i in range(3):
            for j in range(3):
                ax = result_axes[i, j]
                if i == j:
                    ax.hist.assert_called_once()
                else:
                    ax.imshow.assert_called_once()


@patch("numpy.histogram2d")
def test_plot_phase_with_transform(mock_histogram2d, mock_plot_components):
    """
    Tests that the transform function is correctly applied and that the
    new component names are used.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    original_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_pdata.rdata = original_data.copy()

    # Define a transformation function that returns new data and new names
    def scale_and_rename_transform(data):
        transformed_data = data * 2
        new_names = ["x_scaled", "y_scaled"]
        return transformed_data, new_names

    mock_histogram2d.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
    )

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="x_scaled",
        y_variable="y_scaled",
        transform=scale_and_rename_transform,
    )

    # Verify that the data passed to histogram2d is the transformed data
    mock_histogram2d.assert_called_once()
    call_args = mock_histogram2d.call_args[0]
    x_data_passed = call_args[0]
    y_data_passed = call_args[1]

    expected_x_data = original_data[:, 0] * 2
    expected_y_data = original_data[:, 1] * 2

    np.testing.assert_array_almost_equal(x_data_passed, expected_x_data)
    np.testing.assert_array_almost_equal(y_data_passed, expected_y_data)


@patch("numpy.histogram2d")
def test_plot_phase_with_field_aligned_transform(
    mock_histogram2d, mock_plot_components
):
    """
    Tests the transform functionality with a realistic field-aligned
    coordinate transformation.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = [
        "x",
        "y",
        "z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]

    original_data = np.random.rand(100, 6)
    mock_pdata.rdata = original_data.copy()

    # Define a magnetic field direction and create the rotation matrix
    B = np.array([0.0, 1.0, 0.0])
    b_hat = B / np.linalg.norm(B)

    e1 = np.array([1.0, 0.0, 0.0])
    u1 = np.cross(b_hat, e1)
    u1_hat = u1 / np.linalg.norm(u1)

    u2_hat = np.cross(b_hat, u1_hat)

    # Rotation matrix to transform from (vx, vy, vz) to (v_perp2, v_parallel, v_perp1)
    # The rows are the new basis vectors in the old coordinate system.
    rotation_matrix = np.array([u2_hat, b_hat, u1_hat])

    def field_aligned_transform(data):
        # Extract velocity components
        velocities = data[:, 3:6]
        # Apply rotation
        transformed_velocities = np.dot(velocities, rotation_matrix.T)

        # Create the new data array with transformed velocities
        transformed_data = data.copy()
        transformed_data[:, 3:6] = transformed_velocities

        new_names = [
            "x",
            "y",
            "z",
            "v_perp2",
            "v_parallel",
            "v_perp1",
        ]
        return transformed_data, new_names

    mock_histogram2d.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
    )

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="v_parallel",
        y_variable="v_perp1",
        transform=field_aligned_transform,
    )

    mock_histogram2d.assert_called_once()
    call_args = mock_histogram2d.call_args[0]
    x_data_passed = call_args[0]
    y_data_passed = call_args[1]

    # Calculate the expected data that should be passed to the histogram
    original_velocities = original_data[:, 3:6]
    transformed_velocities = np.dot(original_velocities, rotation_matrix.T)
    expected_x_data = transformed_velocities[:, 1]  # v_parallel
    expected_y_data = transformed_velocities[:, 2]  # v_perp1

    np.testing.assert_array_almost_equal(x_data_passed, expected_x_data)
    np.testing.assert_array_almost_equal(y_data_passed, expected_y_data)
