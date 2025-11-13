import pytest
from flekspy.amrex import AMReXParticleData
from flekspy.amrex.plotting import AMReXPlottingMixin
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import os


# To test the mixin, we need a class that uses it
class MockAMReXData(AMReXPlottingMixin):
    def __init__(self, rdata, header):
        self.rdata = rdata
        self.header = header

    def select_particles_in_region(self, x_range=None, y_range=None, z_range=None):
        return self.rdata


@pytest.fixture
def mock_amrex_data():
    """Creates a mock AMReXParticleData object for testing."""
    # Mock header
    header = MagicMock()
    header.real_component_names = ["x", "y", "z", "velocity_x", "velocity_y", "velocity_z", "weight"]

    # Mock particle data
    # Create 100 particles with random data
    rdata = np.random.rand(100, 7)

    return MockAMReXData(rdata, header)


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
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("mpl_toolkits.axes_grid1.make_axes_locatable") as mock_make_axes_locatable,
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
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Normalized Weighted Density",
    )

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
    mock_pdata.get_phase_space_density.assert_called_once()
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

    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    result_fig, result_ax = AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", ax=mock_ax
    )

    assert result_fig is mock_fig
    assert result_ax is mock_ax
    mock_plot_components["subplots"].assert_not_called()
    assert mock_ax.imshow.called


def test_plot_phase_no_colorbar(mock_plot_components):
    """
    Tests that the colorbar is not created when add_colorbar=False.
    """
    mock_fig = mock_plot_components["fig"]

    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", add_colorbar=False
    )

    mock_plot_components["make_axes_locatable"].assert_not_called()
    mock_fig.colorbar.assert_not_called()


def test_plot_phase_no_particles(mock_plot_components):
    """
    Tests that plot_phase returns early when there are no particles.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = None

    result = AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", x_range=(0, 1)
    )

    assert result is None
    mock_plot_components["subplots"].assert_not_called()


def test_plot_phase_with_hist_range(mock_plot_components):
    """
    Tests that the hist_range parameter is correctly passed.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    custom_range = [[0.1, 0.9], [0.2, 0.8]]
    AMReXParticleData.plot_phase(
        mock_pdata, x_variable="x", y_variable="y", hist_range=custom_range
    )

    mock_pdata.get_phase_space_density.assert_called_once()
    _, kwargs = mock_pdata.get_phase_space_density.call_args
    assert "hist_range" in kwargs
    assert kwargs["hist_range"] == custom_range


def test_plot_phase_log_scale_with_vmin_vmax(mock_plot_components):
    """
    Tests that vmin and vmax are correctly used in log scale.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10) + 0.1,
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    with patch("matplotlib.colors.LogNorm") as mock_log_norm:
        AMReXParticleData.plot_phase(
            mock_pdata, x_variable="x", y_variable="y", log_scale=True, vmin=1, vmax=10
        )
        mock_log_norm.assert_called_once_with(vmin=1, vmax=10)


def test_plot_phase_subplots():
    """
    Tests the plot_phase_subplots function by mocking the plotting backend.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Weighted Particle Density",
    )
    mock_pdata._get_axis_label.side_effect = lambda x: x

    x_ranges = [(-1, 1), (-2, 2)]
    y_ranges = [(-1, 1), (-2, 2)]

    fig_mock = MagicMock()
    axes_mock = np.empty((1, 2), dtype=object)
    axes_mock[0, 0] = MagicMock()
    axes_mock[0, 1] = MagicMock()

    with patch(
        "matplotlib.pyplot.subplots", return_value=(fig_mock, axes_mock)
    ) as mock_subplots:
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
        assert mock_pdata.get_phase_space_density.call_count == 2

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
    mock_pdata.get_phase_space_density.side_effect = [
        (
            np.random.rand(10, 10),
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11),
            "Weighted Particle Density",
        ),
        None,
    ]
    mock_pdata._get_axis_label.side_effect = lambda x: x

    x_ranges = [(-1, 1), (-2, 2)]
    y_ranges = [(-1, 1), (-2, 2)]

    fig_mock = MagicMock()
    axes_mock = np.empty((1, 2), dtype=object)
    axes_mock[0, 0] = MagicMock()
    axes_mock[0, 1] = MagicMock()

    with patch(
        "matplotlib.pyplot.subplots", return_value=(fig_mock, axes_mock)
    ):
        # This should execute without raising a ValueError
        AMReXParticleData.plot_phase_subplots(
            mock_pdata,
            x_variable="x",
            y_variable="vy",
            x_ranges=x_ranges,
            y_ranges=y_ranges,
        )


def test_pairplot(mock_amrex_data):
    """
    Tests the pairplot function.
    """
    fig_mock = MagicMock()
    axes_mock = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes_mock[i, j] = MagicMock()

    with patch(
        "matplotlib.pyplot.subplots", return_value=(fig_mock, axes_mock)
    ) as mock_subplots:
        result_fig, result_axes = mock_amrex_data.pairplot()

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


@patch("matplotlib.pyplot.show")
def test_pairplot_corner(mock_show, mock_amrex_data):
    """
    Tests that the pairplot method with corner=True only plots the lower
    triangle of the plot matrix.
    """
    fig, axes = mock_amrex_data.pairplot(
        variables=["velocity_x", "velocity_y", "velocity_z"],
        corner=True
    )

    # The axes that should be visible
    visible_axes = [
        (0, 0),
        (1, 0), (1, 1),
        (2, 0), (2, 1), (2, 2)
    ]

    nvar = len(axes)
    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]
            if (i, j) in visible_axes:
                assert ax.get_visible(), f"Axis ({i}, {j}) should be visible"
            else:
                assert not ax.get_visible(), f"Axis ({i}, {j}) should NOT be visible"

    plt.close(fig)


def test_plot_phase_with_transform(mock_plot_components):
    """
    Tests that the transform function is correctly applied.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    def dummy_transform(data):
        return data, ["x_new", "y_new"]

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="x_new",
        y_variable="y_new",
        transform=dummy_transform,
    )

    mock_pdata.get_phase_space_density.assert_called_once()
    _, kwargs = mock_pdata.get_phase_space_density.call_args
    assert "transform" in kwargs
    assert kwargs["transform"] == dummy_transform


def test_plot_phase_with_kde(mock_plot_components):
    """
    Tests that the KDE parameters are correctly passed.
    """
    mock_fig = mock_plot_components["fig"]
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(50, 50),
        np.linspace(0, 1, 51),
        np.linspace(0, 1, 51),
        "Weighted Density",
    )

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="x",
        y_variable="y",
        use_kde=True,
        kde_bandwidth="silverman",
        kde_grid_size=50,
    )

    mock_pdata.get_phase_space_density.assert_called_once_with(
        x_variable="x",
        y_variable="y",
        bins=100,
        hist_range=None,
        x_range=None,
        y_range=None,
        z_range=None,
        normalize=False,
        use_kde=True,
        kde_bandwidth="silverman",
        kde_grid_size=50,
        transform=None,
    )
    mock_fig.colorbar.assert_called_once()
    cbar_instance = mock_fig.colorbar.return_value
    cbar_instance.set_label.assert_called_once_with("Weighted Density")


def test_plot_phase_with_spatial_transform(mock_plot_components):
    """
    Tests passing a spatial transform function.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    def spatial_transform(data):
        return data, ["pos_parallel", "pos_perp"]

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="pos_parallel",
        y_variable="pos_perp",
        transform=spatial_transform,
    )

    mock_pdata.get_phase_space_density.assert_called_once()
    _, kwargs = mock_pdata.get_phase_space_density.call_args
    assert kwargs["transform"] == spatial_transform


def test_plot_phase_with_field_aligned_transform(mock_plot_components):
    """
    Tests passing a field-aligned transform function.
    """
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.get_phase_space_density.return_value = (
        np.random.rand(10, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 11),
        "Particle Count",
    )

    def field_aligned_transform(data):
        return data, [
            "x",
            "y",
            "z",
            "v_perp2",
            "v_parallel",
            "v_perp1",
        ]

    AMReXParticleData.plot_phase(
        mock_pdata,
        x_variable="v_parallel",
        y_variable="v_perp1",
        transform=field_aligned_transform,
    )

    mock_pdata.get_phase_space_density.assert_called_once()
    _, kwargs = mock_pdata.get_phase_space_density.call_args
    assert kwargs["transform"] == field_aligned_transform


def test_get_phase_space_density_basic():
    """Tests the basic functionality of get_phase_space_density."""
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y", "weight"]
    mock_pdata.rdata = np.random.rand(100, 3)
    mock_pdata.select_particles_in_region.return_value = mock_pdata.rdata

    with patch("numpy.histogram2d") as mock_histogram2d:
        mock_histogram2d.return_value = (
            np.array([[1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        H, xedges, yedges, cbar_label = AMReXParticleData.get_phase_space_density(
            mock_pdata, x_variable="x", y_variable="y"
        )

        mock_histogram2d.assert_called_once()
        _, kwargs = mock_histogram2d.call_args
        assert "weights" in kwargs
        np.testing.assert_array_equal(kwargs["weights"], mock_pdata.rdata[:, 2])
        assert cbar_label == "Weighted Particle Density"


def test_get_phase_space_density_normalized():
    """Tests the normalization in get_phase_space_density."""
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.random.rand(100, 2)
    # Since select_particles_in_region is not mocked, it will return None,
    # and get_phase_space_density will use mock_pdata.rdata
    mock_pdata.select_particles_in_region.return_value = mock_pdata.rdata

    H, _, _, cbar_label = AMReXParticleData.get_phase_space_density(
        mock_pdata, x_variable="x", y_variable="y", normalize=True
    )

    assert np.isclose(H.sum(), 1.0)
    assert cbar_label == "Normalized Density"


def test_get_phase_space_density_with_transform():
    """Tests the transform functionality in get_phase_space_density."""
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    original_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_pdata.rdata = original_data.copy()

    def scale_transform(data):
        return data * 2, ["x_scaled", "y_scaled"]

    with patch("numpy.histogram2d") as mock_histogram2d:
        mock_histogram2d.return_value = (
            np.array([[1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        AMReXParticleData.get_phase_space_density(
            mock_pdata,
            x_variable="x_scaled",
            y_variable="y_scaled",
            transform=scale_transform,
        )

        mock_histogram2d.assert_called_once()
        call_args, _ = mock_histogram2d.call_args
        x_data_passed = call_args[0]
        y_data_passed = call_args[1]
        expected_x = original_data[:, 0] * 2
        expected_y = original_data[:, 1] * 2
        np.testing.assert_array_equal(x_data_passed, expected_x)
        np.testing.assert_array_equal(y_data_passed, expected_y)


def test_get_phase_space_density_with_kde():
    """Tests the KDE functionality in get_phase_space_density."""
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y", "weight"]
    mock_pdata.rdata = np.random.rand(100, 3)
    weights = mock_pdata.rdata[:, 2]

    with patch("scipy.stats.gaussian_kde") as mock_gaussian_kde:
        mock_kde_instance = MagicMock()
        mock_kde_instance.return_value = np.random.rand(10 * 10)
        mock_gaussian_kde.return_value = mock_kde_instance

        _, _, _, cbar_label = AMReXParticleData.get_phase_space_density(
            mock_pdata,
            x_variable="x",
            y_variable="y",
            use_kde=True,
            kde_grid_size=10,
        )

        mock_gaussian_kde.assert_called_once()
        _, kwargs = mock_gaussian_kde.call_args
        assert "weights" in kwargs
        np.testing.assert_array_equal(kwargs["weights"], weights)
        assert cbar_label == "Weighted Density"


def test_get_phase_space_density_particle_selection():
    """Tests particle selection in get_phase_space_density."""
    mock_pdata = MagicMock(spec=AMReXParticleData)
    mock_pdata.header = MagicMock()
    mock_pdata.header.real_component_names = ["x", "y"]
    mock_pdata.rdata = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])

    # Mock select_particles_in_region to filter based on range
    def mock_select(x_range=None, y_range=None, z_range=None):
        data = mock_pdata.rdata
        if x_range:
            data = data[(data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1])]
        if y_range:
            data = data[(data[:, 1] >= y_range[0]) & (data[:, 1] <= y_range[1])]
        return data

    mock_pdata.select_particles_in_region.side_effect = mock_select

    H, _, _, _ = AMReXParticleData.get_phase_space_density(
        mock_pdata,
        x_variable="x",
        y_variable="y",
        x_range=(0.4, 0.6),
        bins=1
    )

    # histogram should be based on 1 particle
    assert H.sum() == 1
