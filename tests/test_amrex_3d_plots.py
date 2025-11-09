import pytest
from unittest.mock import patch
import pyvista as pv
from flekspy.amrex import AMReXParticleData
from pathlib import Path

# Ensure PyVista runs in headless mode
pv.OFF_SCREEN = True

@pytest.mark.parametrize(
    "data_file",
    ["3d_particle_region0_1_t00000002_n00000007_amrex"],
)
def test_plot_phase_3d(setup_test_data, data_file):
    """
    Test the plot_phase_3d method to ensure it runs without errors and returns
    a PyVista plotter object.
    """
    data_dir = Path(setup_test_data)
    ds = AMReXParticleData(data_dir / data_file)

    with patch("pyvista.Plotter.add_volume") as mock_add_volume:
        plotter = ds.plot_phase_3d("x", "y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_volume.assert_called_once()
        plotter.close()

    # Test with vmin and vmax
    with patch("pyvista.Plotter.add_volume") as mock_add_volume:
        plotter = ds.plot_phase_3d(
            "velocity_x", "velocity_y", "velocity_z", vmin=0.1, vmax=0.9
        )
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_volume.assert_called_once()
        _, kwargs = mock_add_volume.call_args
        assert kwargs["clim"] == [0.1, 0.9]
        plotter.close()

    # Test with cmap
    with patch("pyvista.Plotter.add_volume") as mock_add_volume:
        plotter = ds.plot_phase_3d(
            "velocity_x", "velocity_y", "velocity_z", cmap="viridis"
        )
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_volume.assert_called_once()
        _, kwargs = mock_add_volume.call_args
        assert kwargs["cmap"] == "viridis"
        plotter.close()

    # Test with normalization
    with patch("pyvista.Plotter.add_volume") as mock_add_volume:
        plotter = ds.plot_phase_3d("x", "y", "velocity_z", normalize=True)
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_volume.assert_called_once()
        plotter.close()

    # Test with different variables
    with patch("pyvista.Plotter.add_volume") as mock_add_volume:
        plotter = ds.plot_phase_3d("velocity_x", "velocity_y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_volume.assert_called_once()
        plotter.close()


@pytest.mark.parametrize(
    "data_file",
    ["3d_particle_region0_1_t00000002_n00000007_amrex"],
)
def test_plot_intersecting_planes(setup_test_data, data_file):
    """
    Test the plot_intersecting_planes method to ensure it runs without errors and returns
    a PyVista plotter object.
    """
    data_dir = Path(setup_test_data)
    ds = AMReXParticleData(data_dir / data_file)

    with patch("pyvista.Plotter.add_mesh") as mock_add_mesh:
        plotter = ds.plot_intersecting_planes("x", "y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_mesh.assert_called_once()
        plotter.close()

    # Test with normalization
    with patch("pyvista.Plotter.add_mesh") as mock_add_mesh:
        plotter = ds.plot_intersecting_planes("x", "y", "velocity_z", normalize=True)
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_mesh.assert_called_once()
        plotter.close()

    # Test with different variables
    with patch("pyvista.Plotter.add_mesh") as mock_add_mesh:
        plotter = ds.plot_intersecting_planes("velocity_x", "velocity_y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
        mock_add_mesh.assert_called_once()
        plotter.close()
