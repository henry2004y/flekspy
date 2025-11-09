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
@pytest.mark.parametrize("plot_method_name", [
    "plot_phase_3d",
    "plot_intersecting_planes",
])
def test_3d_plots(setup_test_data, data_file, plot_method_name):
    """
    Test 3D plotting methods to ensure they run without errors and return
    a PyVista plotter object.
    """
    data_dir = Path(setup_test_data)
    ds = AMReXParticleData(data_dir / data_file)
    plot_func = getattr(ds, plot_method_name)

    with patch("pyvista.Plotter.show"):
        plotter = plot_func("x", "y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"

    # Test with normalization
    with patch("pyvista.Plotter.show"):
        plotter = plot_func("x", "y", "velocity_z", normalize=True)
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"

    # Test with different variables
    with patch("pyvista.Plotter.show"):
        plotter = plot_func("velocity_x", "velocity_y", "velocity_z")
        assert plotter is not None, "Plotter should not be None"
        assert isinstance(plotter, pv.BasePlotter), "Should return a PyVista plotter"
