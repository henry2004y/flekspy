"""Tests for reconnection analysis, vector potential, flux, and accessor."""

import pytest
import numpy as np
import xarray as xr
from flekspy.reconnection.analysis import (
    calc_vector_potential,
    calc_reconnected_flux,
    ReconnectionSeries,
)
from flekspy.reconnection.plotting import plot_2d_slice, plot_reconnection_rate


def create_synthetic_harris_sheet(nx=64, ny=32, B0=1.0, L=1.0, b1=0.1):
    """Create a synthetic 2D Harris sheet with X-point perturbation."""
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Harris sheet: Bx = B0 * tanh(y/L), By = -b1 * (pi/Lx) * cos(2*pi*x/Lx) * sin(pi*y/Ly)
    # Az0 = B0 * L * ln(cosh(y/L))
    # Perturbation psi1 = b1 * cos(2*pi*x/Lx) * cos(pi*y/Ly)
    Lx = 20.0
    Ly = 10.0
    bx = B0 * np.tanh(Y / L) - b1 * (np.pi / Ly) * np.cos(2 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    by = b1 * (2 * np.pi / Lx) * np.sin(2 * np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    bz = 0.2 * np.sin(2 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)  # Quadrupolar Hall field
    ux = 0.5 * np.sin(2 * np.pi * X / Lx)
    ez = 0.1 * np.ones_like(X)

    ds = xr.Dataset(
        {
            "Bx": (("x", "y"), bx),
            "By": (("x", "y"), by),
            "Bz": (("x", "y"), bz),
            "ux": (("x", "y"), ux),
            "Ez": (("x", "y"), ez),
        },
        coords={"x": x, "y": y},
        attrs={"time": 1.0, "iter": 100, "filename": "synthetic.out"},
    )
    return ds


def test_calc_vector_potential():
    """Test 2D vector potential calculation on synthetic field."""
    ds = create_synthetic_harris_sheet()
    az = calc_vector_potential(ds)

    assert isinstance(az, xr.DataArray)
    assert az.dims == ("x", "y")
    assert az.shape == (64, 32)

    # Verify curl: dAz/dy ~= Bx, -dAz/dx ~= By
    x = ds.coords["x"].values
    y = ds.coords["y"].values
    dAz_dy = np.gradient(az.values, y, axis=1)
    dAz_dx = -np.gradient(az.values, x, axis=0)

    # Interior difference should be second-order accurate
    np.testing.assert_allclose(dAz_dy[2:-2, 2:-2], ds["Bx"].values[2:-2, 2:-2], atol=0.05)
    np.testing.assert_allclose(dAz_dx[2:-2, 2:-2], ds["By"].values[2:-2, 2:-2], atol=0.02)


def test_calc_reconnected_flux():
    """Test reconnected flux computation via midplane and vector potential."""
    ds = create_synthetic_harris_sheet()
    flux_midplane = calc_reconnected_flux(ds, method="midplane")
    flux_az = calc_reconnected_flux(ds, method="az")

    assert flux_midplane > 0.0
    assert flux_az > 0.0
    # Both should be of the same order
    assert np.isclose(flux_midplane, flux_az, rtol=0.3)


def test_dataset_reconnection_accessor():
    """Test dataset.reconnection accessor integration."""
    ds = create_synthetic_harris_sheet()

    az = ds.reconnection.calc_vector_potential()
    assert isinstance(az, xr.DataArray)

    flux = ds.reconnection.reconnected_flux(method="midplane")
    assert isinstance(flux, float)
    assert flux > 0.0


def test_reconnection_plotting_synthetic():
    """Test 2D slice plotting with continuous streamlines and arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = create_synthetic_harris_sheet()
    fig, ax = plot_2d_slice(ds, var="Bz", stream_field="B", stream_density=0.8)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
