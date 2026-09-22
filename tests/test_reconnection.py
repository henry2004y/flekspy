"""Tests for reconnection analysis, vector potential, flux, ReconnectionSeries, and plotting."""

import pytest
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flekspy.reconnection.analysis import (
    calc_vector_potential,
    calc_reconnected_flux,
    ReconnectionSeries,
)
from flekspy.reconnection.plotting import (
    plot_2d_evolution,
    plot_2d_slice,
    plot_peak_state,
    plot_reconnection_rate,
)


def create_synthetic_harris_sheet(nx=64, ny=32, B0=1.0, L=1.0, b1=0.1):
    """Create a synthetic 2D Harris sheet with X-point perturbation."""
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

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


def write_synthetic_idl_ascii(path, time=0.0, iter_num=0, nx=16, ny=8, b1=0.1):
    """Helper to write a valid small synthetic 2D IDL ASCII file."""
    x = np.linspace(-4.0, 4.0, nx)
    y = np.linspace(-2.0, 2.0, ny)
    lines = []
    lines.append("PIC unit")
    lines.append(f"{iter_num} {time} 2 0 6")
    lines.append(f"{nx} {ny}")
    lines.append("x y Bx By Bz ux")

    # Fortran ordering: x varies fastest (column-major)
    for j in range(ny):
        for i in range(nx):
            xi = x[i]
            yj = y[j]
            bx_val = np.tanh(yj)
            by_val = b1 * np.sin(np.pi * xi / 4.0) * np.cos(np.pi * yj / 2.0)
            bz_val = 0.2 * np.sin(np.pi * xi / 4.0) * np.sin(np.pi * yj / 2.0)
            ux_val = 0.4 * np.sin(np.pi * xi / 4.0)
            lines.append(f"{xi:.5e} {yj:.5e} {bx_val:.5e} {by_val:.5e} {bz_val:.5e} {ux_val:.5e}")

    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def synthetic_series_dir(tmp_path):
    """Fixture to generate a sequence of small synthetic IDL output files."""
    data_dir = tmp_path / "reconn_data"
    data_dir.mkdir()
    times = [0.0, 2.0, 4.0, 6.0]
    for i, t in enumerate(times):
        fpath = data_dir / f"z=0_fluid_t{t:05.2f}_n{i*100:04d}.out"
        # Perturbation b1 grows with time to simulate reconnection flux growth
        b1_t = 0.05 + 0.05 * i
        write_synthetic_idl_ascii(fpath, time=t, iter_num=i * 100, nx=16, ny=8, b1=b1_t)
    return data_dir


def test_calc_vector_potential():
    """Test 2D vector potential calculation on synthetic field."""
    ds = create_synthetic_harris_sheet()
    az = calc_vector_potential(ds)

    assert isinstance(az, xr.DataArray)
    assert az.dims == ("x", "y")
    assert az.shape == (64, 32)

    x = ds.coords["x"].values
    y = ds.coords["y"].values
    dAz_dy = np.gradient(az.values, y, axis=1)
    dAz_dx = -np.gradient(az.values, x, axis=0)

    np.testing.assert_allclose(dAz_dy[2:-2, 2:-2], ds["Bx"].values[2:-2, 2:-2], atol=0.05)
    np.testing.assert_allclose(dAz_dx[2:-2, 2:-2], ds["By"].values[2:-2, 2:-2], atol=0.02)


def test_calc_reconnected_flux():
    """Test reconnected flux computation via midplane and vector potential."""
    ds = create_synthetic_harris_sheet()
    flux_midplane = calc_reconnected_flux(ds, method="midplane")
    flux_az = calc_reconnected_flux(ds, method="az")

    assert flux_midplane > 0.0
    assert flux_az > 0.0
    assert np.isclose(flux_midplane, flux_az, rtol=0.3)


def test_dataset_reconnection_accessor():
    """Test dataset.reconnection accessor integration."""
    ds = create_synthetic_harris_sheet()

    az = ds.reconnection.calc_vector_potential()
    assert isinstance(az, xr.DataArray)

    flux = ds.reconnection.reconnected_flux(method="midplane")
    assert isinstance(flux, float)
    assert flux > 0.0


def test_reconnection_series_and_methods(synthetic_series_dir):
    """Test ReconnectionSeries container, metrics caching, rate, and summary."""
    pattern = str(synthetic_series_dir / "*.out")
    series = ReconnectionSeries(pattern, max_cache=4)

    # 1. Basic container properties
    assert len(series) == 4
    np.testing.assert_allclose(series.times, [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(series.iters, [0, 100, 200, 300])

    # 2. Reconnected flux
    flux = series.flux
    assert len(flux) == 4
    # Flux should increase over time as b1 increases
    assert flux[-1] > flux[0]

    # 3. Reconnection rate: verify default smooth_window=1 produces raw gradient
    rate_raw = series.rate()  # default smooth_window=1
    expected_rate = np.gradient(flux, series.times)
    np.testing.assert_allclose(rate_raw, expected_rate)

    # Test smoothed rate with window=3
    rate_smoothed = series.rate(smooth_window=3)
    assert len(rate_smoothed) == 4

    # 4. Peak metrics
    assert len(series.peak_hall_bz) == 4
    assert np.all(series.peak_hall_bz > 0.0)
    assert len(series.peak_outflow_ux) == 4
    assert np.all(series.peak_outflow_ux > 0.0)

    # 5. Peak frame index & peak time
    p_idx = series.peak_frame_index
    assert 0 <= p_idx < len(series)
    assert series.peak_time == series.times[p_idx]

    # 6. summary() dictionary
    summary = series.summary()
    assert isinstance(summary, dict)
    for key in ["peak_rate", "peak_time", "peak_frame", "max_flux", "peak_hall_bz", "peak_outflow_ux", "total_frames", "time_range"]:
        assert key in summary
    assert summary["total_frames"] == 4
    assert summary["time_range"] == (0.0, 6.0)

    # 7. Slicing maintains ReconnectionSeries type
    sub = series[1:3]
    assert isinstance(sub, ReconnectionSeries)
    assert len(sub) == 2


def test_plot_reconnection_rate(synthetic_series_dir, tmp_path):
    """Test plot_reconnection_rate with default smooth_window=1 and custom ax."""
    series = ReconnectionSeries(str(synthetic_series_dir / "*.out"))

    # Test standalone figure creation with default smooth_window=1
    save_file = str(tmp_path / "rate.png")
    fig, (ax1, ax2) = plot_reconnection_rate(series, save_path=save_file)
    assert fig is not None
    assert ax1 is not None
    assert ax2 is not None
    assert (tmp_path / "rate.png").is_file()
    plt.close(fig)

    # Test with provided ax
    fig, ax = plt.subplots()
    fig_ret, (ax1, ax2) = plot_reconnection_rate(series, ax=ax, smooth_window=1)
    assert fig_ret is fig
    assert ax1 is ax
    plt.close(fig)


def test_plot_peak_state(synthetic_series_dir, tmp_path):
    """Test plot_peak_state with auto-detected peak frame, explicit frame, and save."""
    series = ReconnectionSeries(str(synthetic_series_dir / "*.out"))

    save_file = str(tmp_path / "peak_state.png")
    fig, axes = plot_peak_state(series, save_path=save_file)
    assert fig is not None
    assert len(axes) == 6
    assert (tmp_path / "peak_state.png").is_file()
    plt.close(fig)

    # Test explicit time selection
    fig2, axes2 = plot_peak_state(series, time=2.0)
    assert len(axes2) == 6
    plt.close(fig2)


def test_plot_2d_evolution(synthetic_series_dir, tmp_path):
    """Test plot_2d_evolution across multiple snapshots."""
    series = ReconnectionSeries(str(synthetic_series_dir / "*.out"))

    save_file = str(tmp_path / "evolution.png")
    fig, axes = plot_2d_evolution(
        series,
        var="Bz",
        times=[0.0, 4.0],
        save_path=save_file,
    )
    assert fig is not None
    assert len(axes) == 2
    assert (tmp_path / "evolution.png").is_file()
    plt.close(fig)

    # Test default evenly-distributed n_frames
    fig2, axes2 = plot_2d_evolution(series, var="Bx", n_frames=3)
    assert len(axes2) == 3
    plt.close(fig2)


def test_plot_2d_slice_synthetic():
    """Test 2D slice plotting with continuous streamlines and arrows."""
    ds = create_synthetic_harris_sheet()
    fig, ax = plot_2d_slice(ds, var="Bz", stream_field="B", stream_density=0.8)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
