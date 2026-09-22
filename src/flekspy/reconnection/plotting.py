"""
Publication-quality plotting functions for magnetic reconnection.
Includes continuous magnetic field lines with directional arrows using flekspy.plot.streamplot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from flekspy.plot.streamplot import streamplot
from flekspy.reconnection.analysis import ReconnectionSeries, _find_var, calc_vector_potential


def plot_reconnection_rate(
    series: ReconnectionSeries,
    smooth_window: int = 3,
    ax: Optional[plt.Axes] = None,
    color_flux: str = "#1f77b4",
    color_rate: str = "#d62728",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot reconnected magnetic flux and reconnection rate vs time."""
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    else:
        fig = ax.get_figure()
        ax1 = ax

    times = series.times
    flux = series.flux
    rate = series.rate(smooth_window=smooth_window)

    line1 = ax1.plot(times, flux, "o-", color=color_flux, lw=2.2, markersize=3.5, label=r"$\Delta\Psi(t)$")
    ax1.set_xlabel(r"Time $t \, [\Omega_{ci}^{-1}]$", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Reconnected Flux $\Delta\Psi / (B_0 d_i)$", color=color_flux, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_flux)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    line2 = ax2.plot(times, rate, "s-", color=color_rate, lw=2.2, markersize=3.5, label=r"Rate $R(t)$")
    ax2.set_ylabel(r"Reconnection Rate $R(t) = d\Delta\Psi/dt$", color=color_rate, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_rate)

    p_idx = int(np.argmax(rate))
    if len(times) > 0 and p_idx < len(times):
        ax2.axvline(times[p_idx], color=color_rate, linestyle=":", alpha=0.6)
        ax2.plot(times[p_idx], rate[p_idx], "*", color=color_rate, markersize=10)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_2d_slice(
    ds: xr.Dataset,
    var: str = "Bz",
    stream_field: Optional[str] = "B",
    ax: Optional[plt.Axes] = None,
    stream_density: float = 1.0,
    stream_color: str = "black",
    stream_alpha: float = 0.8,
    stream_arrowsize: float = 1.1,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D color contour of scalar variable with continuous field lines with arrows."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    else:
        fig = ax.get_figure()

    target_var = _find_var(ds, [var])
    if target_var is None:
        raise KeyError(f"Variable '{var}' not found in dataset. Available: {list(ds.data_vars)}")

    x = ds.coords["x"].values
    y = ds.coords["y"].values
    val = ds[target_var].values

    if ds[target_var].dims == ("y", "x"):
        val = val.T

    if cmap is None:
        if any(b in var.lower() for b in ["bz", "ey", "ez", "ux", "uy", "jx", "jy", "jz"]):
            cmap = "RdBu_r"
        else:
            cmap = "viridis"

    if vmin is None or vmax is None:
        if cmap == "RdBu_r":
            bound = float(np.max(np.abs(val)))
            if vmin is None:
                vmin = -bound
            if vmax is None:
                vmax = bound
        else:
            if vmin is None:
                vmin = float(np.min(val))
            if vmax is None:
                vmax = float(np.max(val))

    X, Y = np.meshgrid(x, y, indexing="ij")
    cf = ax.pcolormesh(X, Y, val, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    if stream_field == "B":
        bx_name = _find_var(ds, ["Bx", "bx"])
        by_name = _find_var(ds, ["By", "by"])
        if bx_name and by_name:
            bx = ds[bx_name].values
            by = ds[by_name].values
            if ds[bx_name].dims == ("x", "y"):
                bx_grid = bx.T
                by_grid = by.T
            else:
                bx_grid = bx
                by_grid = by

            streamplot(
                ax,
                x,
                y,
                bx_grid,
                by_grid,
                density=stream_density,
                color=stream_color,
                linewidth=0.85,
                arrowsize=stream_arrowsize,
            )

    elif stream_field == "u":
        ux_name = _find_var(ds, ["ux", "Ux", "ux_i", "ux_S0"])
        uy_name = _find_var(ds, ["uy", "Uy", "uy_i", "uy_S0"])
        if ux_name and uy_name:
            ux = ds[ux_name].values
            uy = ds[uy_name].values
            if ds[ux_name].dims == ("x", "y"):
                ux_grid = ux.T
                uy_grid = uy.T
            else:
                ux_grid = ux
                uy_grid = uy

            streamplot(
                ax,
                x,
                y,
                ux_grid,
                uy_grid,
                density=stream_density,
                color=stream_color,
                linewidth=0.85,
                arrowsize=stream_arrowsize,
            )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / d_i$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$y / d_i$", fontsize=11, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    if colorbar:
        cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.9)
        cb.set_label(f"{target_var}", fontsize=10, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_peak_state(
    series: ReconnectionSeries,
    frame_idx: Optional[int] = None,
    time: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel overview plot of the reconnection structure at peak reconnection rate."""
    if frame_idx is None:
        if time is not None:
            frame_idx, ds = series.get_frame(time=time)
        else:
            frame_idx = series.peak_frame_index
            ds = series[frame_idx]
    else:
        ds = series[frame_idx]

    t_frame = float(series.times[frame_idx])
    fig = plt.figure(figsize=(13, 10), dpi=dpi)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_bz = fig.add_subplot(gs[0, 1])
    ax_ux = fig.add_subplot(gs[1, 0])
    ax_cutx = fig.add_subplot(gs[1, 1])
    ax_cuty = fig.add_subplot(gs[2, 0])
    ax_rate = fig.add_subplot(gs[2, 1])

    rho_var = _find_var(ds, ["rho", "Rho", "rho_i", "rho_S0", "pXXS0"]) or list(ds.data_vars.keys())[0]
    plot_2d_slice(ds, var=rho_var, stream_field="B", ax=ax_rho, title=rf"(a) {rho_var} & $\mathbf{{B}}$ Field Lines", stream_arrowsize=1.0)

    bz_var = _find_var(ds, ["Bz", "bz"]) or "Bz"
    plot_2d_slice(ds, var=bz_var, stream_field="B", ax=ax_bz, title=rf"(b) Hall Field $B_z$ & $\mathbf{{B}}$ Field Lines", stream_arrowsize=1.0)

    ux_var = _find_var(ds, ["ux", "Ux", "ux_i", "ux_S0", "By", "by"])
    plot_2d_slice(ds, var=ux_var, stream_field="B", ax=ax_ux, title=rf"(c) Outflow Velocity {ux_var} & $\mathbf{{B}}$ Field Lines", stream_arrowsize=1.0)

    x = ds.coords["x"].values
    y = ds.coords["y"].values
    iy0 = int(np.argmin(np.abs(y)))
    by_var = _find_var(ds, ["By", "by"])
    if by_var:
        by_mid = ds[by_var].values[:, iy0] if ds[by_var].dims == ("x", "y") else ds[by_var].values[iy0, :]
        ax_cutx.plot(x, by_mid, "b-", lw=2, label=r"$B_y(x, y=0)$")
    if ux_var and ux_var != by_var:
        ux_mid = ds[ux_var].values[:, iy0] if ds[ux_var].dims == ("x", "y") else ds[ux_var].values[iy0, :]
        ax_cutx.plot(x, ux_mid, "r--", lw=2, label=f"${ux_var}(x, y=0)$")
    ax_cutx.set_xlabel(r"$x / d_i$", fontsize=10, fontweight="bold")
    ax_cutx.set_ylabel("Midplane Amplitudes", fontsize=10, fontweight="bold")
    ax_cutx.set_title("(d) Reconnection Midplane Cut", fontsize=10.5, fontweight="bold")
    ax_cutx.grid(True, linestyle="--", alpha=0.5)
    ax_cutx.legend(loc="best")

    ix0 = int(np.argmin(np.abs(x)))
    bx_var = _find_var(ds, ["Bx", "bx"])
    if bx_var:
        bx_cut = ds[bx_var].values[ix0, :] if ds[bx_var].dims == ("x", "y") else ds[bx_var].values[:, ix0]
        ax_cuty.plot(y, bx_cut, "g-", lw=2, label=r"$B_x(x=0, y)$")
    ax_cuty.set_xlabel(r"$y / d_i$", fontsize=10, fontweight="bold")
    ax_cuty.set_ylabel("Inflow Field", fontsize=10, fontweight="bold")
    ax_cuty.set_title("(e) Inflow Profile Across Current Sheet", fontsize=10.5, fontweight="bold")
    ax_cuty.grid(True, linestyle="--", alpha=0.5)
    ax_cuty.legend(loc="best")

    plot_reconnection_rate(series, ax=ax_rate)
    ax_rate.axvline(t_frame, color="black", linestyle="--", lw=1.5, label=f"t = {t_frame:.1f}")

    fig.suptitle(rf"Magnetic Reconnection State at $t = {t_frame:.1f} \, \Omega_{{ci}}^{{-1}}$ (Frame {frame_idx})", fontsize=13, fontweight="bold", y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, np.array([ax_rho, ax_bz, ax_ux, ax_cutx, ax_cuty, ax_rate])


def plot_2d_evolution(
    series: ReconnectionSeries,
    var: str = "Bz",
    times: Optional[Sequence[float]] = None,
    n_frames: int = 4,
    stream_field: str = "B",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot multi-panel time evolution showing current sheet thinning and reconnection development."""
    if times is None:
        idx_list = np.linspace(0, len(series) - 1, n_frames, dtype=int)
    else:
        idx_list = [series.get_frame(time=t)[0] for t in times]

    fig, axes = plt.subplots(len(idx_list), 1, figsize=(9, 2.7 * len(idx_list)), dpi=dpi, sharex=True)
    if len(idx_list) == 1:
        axes = np.array([axes])

    for i, idx in enumerate(idx_list):
        ds = series[idx]
        t = series.times[idx]
        plot_2d_slice(
            ds,
            var=var,
            stream_field=stream_field,
            ax=axes[i],
            title=rf"$t = {t:.1f} \, \Omega_{{ci}}^{{-1}}$",
            colorbar=True,
            stream_arrowsize=1.0,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes
