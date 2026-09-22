"""
Magnetic reconnection analysis routines:
- Vector potential Az calculation
- Reconnected flux Delta Psi(t) calculation
- Reconnection rate dDeltaPsi/dt calculation
- ReconnectionSeries container subclassing IDLSeries
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import xarray as xr

from flekspy.idl.series import IDLSeries
from flekspy.util.logger import get_logger

logger = get_logger(name=__name__)


def _find_var(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    """Find the first matching variable name in dataset among candidate strings (case-insensitive)."""
    ds_vars = list(ds.data_vars.keys())
    ds_vars_lower = {v.lower(): v for v in ds_vars}
    for c in candidates:
        if c in ds_vars:
            return c
        if c.lower() in ds_vars_lower:
            return ds_vars_lower[c.lower()]
    return None


def calc_vector_potential(
    ds: xr.Dataset,
    x_coord: str = "x",
    y_coord: str = "y",
    bx_var: Optional[str] = None,
    by_var: Optional[str] = None,
) -> xr.DataArray:
    """Calculate the 2D out-of-plane magnetic vector potential Az(x, y).

    In 2D (z-invariant geometry):
        Bx = dAz/dy  =>  Az(x0, y) = int_{y0}^y Bx(x0, y') dy'
        By = -dAz/dx =>  Az(x, y) = Az(x0, y) - int_{x0}^x By(x', y) dx'

    Parameters
    ----------
    ds : xr.Dataset
        2D dataset containing magnetic field components.
    x_coord : str, default 'x'
        Coordinate name along reconnection outflow/current sheet.
    y_coord : str, default 'y'
        Coordinate name normal to the current sheet.
    bx_var : str, optional
        Variable name for Bx. If None, auto-detected from ('Bx', 'bx').
    by_var : str, optional
        Variable name for By. If None, auto-detected from ('By', 'by').

    Returns
    -------
    xr.DataArray
        Computed 2D magnetic vector potential Az(x, y).
    """
    if bx_var is None:
        bx_var = _find_var(ds, ["Bx", "bx"])
    if by_var is None:
        by_var = _find_var(ds, ["By", "by"])

    if bx_var is None or by_var is None:
        raise KeyError(f"Could not find Bx or By variables in dataset. Found: {list(ds.data_vars)}")

    x = ds.coords[x_coord].values
    y = ds.coords[y_coord].values
    bx = ds[bx_var].values
    by = ds[by_var].values

    # Ensure array dimensions match (x, y)
    if ds[bx_var].dims == (y_coord, x_coord):
        bx = bx.T
        by = by.T

    # Find index closest to x = 0 (central line)
    ix0 = int(np.argmin(np.abs(x)))

    # Step 1: 1D integral along central vertical line at x = x0: Az(x0, y)
    az_y = cumulative_trapezoid(bx[ix0, :], y, initial=0.0)

    # Step 2: 2D integral along x: - int_{x0}^x By(x', y) dx'
    int_by = cumulative_trapezoid(by, x, axis=0, initial=0.0)
    int_by_rel_x0 = int_by - int_by[ix0:ix0 + 1, :]

    az_2d = az_y[np.newaxis, :] - int_by_rel_x0

    # Match dataset dimensions
    if ds[bx_var].dims == (y_coord, x_coord):
        az_2d = az_2d.T
        dims = (y_coord, x_coord)
        coords = [ds.coords[y_coord], ds.coords[x_coord]]
    else:
        dims = (x_coord, y_coord)
        coords = [ds.coords[x_coord], ds.coords[y_coord]]

    return xr.DataArray(
        az_2d,
        coords=coords,
        dims=dims,
        name="Az",
        attrs={"long_name": "Out-of-plane magnetic vector potential", "units": "B0 * di"},
    )


def calc_reconnected_flux(
    ds: xr.Dataset,
    method: str = "midplane",
    x_coord: str = "x",
    y_coord: str = "y",
    by_var: Optional[str] = None,
) -> float:
    """Calculate the reconnected magnetic flux Delta Psi.

    Parameters
    ----------
    ds : xr.Dataset
        2D dataset with magnetic field.
    method : {'midplane', 'az'}, default 'midplane'
        Calculation method:
        - 'midplane': Integrates |By(x, y_mid)| dx from central X-line (x=0) to domain edge (x=Lx/2).
        - 'az': Calculates max(|Az(x, y_mid) - Az(x0, y_mid)|) along the midplane (O-point minus X-point).
    x_coord : str, default 'x'
        Coordinate name along outflow direction.
    y_coord : str, default 'y'
        Coordinate name across current sheet.
    by_var : str, optional
        Variable name for By.

    Returns
    -------
    float
        The reconnected magnetic flux Delta Psi.
    """
    if method == "midplane":
        if by_var is None:
            by_var = _find_var(ds, ["By", "by"])
        if by_var is None:
            raise KeyError(f"Could not find By variable in dataset. Found: {list(ds.data_vars)}")

        x = ds.coords[x_coord].values
        y = ds.coords[y_coord].values
        by = ds[by_var].values

        if ds[by_var].dims == (y_coord, x_coord):
            by = by.T

        # Find midplane y index (near y = 0)
        iy_mid = int(np.argmin(np.abs(y)))
        if iy_mid + 1 < len(y) and np.abs(y[iy_mid]) > 1e-12:
            by_slice = 0.5 * (by[:, iy_mid] + by[:, iy_mid + 1])
        else:
            by_slice = by[:, iy_mid]

        ix0 = int(np.argmin(np.abs(x)))
        flux = float(trapezoid(np.abs(by_slice[ix0:]), x[ix0:]))
        return flux

    elif method == "az":
        az = calc_vector_potential(ds, x_coord=x_coord, y_coord=y_coord)
        x = ds.coords[x_coord].values
        y = ds.coords[y_coord].values
        ix0 = int(np.argmin(np.abs(x)))
        iy0 = int(np.argmin(np.abs(y)))

        az_val = az.values
        if az.dims == (y_coord, x_coord):
            az_val = az_val.T

        az_mid = az_val[:, iy0]
        az_xpoint = az_mid[ix0]
        flux = float(np.max(np.abs(az_mid - az_xpoint)))
        return flux

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'midplane' or 'az'.")


class ReconnectionSeries(IDLSeries):
    """Specialized series container for magnetic reconnection simulation results.

    Subclasses IDLSeries with lazy-loading and provides automated computation of:
    - Reconnected magnetic flux Delta Psi(t)
    - Reconnection rate R(t) = d(Delta Psi)/dt
    - Peak quadrupolar Hall magnetic field Bz(t)
    - Peak reconnection outflow velocity ux(t)
    - Electric field Ez at the X-line
    """

    def __init__(
        self,
        files: Union[str, Sequence[Union[str, Any]]],
        eager: bool = False,
        max_cache: int = 32,
        flux_method: str = "midplane",
        **kwargs,
    ):
        super().__init__(files, eager=eager, max_cache=max_cache, **kwargs)
        self.flux_method = flux_method
        self._flux: Optional[np.ndarray] = None
        self._peak_bz: Optional[np.ndarray] = None
        self._peak_ux: Optional[np.ndarray] = None
        self._xpoint_ez: Optional[np.ndarray] = None

    def _compute_metrics(self) -> None:
        """Compute all time series metrics in a single efficient pass over frames."""
        if (
            self._flux is not None
            and self._peak_bz is not None
            and self._peak_ux is not None
            and self._xpoint_ez is not None
        ):
            return

        flux_list = []
        bz_list = []
        ux_list = []
        ez_list = []

        for i in range(len(self)):
            ds = self[i]
            # flux
            flux_list.append(calc_reconnected_flux(ds, method=self.flux_method))
            # peak bz
            bz_var = _find_var(ds, ["Bz", "bz"])
            bz_list.append(float(np.max(np.abs(ds[bz_var].values))) if bz_var else 0.0)
            # peak ux
            ux_var = _find_var(ds, ["ux", "Ux", "ux_i", "ux_S0", "ux_S1"])
            ux_list.append(float(np.max(np.abs(ds[ux_var].values))) if ux_var else 0.0)
            # xpoint ez
            ez_var = _find_var(ds, ["Ez", "ez"])
            if ez_var:
                x = ds.coords["x"].values
                y = ds.coords["y"].values
                ix = int(np.argmin(np.abs(x)))
                iy = int(np.argmin(np.abs(y)))
                val = float(ds[ez_var].values[ix, iy]) if ds[ez_var].dims == ("x", "y") else float(ds[ez_var].values[iy, ix])
                ez_list.append(val)
            else:
                ez_list.append(0.0)

        self._flux = np.array(flux_list, dtype=float)
        self._peak_bz = np.array(bz_list, dtype=float)
        self._peak_ux = np.array(ux_list, dtype=float)
        self._xpoint_ez = np.array(ez_list, dtype=float)

    @property
    def flux(self) -> np.ndarray:
        """Array of reconnected magnetic flux Delta Psi for each time frame."""
        if self._flux is None:
            self._compute_metrics()
        return self._flux

    def rate(self, smooth_window: int = 1) -> np.ndarray:
        """Calculate reconnection rate R(t) = d(Delta Psi)/dt."""
        if len(self) < 2:
            return np.zeros(len(self))

        raw_rate = np.gradient(self.flux, self.times)
        if smooth_window <= 1 or len(raw_rate) < smooth_window:
            return raw_rate

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(raw_rate, kernel, mode="same")
        smoothed[0] = raw_rate[0]
        smoothed[-1] = raw_rate[-1]
        return smoothed

    @property
    def peak_hall_bz(self) -> np.ndarray:
        """Peak out-of-plane Hall magnetic field max(|Bz|) for each time frame."""
        if self._peak_bz is None:
            self._compute_metrics()
        return self._peak_bz

    @property
    def peak_outflow_ux(self) -> np.ndarray:
        """Peak reconnection outflow velocity max(|ux|) for each time frame."""
        if self._peak_ux is None:
            self._compute_metrics()
        return self._peak_ux

    @property
    def xpoint_ez(self) -> np.ndarray:
        """Electric field Ez at the central reconnection X-point (x=0, y=0) for each time frame."""
        if self._xpoint_ez is None:
            self._compute_metrics()
        return self._xpoint_ez

    @property
    def peak_frame_index(self) -> int:
        """Frame index where reconnection rate reaches its maximum."""
        r = self.rate(smooth_window=1)
        return int(np.argmax(r))

    @property
    def peak_time(self) -> float:
        """Time where reconnection rate reaches its maximum."""
        return float(self.times[self.peak_frame_index])

    def summary(self) -> Dict[str, Any]:
        """Compute key summary metrics of the reconnection process."""
        self._compute_metrics()
        r = self.rate(smooth_window=1)
        p_idx = int(np.argmax(r))
        return {
            "peak_rate": float(r[p_idx]),
            "peak_time": float(self.times[p_idx]),
            "peak_frame": p_idx,
            "max_flux": float(np.max(self.flux)),
            "peak_hall_bz": float(np.max(self.peak_hall_bz)),
            "peak_outflow_ux": float(np.max(self.peak_outflow_ux)),
            "total_frames": len(self),
            "time_range": (float(self.times[0]), float(self.times[-1])),
        }
