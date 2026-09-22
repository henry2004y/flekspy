"""
Xarray Dataset accessor for magnetic reconnection.
Enables ds.reconnection.calc_vector_potential(), ds.reconnection.reconnected_flux(),
and ds.reconnection.plot_slice().
"""

from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt
import xarray as xr

from flekspy.reconnection.analysis import calc_reconnected_flux, calc_vector_potential
from flekspy.reconnection.plotting import plot_2d_slice


@xr.register_dataset_accessor("reconnection")
class ReconnectionAccessor:
    """Xarray Dataset accessor for magnetic reconnection analysis and plotting."""

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def calc_vector_potential(self, **kwargs) -> xr.DataArray:
        """Calculate out-of-plane vector potential Az(x, y)."""
        return calc_vector_potential(self._obj, **kwargs)

    def reconnected_flux(self, method: str = "midplane", **kwargs) -> float:
        """Calculate reconnected magnetic flux Delta Psi."""
        return calc_reconnected_flux(self._obj, method=method, **kwargs)

    def plot_slice(
        self,
        var: str = "Bz",
        stream_field: Optional[str] = "B",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ):
        """Plot 2D color contour with continuous magnetic field lines with arrows."""
        return plot_2d_slice(
            self._obj,
            var=var,
            stream_field=stream_field,
            ax=ax,
            **kwargs,
        )
