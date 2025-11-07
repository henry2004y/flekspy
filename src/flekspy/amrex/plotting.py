import numpy as np
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Optional, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors

logger = logging.getLogger(__name__)


class AMReXPlottingMixin:
    """A mixin class for AMReXParticleData plotting functionalities."""

    def plot_phase(
        self,
        x_variable: str,
        y_variable: str,
        bins: Union[int, Tuple[int, int]] = 100,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        log_scale: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        plot_zero_lines: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax: Optional[Axes] = None,
        add_colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 2D phase space distribution for any two selected variables.

        This function creates a 2D weighted histogram to visualize the particle
        density. If a 'weight' component is present in the data, it will be
        used for the histogram weighting. Otherwise, a standard (unweighted)
        histogram is generated.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            bins (int or tuple, optional): The number of bins. This can be a
                                           single integer for the same number of
                                           bins in each dimension, or a two-element
                                           tuple for different numbers of bins in the
                                           x and y dimension, respectively.
                                           Defaults to 100.
            hist_range (list of lists, optional): The leftmost and rightmost edges of the
                                             bins along each dimension. It should be
                                             in the format [[xmin, xmax], [ymin, ymax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for the x-axis boundary.
            y_range (tuple, optional): A tuple (min, max) for the y-axis boundary.
            z_range (tuple, optional): A tuple (min, max) for the z-axis boundary.
                                       For 2D data, this is ignored.
            normalize (bool, optional): If True, the histogram is normalized to
                                        form a probability density. Defaults to False.
            log_scale (bool, optional): If True, the colorbar is plotted in log scale.
                                        Defaults to True.
            plot_zero_lines (bool, optional): If True, plot dashed lines at x=0 and y=0.
                                              Defaults to True.
            title (str, optional): The title for the plot. Defaults to "Phase Space Distribution".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
                                                 If None, a new figure and axes are created.
                                                 Defaults to None.
            add_colorbar (bool, optional): If True, a colorbar is added to the plot.
                                           Defaults to True.
            **imshow_kwargs: Additional keyword arguments to be passed to `ax.imshow()`.
                             This can be used to control colormaps (`cmap`), normalization (`norm`), etc.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
                   This allows for a further customization of the plot after its creation.
        """
        # --- 1. Select data ---
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            logger.warning("No particles to plot.")
            return None

        # --- 2. Map component names to column indices ---
        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        # --- 3. Validate input variable names ---
        if x_variable not in component_map or y_variable not in component_map:
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]

        # --- 4. Extract the relevant data columns ---
        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]

        # --- 5. Create the 2D histogram ---
        weights = None
        if "weight" in component_map:
            weight_index = component_map["weight"]
            weights = rdata[:, weight_index]
            cbar_label = "Weighted Particle Density"
        else:
            cbar_label = "Particle Count"

        H, xedges, yedges = np.histogram2d(
            x_data, y_data, bins=bins, range=hist_range, weights=weights
        )

        if normalize:
            total = H.sum()
            if total > 0:
                H /= total
            if weights is not None:
                cbar_label = "Normalized Weighted Density"
            else:
                cbar_label = "Normalized Density"

        # --- 6. Plot the resulting histogram as a color map ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Default imshow settings that can be overridden by user
        imshow_settings = {
            "cmap": "turbo",
            "interpolation": "nearest",
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
        }
        imshow_settings.update(imshow_kwargs)

        # --- Handle log scale ---
        if log_scale:
            # Mask zero values to handle them separately
            masked_H = np.ma.masked_where(H <= 0, H)

            # Get the colormap and set the color for masked values (zeros) to white
            cmap = plt.get_cmap(imshow_settings["cmap"])
            cmap.set_bad(color="white")
            imshow_settings["cmap"] = cmap

            # Apply logarithmic normalization
            # Set vmin to the smallest non-zero value in the data to avoid issues with log(0)
            if masked_H.count() > 0:  # Check if there is any unmasked data
                min_val = masked_H.min() if vmin is None else vmin
                max_val = masked_H.max() if vmax is None else vmax
                if min_val < max_val:
                    imshow_settings["norm"] = colors.LogNorm(vmin=min_val, vmax=max_val)
            im = ax.imshow(masked_H.T, **imshow_settings)
        else:
            if vmin is not None:
                imshow_settings["vmin"] = vmin
            if vmax is not None:
                imshow_settings["vmax"] = vmax
            im = ax.imshow(H.T, **imshow_settings)

        if plot_zero_lines:
            ax.axhline(0, color="gray", linestyle="--")
            ax.axvline(0, color="gray", linestyle="--")

        # --- 7. Add labels and a color bar for context ---
        final_title = title if title is not None else "Phase Space Distribution"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
        ax.tick_params(
            which="minor",
            top=True,
            bottom=True,
            left=True,
            right=True,
        )

        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(cbar_label)

        # --- 8. Return the plot objects ---
        return fig, ax

    def plot_phase_subplots(
        self,
        x_variable: str,
        y_variable: str,
        x_ranges: List[Tuple[float, float]],
        y_ranges: List[Tuple[float, float]],
        bins: Union[int, Tuple[int, int]] = 100,
        normalize: bool = False,
        suptitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **imshow_kwargs: Any,
    ) -> Optional[Tuple[Figure, np.ndarray]]:
        """
        Plots the 2D phase space distribution for multiple regions as subplots.

        This function creates a grid of subplots, each showing the phase space
        distribution for a specified x and y range. All subplots share a common
        colorbar.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            x_ranges (List[Tuple[float, float]]): A list of tuples, where each
                                                  tuple defines the (min, max)
                                                  for the x-axis of a subplot.
            y_ranges (List[Tuple[float, float]]): A list of tuples, where each
                                                  tuple defines the (min, max)
                                                  for the y-axis of a subplot.
            bins (int or tuple, optional): The number of bins for the histogram.
                                           Defaults to 100.
            normalize (bool, optional): If True, the histogram is normalized.
                                        Defaults to False.
            suptitle (str, optional): The main title for the entire figure.
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            **imshow_kwargs: Additional keyword arguments for `ax.imshow()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and the array of axes
                   objects (`fig`, `axes`).
        """
        if len(x_ranges) != len(y_ranges):
            raise ValueError("x_ranges and y_ranges must have the same length.")

        num_plots = len(x_ranges)
        if num_plots == 0:
            return None

        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        if x_variable not in component_map or y_variable not in component_map:
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]

        weights_present = "weight" in component_map
        weight_index = component_map.get("weight")

        histograms = []
        xedges_list = []
        yedges_list = []
        for i in range(num_plots):
            rdata_region = self.select_particles_in_region(
                x_range=x_ranges[i], y_range=y_ranges[i]
            )

            if rdata_region.size == 0:
                histograms.append(np.array([[]]))
                xedges_list.append(np.array([x_ranges[i][0], x_ranges[i][1]]))
                yedges_list.append(np.array([y_ranges[i][0], y_ranges[i][1]]))
                continue

            x_data = rdata_region[:, x_index]
            y_data = rdata_region[:, y_index]
            weights = rdata_region[:, weight_index] if weights_present else None

            H, xedges, yedges = np.histogram2d(
                x_data,
                y_data,
                bins=bins,
                weights=weights,
                density=normalize,
            )
            histograms.append(H)
            xedges_list.append(xedges)
            yedges_list.append(yedges)

        # Determine the global min and max for the color scale
        vmin, vmax = float("inf"), float("-inf")
        for H in histograms:
            if H is not None and H.size > 0:
                vmin = min(vmin, H.min())
                vmax = max(vmax, H.max())

        if vmin == float("inf"):  # All histograms were empty
            vmin, vmax = 0, 1

        # Create subplots
        ncols = int(np.ceil(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
        )
        axes_flat = axes.flatten()

        imshow_settings = {
            "cmap": "turbo",
            "interpolation": "nearest",
            "origin": "lower",
            "aspect": "auto",
            "norm": plt.Normalize(vmin=vmin, vmax=vmax),
        }
        imshow_settings.update(imshow_kwargs)

        im = None
        for i in range(num_plots):
            ax = axes_flat[i]
            H = histograms[i]
            xedges = xedges_list[i]
            yedges = yedges_list[i]
            if H is not None and H.size > 0 and np.any(H):
                im = ax.imshow(
                    H.T,
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    **imshow_settings,
                )

            ax.set_title(f"x range: {x_ranges[i]}, y range: {y_ranges[i]}")
            ax.minorticks_on()

        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Add a single colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])

        if im is None:
            # Create a dummy ScalarMappable if no images were drawn
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=imshow_settings["cmap"], norm=norm)
            sm.set_array([])
            im = sm

        cbar = fig.colorbar(im, cax=cbar_ax)

        if normalize:
            cbar_label = "Normalized Density"
            if weights_present:
                cbar_label = "Normalized Weighted Density"
        else:
            cbar_label = "Particle Count"
            if weights_present:
                cbar_label = "Weighted Particle Density"
        cbar.set_label(cbar_label)

        if suptitle:
            fig.suptitle(suptitle, fontsize="x-large")

        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        fig.text(0.5, 0.04, final_xlabel, ha="center", va="center", fontsize="x-large")
        fig.text(
            0.06,
            0.5,
            final_ylabel,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize="x-large",
        )

        return fig, axes

    def _prepare_3d_histogram_data(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        bins: Union[int, Tuple[int, int, int]],
        hist_range: Optional[List[List[float]]],
        x_range: Optional[Tuple[float, float]],
        y_range: Optional[Tuple[float, float]],
        z_range: Optional[Tuple[float, float]],
        normalize: bool,
    ) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, ...], str]]:
        """Prepares 3D histogram data for plotting methods."""
        if x_range or y_range or z_range:
            rdata = self.select_particles_in_region(x_range, y_range, z_range)
        else:
            rdata = self.rdata

        if rdata.size == 0:
            logger.warning("No particles to plot.")
            return None

        component_map = {
            name: i for i, name in enumerate(self.header.real_component_names)
        }

        if (
            x_variable not in component_map
            or y_variable not in component_map
            or z_variable not in component_map
        ):
            raise ValueError(
                f"Invalid variable name. Choose from {list(component_map.keys())}"
            )

        x_index = component_map[x_variable]
        y_index = component_map[y_variable]
        z_index = component_map[z_variable]

        x_data = rdata[:, x_index]
        y_data = rdata[:, y_index]
        z_data = rdata[:, z_index]
        sample = np.vstack([x_data, y_data, z_data]).T

        weights = None
        cbar_label = "Particle Count"
        if "weight" in component_map:
            weight_index = component_map["weight"]
            weights = rdata[:, weight_index]
            cbar_label = "Weighted Particle Density"

        H, edges = np.histogramdd(sample, bins=bins, range=hist_range, weights=weights)

        if normalize:
            total = H.sum()
            if total > 0:
                H /= total
            if weights is not None:
                cbar_label = "Normalized Weighted Density"
            else:
                cbar_label = "Normalized Density"

        return H, edges, cbar_label

    def plot_phase_3d(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        bins: Union[int, Tuple[int, int, int]] = 50,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **scatter_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 3D phase space distribution for any three selected variables.

        This function creates a 3D histogram and visualizes it as a scatter plot,
        where the color of each point corresponds to the particle density in that bin.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            z_variable (str): The name of the variable for the z-axis.
            bins (int or tuple, optional): The number of bins for each dimension.
                                           Defaults to 50.
            hist_range (list of lists, optional): The range for the bins in the format
                                             [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for filtering particles by x-position.
            y_range (tuple, optional): A tuple (min, max) for filtering particles by y-position.
            z_range (tuple, optional): A tuple (min, max) for filtering particles by z-position.
            normalize (bool, optional): If True, normalize the histogram to form a
                                        probability density. Defaults to False.
            title (str, optional): The title for the plot. Defaults to "3D Phase Space Distribution".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            **scatter_kwargs: Additional keyword arguments to be passed to `ax.scatter()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
        """
        # --- 1. Prepare histogram data ---
        hist_data = self._prepare_3d_histogram_data(
            x_variable,
            y_variable,
            z_variable,
            bins,
            hist_range,
            x_range,
            y_range,
            z_range,
            normalize,
        )
        if hist_data is None:
            return None
        H, edges, cbar_label = hist_data

        # --- 6. Prepare data for scatter plot ---
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2

        # Create a meshgrid of bin centers
        x_grid, y_grid, z_grid = np.meshgrid(
            x_centers, y_centers, z_centers, indexing="ij"
        )

        # Flatten the arrays for scatter plot
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        density = H.flatten()

        # Filter out empty bins
        non_empty = density > 0
        x_flat = x_flat[non_empty]
        y_flat = y_flat[non_empty]
        z_flat = z_flat[non_empty]
        density = density[non_empty]

        # --- 7. Plot the resulting histogram as a 3D scatter plot ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter_settings = {
            "c": density,
            "cmap": "turbo",
            "s": 20,  # a default size
        }
        scatter_settings.update(scatter_kwargs)

        sc = ax.scatter(x_flat, y_flat, z_flat, **scatter_settings)

        # --- 8. Add labels and a color bar ---
        final_title = title if title is not None else "3D Phase Space Distribution"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.set_zlabel(final_zlabel, fontsize="x-large")

        cbar = fig.colorbar(sc)
        cbar.set_label(cbar_label)

        # --- 9. Return the plot objects ---
        return fig, ax

    @staticmethod
    def _plot_plane(ax, H, edges, fixed_coord, cmap, **surface_kwargs):
        """Helper function to plot a single plane."""
        nx, ny, nz = H.shape
        x_edges, y_edges, z_edges = edges

        min_val, max_val = H.min(), H.max()
        cmap = plt.get_cmap(cmap)

        slice_index = {"x": nx // 2, "y": ny // 2, "z": nz // 2}[fixed_coord]

        # Prepare coordinates and data for the selected plane
        if fixed_coord == "x":
            Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
            Y, Z = np.meshgrid(Y_centers, Z_centers, indexing="ij")
            X = np.full_like(Y, (x_edges[slice_index] + x_edges[slice_index + 1]) / 2)
            plane_data = H[slice_index, :, :]
        elif fixed_coord == "y":
            X_centers = (x_edges[:-1] + x_edges[1:]) / 2
            Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
            X, Z = np.meshgrid(X_centers, Z_centers, indexing="ij")
            Y = np.full_like(X, (y_edges[slice_index] + y_edges[slice_index + 1]) / 2)
            plane_data = H[:, slice_index, :]
        else:  # fixed_coord == 'z'
            X_centers = (x_edges[:-1] + x_edges[1:]) / 2
            Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            X, Y = np.meshgrid(X_centers, Y_centers, indexing="ij")
            Z = np.full_like(X, (z_edges[slice_index] + z_edges[slice_index + 1]) / 2)
            plane_data = H[:, :, slice_index]

        # Normalize data for coloring
        denominator = max_val - min_val
        if denominator == 0:
            normalized_data = np.full(plane_data.shape, 0.5)
        else:
            normalized_data = (plane_data - min_val) / denominator

        facecolors = cmap(normalized_data)

        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            shade=False,
            **surface_kwargs,
        )

    def plot_intersecting_planes(
        self,
        x_variable: str,
        y_variable: str,
        z_variable: str,
        bins: Union[int, Tuple[int, int, int]] = 50,
        hist_range: Optional[List[List[float]]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        cmap: str = "turbo",
        **surface_kwargs: Any,
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plots the 3D phase space distribution using three intersecting planes.

        This function creates a 3D histogram and visualizes the density on three
        orthogonal planes that intersect at the center of the histogrammed data.

        Args:
            x_variable (str): The name of the variable for the x-axis.
            y_variable (str): The name of the variable for the y-axis.
            z_variable (str): The name of the variable for the z-axis.
            bins (int or tuple, optional): The number of bins for each dimension.
                                           Defaults to 50.
            hist_range (list of lists, optional): The range for the bins in the format
                                             [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
                                             Defaults to None.
            x_range (tuple, optional): A tuple (min, max) for filtering particles by x-position.
            y_range (tuple, optional): A tuple (min, max) for filtering particles by y-position.
            z_range (tuple, optional): A tuple (min, max) for filtering particles by z-position.
            normalize (bool, optional): If True, normalize the histogram to form a
                                        probability density. Defaults to False.
            title (str, optional): The title for the plot. Defaults to "Intersecting Planes of Phase Space".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            cmap (str, optional): The colormap to use for the planes. Defaults to "viridis".
            **surface_kwargs: Additional keyword arguments to be passed to `ax.plot_surface()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`).
        """
        # --- 1. Prepare histogram data ---
        hist_data = self._prepare_3d_histogram_data(
            x_variable,
            y_variable,
            z_variable,
            bins,
            hist_range,
            x_range,
            y_range,
            z_range,
            normalize,
        )
        if hist_data is None:
            return None
        H, edges, cbar_label = hist_data

        # --- 6. Plot the intersecting planes ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        self._plot_plane(ax, H, edges, "x", cmap, **surface_kwargs)
        self._plot_plane(ax, H, edges, "y", cmap, **surface_kwargs)
        self._plot_plane(ax, H, edges, "z", cmap, **surface_kwargs)

        # --- 7. Add labels and title ---
        final_title = (
            title if title is not None else "Intersecting Planes of Phase Space"
        )
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        ax.set_title(final_title, fontsize="x-large")
        ax.set_xlabel(final_xlabel, fontsize="x-large")
        ax.set_ylabel(final_ylabel, fontsize="x-large")
        ax.set_zlabel(final_zlabel, fontsize="x-large")

        # --- 8. Add a colorbar ---
        norm = plt.Normalize(vmin=H.min(), vmax=H.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for the mappable
        fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1, label=cbar_label)

        # --- 9. Return the plot objects ---
        return fig, ax
