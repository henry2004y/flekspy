import numpy as np
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Optional, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors
import pyvista as pv
from pyvista import BasePlotter

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
        log_scale: bool = True,
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
            log_scale (bool, optional): If True, the colorbar is plotted in log scale.
                                        Defaults to True.
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
                data_to_consider = H[H > 0] if log_scale else H
                if data_to_consider.size > 0:
                    vmin = min(vmin, data_to_consider.min())
                    vmax = max(vmax, data_to_consider.max())

        if vmin == float("inf"):  # All histograms were empty or all zeros
            vmin, vmax = (1, 10) if log_scale else (0, 1)  # Dummy range for empty plots

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
        }
        imshow_settings.update(imshow_kwargs)

        # Handle log scale normalization
        if log_scale:
            if "norm" not in imshow_settings:
                imshow_settings["norm"] = colors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(imshow_settings["cmap"])
            cmap.set_bad(color="white")
            imshow_settings["cmap"] = cmap
        elif "norm" not in imshow_settings:
            imshow_settings["norm"] = plt.Normalize(vmin=vmin, vmax=vmax)

        im = None
        for i in range(num_plots):
            ax = axes_flat[i]
            H = histograms[i]
            xedges = xedges_list[i]
            yedges = yedges_list[i]

            if H is not None and H.size > 0 and np.any(H):
                # For log scale, mask zero values
                plot_H = np.ma.masked_where(H <= 0, H) if log_scale else H
                im = ax.imshow(
                    plot_H.T,
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
        log_scale: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **plotter_kwargs: Any,
    ) -> Optional[BasePlotter]:
        """
        Plots the 3D phase space distribution using PyVista volume rendering.
        This function creates a 3D histogram and visualizes it as a volume.
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
            log_scale (bool, optional): If True, the colorbar is plotted in log scale.
                                        Defaults to True.
            vmin (float, optional): The minimum value for the colorbar. Defaults to None.
            vmax (float, optional): The maximum value for the colorbar. Defaults to None.
            title (str, optional): The title for the plot. Defaults to "3D Phase Space Distribution".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            **plotter_kwargs: Additional keyword arguments to be passed to `pyvista.Plotter.add_volume()`.
        Returns:
            pyvista.BasePlotter: A PyVista plotter object.
        Note:
            It is recommended to close the plotter object using `plotter.close()`
            after you are done with it to prevent potential issues with a unclean
            shutdown of the VTK backend.
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

        if np.sum(H) == 0:
            logger.warning("Histogram is empty, nothing to plot.")
            return None

        # --- 2. Create PyVista grid ---
        grid = pv.ImageData()
        grid.dimensions = np.array(H.shape) + 1
        grid.origin = (edges[0][0], edges[1][0], edges[2][0])
        grid.spacing = (
            (edges[0][-1] - edges[0][0]) / H.shape[0],
            (edges[1][-1] - edges[1][0]) / H.shape[1],
            (edges[2][-1] - edges[2][0]) / H.shape[2],
        )
        grid.cell_data["density"] = H.flatten(order="F")

        # --- 3. Create PyVista plot ---
        plotter = pv.Plotter()

        if "clim" not in plotter_kwargs:
            plotter_kwargs["clim"] = [vmin, vmax]

        plotter.add_volume(
            grid,
            scalars="density",
            log_scale=log_scale,
            scalar_bar_args={"title": cbar_label},
            **plotter_kwargs,
        )

        final_title = title if title is not None else "3D Phase Space Distribution"
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        plotter.add_axes(
            xlabel=final_xlabel, ylabel=final_ylabel, zlabel=final_zlabel
        )
        plotter.camera_position = "iso"
        plotter.set_background("white")
        plotter.add_text(final_title, position="upper_edge", font_size=10)

        return plotter

    def pairplot(
        self,
        variables: List[str] = ["velocity_x", "velocity_y", "velocity_z"],
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        bins: int = 50,
        log_scale: bool = True,
        figsize=(10, 10),
        title: str = "Velocity Space Pairplot",
        **imshow_kwargs: Any,
    ) -> Optional[Tuple[Figure, np.ndarray]]:
        """
        Plots a pairplot of the velocity space distributions (vx, vy, vz).

        This function creates a 3x3 grid of subplots. The diagonal plots
        show the 1D histogram for each velocity component. The off-diagonal
        plots show the 2D histogram for each pair of velocity components.

        Args:
            variables (list, optional): A list of velocity components to plot.
                                         Defaults to ["velocity_x", "velocity_y", "velocity_z"].
            x_range (tuple, optional): A tuple (min, max) for filtering particles
                                       by x-position.
            y_range (tuple, optional): A tuple (min, max) for filtering particles
                                       by y-position.
            z_range (tuple, optional): A tuple (min, max) for filtering particles
                                       by z-position.
            bins (int, optional): The number of bins for histograms. Defaults to 50.
            log_scale (bool, optional): If True, the colorbar is plotted in log scale.
                                        Defaults to True.
            figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
            title (str, optional): The title for the plot. Defaults to "Velocity Space Pairplot".
            **imshow_kwargs: Additional keyword arguments for `ax.imshow()`.

        Returns:
            tuple: A tuple containing the matplotlib figure and the array of axes
                   objects (`fig`, `axes`).
        """
        # --- 1. Select data ---
        nvar = len(variables)
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
        for comp in variables:
            if comp not in component_map:
                raise ValueError(f"Component '{comp}' not found in data.")

        vel_indices = [component_map[comp] for comp in variables]
        vel_data = rdata[:, vel_indices]

        # Default imshow settings that can be overridden by user
        imshow_settings = {
            "cmap": "turbo",
            "interpolation": "nearest",
            "origin": "lower",
            "aspect": "auto",
        }
        imshow_settings.update(imshow_kwargs)

        # Get universal ranges
        ranges = [(vel_data[:, k].min(), vel_data[:, k].max()) for k in range(nvar)]

        # --- 3. Create subplot grid ---
        fig, axes = plt.subplots(nvar, nvar, figsize=figsize, constrained_layout=True)

        # --- 4. Plot histograms ---
        # Find the min and max for the color scale across all 2D histograms
        h_min, h_max = float("inf"), float("-inf")
        histograms = {}
        for i in range(nvar):
            for j in range(nvar):
                if i != j:
                    H, _, _ = np.histogram2d(
                        vel_data[:, j],
                        vel_data[:, i],
                        bins=bins,
                        range=[ranges[j], ranges[i]],
                    )
                    histograms[(i, j)] = H
                    data_to_consider = H[H > 0] if log_scale else H
                    if data_to_consider.size > 0:
                        h_min = min(h_min, data_to_consider.min())
                        h_max = max(h_max, data_to_consider.max())

        if h_min == float("inf"):
            h_min, h_max = (1, 10) if log_scale else (0, 1)
        if h_min >= h_max:
            h_max = h_min + 1

        if log_scale:
            imshow_settings["norm"] = colors.LogNorm(vmin=h_min, vmax=h_max)
            cmap = plt.get_cmap(imshow_settings["cmap"])
            cmap.set_bad(color="white")
            imshow_settings["cmap"] = cmap
        else:
            imshow_settings["norm"] = plt.Normalize(vmin=h_min, vmax=h_max)

        for i in range(nvar):
            for j in range(nvar):
                ax = axes[i, j]
                if i == j:  # Diagonal: 1D histogram
                    ax.hist(vel_data[:, i], bins=bins, color="gray", range=ranges[i])
                    ax.set_yticklabels([])
                else:  # Off-diagonal: 2D histogram
                    H = histograms.get((i, j), np.array([[]]))
                    x_edges = np.linspace(ranges[j][0], ranges[j][1], bins + 1)
                    y_edges = np.linspace(ranges[i][0], ranges[i][1], bins + 1)
                    plot_H = np.ma.masked_where(H <= 0, H) if log_scale else H

                    im = ax.imshow(
                        plot_H.T,
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        **imshow_settings,
                    )

                # --- 5. Set labels ---
                if i == nvar - 1:
                    ax.set_xlabel(variables[j])
                if j == 0:
                    ax.set_ylabel(variables[i])

        fig.suptitle(title, fontsize="x-large")

        return fig, axes

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
        log_scale: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        cmap: str = "turbo",
        **plotter_kwargs: Any,
    ) -> Optional[BasePlotter]:
        """
        Plots the 3D phase space distribution using three intersecting planes with PyVista.
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
            log_scale (bool, optional): If True, the colorbar is plotted in log scale.
                                        Defaults to True.
            title (str, optional): The title for the plot. Defaults to "Intersecting Planes of Phase Space".
            xlabel (str, optional): The label for the x-axis. Defaults to `x_variable`.
            ylabel (str, optional): The label for the y-axis. Defaults to `y_variable`.
            zlabel (str, optional): The label for the z-axis. Defaults to `z_variable`.
            cmap (str, optional): The colormap to use for the planes. Defaults to "turbo".
            **plotter_kwargs: Additional keyword arguments to be passed to `pyvista.Plotter.add_mesh()`.
        Returns:
            pyvista.BasePlotter: A PyVista plotter object.
        Note:
            It is recommended to close the plotter object using `plotter.close()`
            after you are done with it to prevent potential issues with a unclean
            shutdown of the VTK backend.
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

        if np.sum(H) == 0:
            logger.warning("Histogram is empty, nothing to plot.")
            return None

        # --- 2. Create PyVista grid ---
        grid = pv.ImageData()
        grid.dimensions = np.array(H.shape) + 1
        grid.origin = (edges[0][0], edges[1][0], edges[2][0])
        grid.spacing = (
            (edges[0][-1] - edges[0][0]) / H.shape[0],
            (edges[1][-1] - edges[1][0]) / H.shape[1],
            (edges[2][-1] - edges[2][0]) / H.shape[2],
        )
        grid.cell_data["density"] = H.flatten(order="F")

        # --- 3. Create PyVista plot ---
        plotter = pv.Plotter()

        plotter_settings = {"cmap": cmap}
        plotter_settings.update(plotter_kwargs)

        plotter.add_mesh(
            grid.slice_orthogonal(),
            scalars="density",
            log_scale=log_scale,
            scalar_bar_args={"title": cbar_label},
            **plotter_settings,
        )

        final_title = (
            title if title is not None else "Intersecting Planes of Phase Space"
        )
        final_xlabel = xlabel if xlabel is not None else x_variable
        final_ylabel = ylabel if ylabel is not None else y_variable
        final_zlabel = zlabel if zlabel is not None else z_variable

        plotter.add_axes(
            xlabel=final_xlabel, ylabel=final_ylabel, zlabel=final_zlabel
        )
        plotter.camera_position = "iso"
        plotter.set_background("white")
        plotter.add_text(final_title, position="upper_edge", font_size=10)

        return plotter
