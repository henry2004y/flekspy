"""Exosphere models."""
import typing
import numpy as np
from scipy.constants import G, k as k_B, m_p


class Exosphere:
    """A class to represent an exosphere model."""

    def __init__(
        self,
        neutral_profile: typing.Literal["exponential", "power_law", "chamberlain"] = "exponential",
        n0: float = 1.0e10,
        H0: float = 100.0e3,
        T0: float = 1000.0,
        k0: float = 2.0,
        exobase_radius: float = 6371.0e3,
        M_planet: float = 5.972e24,  # Mass of Earth in kg
        m_neutral: float = 1.008 * m_p,  # Mass of Hydrogen in kg
    ):
        """
        Initialize the Exosphere model.

        Parameters
        ----------
        neutral_profile : str, optional
            Name of the density profile.
            Supported values: "exponential", "power_law", "chamberlain".
            The default is "exponential".
        n0 : float, optional
            Number densities at reference radius [m⁻³]. The default is 1.0e10.
        H0 : float, optional
            Scale heights [m]. The default is 100.0e3.
        T0 : float, optional
            Temperatures [K]. The default is 1000.0.
        k0 : float, optional
            Power-law exponents. The default is 2.0.
        exobase_radius : float, optional
            Radius of the exobase [m]. Neutral density is zero below this
            radius. The default is 6371.0e3.
        M_planet : float, optional
            Mass of the planet [kg]. The default is Earth's mass.
        m_neutral : float, optional
            Mass of the neutral particle [kg]. The default is Hydrogen's mass.
        """
        self.neutral_profile = neutral_profile
        self.n0 = n0
        self.H0 = H0
        self.T0 = T0
        self.k0 = k0
        self.exobase_radius = exobase_radius
        self.M_planet = M_planet
        self.m_neutral = m_neutral

    def get_neutral_density(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate the neutral density as a function of radius.

        Parameters
        ----------
        r : np.ndarray
            Radius from the center of the body [m].

        Returns
        -------
        np.ndarray
            Neutral density [m⁻³].
        """
        if self.neutral_profile == "exponential":
            density = self.n0 * np.exp(-(r - self.exobase_radius) / self.H0)
        elif self.neutral_profile == "power_law":
            density = self.n0 * (self.exobase_radius / r) ** self.k0
        elif self.neutral_profile == "chamberlain":
            lambda_c = (
                G
                * self.M_planet
                * self.m_neutral
                / (k_B * self.T0 * self.exobase_radius)
            )
            density = self.n0 * np.exp(lambda_c * (self.exobase_radius / r - 1))
        else:
            raise ValueError(f"Unknown neutral profile: {self.neutral_profile}")

        density[r < self.exobase_radius] = 0
        return density

    def plot_neutral_profile(
        self,
        max_altitude: float = 10000e3,
        num_points: int = 1000,
        ax=None,
        **kwargs,
    ):
        """
        Plot the neutral density profile as a function of altitude.

        Parameters
        ----------
        max_altitude : float, optional
            Maximum altitude for the plot [m]. The default is 10000e3.
        num_points : int, optional
            Number of points to use in the plot. The default is 1000.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object to plot on. If None, a new figure and axes
            will be created. The default is None.
        **kwargs
            Additional keyword arguments to be passed to `ax.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object for the plot.
        ax : matplotlib.axes.Axes
            The Axes object for the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        altitudes = np.linspace(0, max_altitude, num_points)
        radii = self.exobase_radius + altitudes
        densities = self.get_neutral_density(radii)

        ax.plot(altitudes / 1e3, densities, **kwargs)
        ax.set_xlabel("Altitude [km]")
        ax.set_ylabel("Neutral Density [m⁻³]")
        ax.set_title(f"Neutral Density Profile: {self.neutral_profile}")
        ax.grid(True)
        ax.set_yscale("log")

        return fig, ax
