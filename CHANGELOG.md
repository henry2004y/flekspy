# Changelog

<!--next-version-placeholder-->

## v0.1.0 (2024/11/29)

- First release of `flekspy`, based on the original scripts under the FLEKS repo!

## v0.1.3 (2024/12/13)

- Port all the functionalities from the original repo; add runnable examples.

## v0.2.0 (2024/12/20)

- `plot_phase_region` is deprecated; use `plot_phase` with the `region` keyword instead. Add customized decoration option.
- `plot_particles_region` is deprecated; use `plot_particles` with the `region` keyword instead.
- Add utility method `extract_phase` for extracting phase space coordinates and values.
- `_unit_one` -> `unit_one`.

## v0.2.1 (2024/12/22)

- `dataContainer` and its related classes are renamed to be consistent with Python naming conventions.
- `contour` has been renamed to `plot` for `DataContainer2D`.

## v0.2.8 (2025/02/15)

- Add `get_phase` to obtain the direct phase space distributions, which is useful for customized decoration of plots.

## v0.3.0 (2025/07/15)

- Refactor the test particle module to bind the particle ID and data closely through a new class `ParticleTrajectory`.

## v0.3.1 (2025/07/23)

- Switch from `poetry` to `uv` for package management.

## v0.3.2 (2025/08/01)

- `read_initial_location` -> `read_initial_condition` for test particles.

## v0.4.0 (2025/08/03)

- Use XArray for handling IDL format data.

## v0.5.0 (2025/08/05)

- Use Polars for storing test particle data.

## v0.5.3 (2025/08/13)

- Support test particle gradient B storage.

## v0.5.4 (2025/08/14)

- Add ExB, gradient, and curvature drift calculation for test particles.

## v0.5.5 (2025/08/15)

- Use lazy Polars expression for test particle analysis if possible.

## v0.5.21 (2025/10/08)

- Add experimental native AMReX particle data loader.

## v0.6.0 (2025/10/09)

- Switch the default AMReX particle loader. Add plotting support.
- Separate IDL, AMReX, and test particle documentations.

## v0.6.2 (2025/10/30)

- Add `save_trajectory` (CSV, Parquet) and `save_trajectories` (HDF5) to test particles.

## v0.6.8 (2025/11/07)

- Make the native AMReX particle loader into a module.

## v0.6.9 (2025/11/09)

- Add transform to `plot_phase`.

## v0.6.10 (2025/11/09)

- Support B and E field velocity transformations.

## v0.6.11 (2025/11/11)

- Add KDE and GMM fit for phase space in the amrex module.
- Use lazy loading and optional dependency techniques to improve the package loading time from 10s to 0.03s.
- Separate `get_phase_space_density` from `plot_phase` to make further analysis possible.
- Add `get_gmm_parameters` for extracting center velocity and temperatures from GMM fitting.

## v0.6.12 (2025/11/13)

- Add marginal plot option for `plot_phase` if `marginals == True`.
- Allow specifying "vx", "vy", "vz" in the phase space analysis.

## v0.6.13 (2025/11/13)

- Add `get_pressure_anisotropy` for IDL data.

## v0.6.14 (2025/11/14)

- Add `get_current_density` and `get_current_density_from_definition` for IDL data.
- Breaking: the IDL data attribute "para" has been renamed to "parameters".

## v0.6.15 (2025/11/15)

- XArray DataSet for IDL format now contains cleaner "parameters" and "variables" attributes for access.
- The `IDLAccessor` class has been renamed to `DerivedAccessor` and registered as `ds.derived`.

## v0.6.16 (2025/11/17)

- Separate `get_gmm_temperatures` and `get_gmm_parameters` for covariance unit conversions in GMM fitting.

## v0.7.0 (2025/11/18)

- Deprecate customized `DataContainer` class and replace with xarray accessor.
- Move GMM utilities to a separate module.
- Use PyData template for documentation.

## v0.7.1 (2025/11/19)

- Add an analytical exosphere module.
