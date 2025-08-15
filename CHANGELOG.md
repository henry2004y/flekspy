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
