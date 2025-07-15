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
