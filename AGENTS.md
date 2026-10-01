# Repository guide for agents

## Purpose and architecture

`flekspy` is a Python toolkit for processing outputs from FLEKS (Flexible
Exascale Kinetic Simulator), including BATSRUS-style IDL files. It provides
field visualization, particle phase-space analysis, trajectory diagnostics,
and magnetic reconnection analysis. This is a `src/` layout package built
with Hatchling; Python requirements and dependency extras live in
`pyproject.toml`.

| Location | Responsibility |
| --- | --- |
| `src/flekspy/__init__.py` | Public API, lazy imports, and `load()` format dispatch. |
| `src/flekspy/idl/` | ASCII/binary `.out` and multi-snapshot `.outs` readers, `.derived` xarray accessor, and lazy/cached `IDLSeries`. |
| `src/flekspy/xarray/accessor.py` | `.fleks` dataset accessor: expressions, derived variables, plots, contours, and streamlines. |
| `src/flekspy/yt/yt.py` | yt AMReX field/particle integration, field definitions, slices, domain extraction, and phase plots. |
| `src/flekspy/amrex/` | Native AMReX particle headers and binary readers, region selection, weighted phase-space plots, GMM fitting, and population diagnostics. |
| `src/flekspy/tp/test_particles.py` | `FLEKSTP`: particle IDs, Polars trajectories, energy/pitch-angle/drift diagnostics, exports, and trajectory plots. This is library code despite its filename. |
| `src/flekspy/reconnection/` | Vector potential, reconnected flux, `ReconnectionSeries`, `.reconnection` accessor, and rate/slice/evolution plots. |
| `src/flekspy/util/` | Units, restricted expression evaluation, logging, sample-data downloads, field transformations, GMM helpers, and exosphere models. |
| `src/flekspy/plot/streamplot.py` | Custom streamline implementation. |
| `src/paraview/BATSRUSReader.py` | ParaView reader plugin; see setup instructions in `README.md`. Separate from the packaged `src/flekspy` tree. |
| `tests/` | pytest coverage of loaders, dimensions, unstructured grids, species, expressions, transformations, GMM, exosphere, and reconnection. |
| `docs/` | Sphinx configuration, format/algorithm references, and usage notebooks. |
| `.github/workflows/` | CI tests/coverage/docs, benchmarks, and releases. |

### Loading and return types

Use `import flekspy as fs` and inspect the returned object before choosing an
analysis API. `fs.load()` dispatches by the matched file or directory name:

- `.out` or `.outs`: IDL reader; structured data use `xarray.Dataset`,
  while unstructured data use xugrid wrappers. `npict=1` selects the first
  snapshot in `.outs` (one-based).
- Directory ending in `_amrex` with `particle` in its name: native
  `AMReXParticle`, unless `use_yt_loader=True`.
- Other directories ending in `_amrex`: `YtFLEKSData`.
- Directory named exactly `test_particles`: `FLEKSTP`, with `iDomain` and
  `iSpecies` selectors. Use `fs.FLEKSTP(path, ...)` directly for differently
  named trajectory directories.

`iFile` is zero-based. `load()` uses recursive filesystem matching without
explicit sorting, so use a concrete path for reproducibility. Use `IDLSeries`
for chronological collections: it sorts headers by simulation time and iteration,
loads frames on demand, and maintains a bounded cache.

## Setup and checks

From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/
python -m pytest tests/test_reconnection.py
make -C docs html
```

CI uses `uv sync --all-extras --dev`, `uv run pytest tests/ --cov
--cov-report=xml`, and `uv run make html --directory docs/`.
For analysis without the full development environment:

```bash
python -m pip install -e '.[plot,yt,ml]'
```

Extras are `plot` (Matplotlib/SciPy), `yt`, `ml` (scikit-learn), `hdf5`,
`docs`, and `dev`. Some current modules import optional dependencies eagerly:
IDL loading imports the plotting accessor and therefore Matplotlib, SciPy,
and yt; native AMReX particles import a plotting mixin using scikit-learn.
The combined extras above cover these workflows.

`tests/conftest.py` has a session-wide autouse fixture that downloads and
extracts four sample archives into `tests/data/` if their expected files are
missing. Even synthetic tests may trigger those downloads. Network access or
previously extracted fixtures are needed to run the suite. Do not commit sample
outputs, archives, generated images, caches, or documentation build products.

## Visualization and analysis examples

Run examples from the repository root after installing dependencies. Replace
paths with your simulation outputs; `tests/data/` paths refer to fixtures
downloaded by pytest and are not tracked in Git. Field names and species depend
on the simulation: inspect `data_vars`, coordinates, attributes, and particle
component names rather than assuming a fixed schema.

### IDL field map and xarray reductions

For a structured 2D snapshot containing `Bx`, `By`, and `Bz`:

```python
import flekspy as fs
import matplotlib.pyplot as plt

ds = fs.load("tests/data/z=0_fluid_region0_0_t00001640_n00010142.out")
print(ds.sizes, list(ds.data_vars), ds.attrs)
bmag = (ds["Bx"] ** 2 + ds["By"] ** 2 + ds["Bz"] ** 2) ** 0.5
print("Mean magnetic-field magnitude:", float(bmag.mean()))
fig, ax = plt.subplots()
bmag.plot(ax=ax, x="x", y="y", cmap="viridis")
fig.savefig("magnetic_field.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

The `.fleks` accessor provides multi-panel plots and unit-bearing expressions:

```python
fig, axes = ds.fleks.plot("Bx By Bz", unit="planet")
fig.savefig("field_components.png", dpi=150, bbox_inches="tight")
plt.close(fig)
b = ds.fleks.evaluate_expression("np.sqrt({Bx}**2+{By}**2+{Bz}**2)")
```

The IDL reader registers `.derived` and `.fleks` accessors. For manually
constructed xarray datasets, import `flekspy.idl` and/or
`flekspy.xarray.accessor` explicitly to register them.
For structured 3D IDL data use `ds.derived.get_slice("z", 0.0)` before a 2D
plot. `.derived.get_pressure_anisotropy(species=1)` needs the corresponding
species pressure tensor and magnetic field; `.derived.get_current_density()`
needs a suitable 3D field and metadata.

### Time-series field analysis

Choose snapshots with the same grid and variables for concatenation:

```python
import flekspy as fs

series = fs.IDLSeries("simulation/z=0_fluid_*.out", max_cache=4)
index, frame = series.get_frame(time=100.0)  # nearest simulation time
print(index, frame.attrs["time"])
history = series.to_dataset(vars=["Bx", "By", "Bz"])
mean_bz = history["Bz"].mean(dim=["x", "y"])
mean_bz.plot(x="time")
```

`to_dataset()` materializes the series. For large outputs, iterate over frames
and retain only reduced statistics rather than concatenating full fields.

### AMReX field slice through yt

```python
import flekspy as fs
import matplotlib.pyplot as plt

amr = fs.load("simulation/3d_fluid_region0_0_t00000002_n00000007_amrex")
print(amr.field_list)
slice_ds = amr.get_slice("z", 0.0)  # bare cut location uses code_length
print(list(slice_ds.data_vars))
fig, axes = slice_ds.fleks.plot("Bx By Bz")
fig.savefig("amrex_slice.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### AMReX particle distribution and population fit

```python
import flekspy as fs
import matplotlib.pyplot as plt

particles = fs.load("tests/data/3d_particle_region0_1_t00000002_n00000007_amrex")
print(particles.header.real_component_names)
selected = particles.select_particles_in_region(x_range=(-0.5, 0.5))
print("Selected particles:", selected.shape[0])
fig, ax = particles.plot_phase("vx", "vy", bins=64, x_range=(-0.5, 0.5))
fig.savefig("velocity_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
gmm = particles.fit_gmm(
    n_components=2,
    variables=["velocity_x", "velocity_y"],
    x_range=(-0.5, 0.5),
)
print(gmm.weights_, gmm.means_, gmm.covariances_)
```

Choose spatial ranges inside your domain and ensure enough particles for the
fit. `vx`, `vy`, and `vz` are plotting aliases for `velocity_x`, `velocity_y`,
and `velocity_z`. Use actual header names for fitting. Region selection can
avoid reading unrelated grids; accessing `.rdata` or `.idata` loads the full
particle data. Phase-space plots use particle weights.

### Test particle trajectory and kinetic-energy proxy

```python
import flekspy as fs
import polars as pl

tp = fs.FLEKSTP("tests/data/test_particles", iSpecies=0)
pid = tp.getIDs()[0]  # IDs are tuples, not array row numbers
trajectory = tp.read_particle_trajectory(pid)
table = trajectory.with_columns(
    (0.5 * (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2))
    .alias("specific_kinetic_energy")
).collect()
print(table.select("time", "specific_kinetic_energy"))
tp.plot_trajectory(pid, outname="trajectory.png")
```

Trajectories are `polars.LazyFrame` objects; call `.collect()` when you need
materialized data. The expression above is energy per unit mass in the squared
velocity units of the input, not energy in joules. Pitch-angle and drift
diagnostics need magnetic/electric fields and, for some methods, gradients
recorded in the trajectory. Inspect `tp.get_column_names()` first.

### Reconnection flux, rate, and peak-state visualization

For a sequence of 2D x-y snapshots with in-plane magnetic fields:

```python
import flekspy as fs
import matplotlib.pyplot as plt
from flekspy.reconnection import plot_reconnection_rate, plot_peak_state

series = fs.ReconnectionSeries("simulation/z=0_fluid_*.out", max_cache=4)
print(series.summary())
print(series.times, series.flux, series.rate())
az = series[0].reconnection.calc_vector_potential()
flux = series[0].reconnection.reconnected_flux(method="midplane")
fig, axes = plot_reconnection_rate(series, save_path="reconnection_rate.png")
plt.close(fig)
fig, axes = plot_peak_state(series, save_path="peak_state.png")
plt.close(fig)
```

Importing `flekspy.reconnection` registers its dataset accessor. Flux supports
`midplane` and `az` methods; the default rate is the numerical time derivative
of flux. Use distinct simulation times for derivatives. Additional peak-state
panels depend on available Hall-field, outflow, and electric-field variables.

## Guidance for changes and scientific interpretation

- Keep format-specific parsing in its reader and plotting in the relevant
  accessor or mixin. Preserve public lazy imports and existing return types.
- Preserve coordinates, dimensions, species metadata, file attributes, and
  binary/Fortran ordering. Cover both 2D and 3D paths when a change affects them.
- Check `ds.attrs["unit"]`, `ds.attrs["parameters"]`, and field metadata.
  Raw xarray arithmetic does not automatically convert physical units. The
  `.fleks.get_variable()` implementation assumes raw values already have the
  requested unit convention; do not treat `unit="si"` as a universal conversion
  of arbitrary source data. Verify normalization and species masses/charges
  before interpreting current, temperature, energy, or flux quantitatively.
- Use the restricted evaluator in `util/safe_eval.py` for expressions; do not
  replace it with unrestricted `eval()`.
- Validate numerical changes against analytical/synthetic cases and relevant
  real-data tests. Use a noninteractive Matplotlib backend (`MPLBACKEND=Agg`)
  for automated plot checks and close figures after saving.
- Follow `CONTRIBUTING.md`; update examples/docs when public behavior changes.
  Run focused tests first and the full suite when appropriate. Report unavailable
  dependencies or sample downloads separately from implementation failures.

## Further examples

- `docs/idl_data.ipynb`: IDL field loading and plotting.
- `docs/amrex_data.ipynb`: yt/AMReX workflows.
- `docs/select_and_plot_particles.ipynb`: particle selection and distributions.
- `docs/test_particle_data.ipynb`: trajectory analysis.
- `docs/reconnection.ipynb`: reconnection diagnostics and plots.
- `docs/exosphere.ipynb`: neutral density models.
- `docs/format.md` and `docs/algorithm.md`: format and algorithm details.
