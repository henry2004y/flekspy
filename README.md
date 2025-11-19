# flekspy: FLEKS Python toolkit

<p align="center">
  <a href="https://badge.fury.io/py/flekspy">
    <img src="https://badge.fury.io/py/flekspy.svg" alt="PyPI version" height="18">
  </a>
  <a href="https://github.com/henry2004y/flekspy/actions">
    <img src="https://github.com/henry2004y/flekspy/actions/workflows/CI.yml/badge.svg">
  </a>
  <a href="https://henry2004y.github.io/flekspy/">
    <img src="https://img.shields.io/badge/docs-dev-blue">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue">
  </a>
  <a href="https://app.codecov.io/gh/henry2004y/flekspy/">
    <img src="https://img.shields.io/codecov/c/github/henry2004y/flekspy">
  </a>
  <a href="https://doi.org/10.5281/zenodo.16912301">
    <img src="https://zenodo.org/badge/307380211.svg">
  </a>
  
</p>

Python package for processing [FLEKS](https://github.com/SWMFsoftware/FLEKS) (FLexible Exascale Kinetic Simulator) data.

## Installation

```bash
python -m pip install flekspy
```

## Usage

`flekspy` can load files generated from FLEKS.

```python
import flekspy

ds = flekspy.load("sample_data/3*amrex")
```

Plotting is supported via Matplotlib and YT. For more detailed usage and contribution guide, please refer to the [documentation](https://henry2004y.github.io/flekspy/).

### ParaView Plugin

`flekspy` provides a ParaView plugin for visualizing FLEKS data. To use the plugin, follow these steps:

1. Make sure the ParaView version is 5.13 or higher.

2. Install flekspy for pvpython.
  - Go into the ParaView directory and create a virtual environment for Python

```bash
./bin/pvpython -m venv .venv
```

  - Check the Python version of `pvpython`

```bash
./bin/pvpython -c "import sys; print(sys.version)"
```

  - Install flekspy in the directory right in the venv. You may need to adjust the target path if `pvpython` does not detect `flekspy`. Assuming the Python version is 3.10.11, run

```bash
python3 -m pip  install --only-binary=:all: --python-version 3.10.11 --target ./.venv/lib/python3.10/site-packages/ flekspy
```

3. Launch ParaView with the virtual environment

```bash
./bin/paraview --venv .venv
```

4. In Tools > Manage Plugins, load the `BATSRUSReader` plugin.

## License

`flekspy` is licensed under the terms of the MIT license.
