# flekspy

<p align="center">
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
</p>

Python package for processing FLEKS (FLexible Exascale Kinetic Simulator) data.

## Installation

```bash
python -m pip install flekspy
```

## Usage

`flekspy` can load files generated from FLEKS.

```python
import flekspy

ds = fleks.load("sample_data/3*amrex")
```

Plotting is supported via Matplotlib and YT. For more detailed usage, please refer to the [documentation](https://henry2004y.github.io/flekspy/).

## License

`flekspy` is licensed under the terms of the MIT license.
