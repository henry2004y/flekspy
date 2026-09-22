"""Tests for IDLSeries container."""

import pytest
import numpy as np
import xarray as xr
from flekspy.idl.series import IDLSeries, read_idl_header


def test_read_idl_header_real(idl_data_files):
    """Test fast header reading on actual IDL files."""
    for fpath in idl_data_files:
        header = read_idl_header(fpath)
        assert "time" in header
        assert "iter" in header
        assert "ndim" in header
        assert "variables" in header
        assert isinstance(header["time"], float)
        assert isinstance(header["iter"], int)


def test_idl_series_loading(idl_data_files):
    """Test IDLSeries lazy loading, indexing, slicing, and nearest frame lookup."""
    # Use a subset of available files
    files = idl_data_files[:2]
    series = IDLSeries(files, eager=False, max_cache=2)

    assert len(series) == len(files)
    assert len(series.times) == len(files)
    assert len(series.iters) == len(files)

    # Test indexing
    ds0 = series[0]
    assert isinstance(ds0, xr.Dataset)

    # Test negative indexing
    ds_last = series[-1]
    assert isinstance(ds_last, xr.Dataset)

    # Test slicing
    sub = series[0:1]
    assert isinstance(sub, IDLSeries)
    assert len(sub) == 1

    # Test nearest frame lookup
    target_time = float(series.times[0])
    idx, ds_near = series.get_frame(time=target_time)
    assert idx == 0
    assert isinstance(ds_near, xr.Dataset)

    # Test cache clearing
    series.clear_cache()
    assert len(series._cache) == 0


def test_idl_series_eager(idl_data_files):
    """Test IDLSeries in eager mode."""
    files = idl_data_files[:2]
    series = IDLSeries(files, eager=True)
    assert series.eager is True
    assert series._datasets is not None
    assert len(series._datasets) == len(files)
    ds0 = series[0]
    assert isinstance(ds0, xr.Dataset)
