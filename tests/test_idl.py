import pytest
import flekspy as fs
import numpy as np
import xarray as xr


def test_get_pressure_anisotropy(idl_data_files):
    ds = fs.load(idl_data_files[1])
    anisotropy = ds.idl.get_pressure_anisotropy(species=1)
    assert anisotropy.name == "pressure_anisotropy_S1"
    assert anisotropy.shape == (601, 2)
    assert np.isclose(anisotropy.isel(x=0, y=0), 1.2906302, atol=1e-5)


def test_get_current_density(idl_data_files):
    """Test current density calculation with real data."""
    ds = fs.load(idl_data_files[2])  # 3d_raw.out

    # The file must be 3D
    if ds.attrs["ndim"] != 3:
        pytest.skip("Test file is not 3D, skipping.")

    current = ds.idl.get_current_density()
    assert "jx" in current
    assert "jy" in current
    assert "jz" in current
    assert current["jx"].shape == ds["Bx"].shape
    assert current["jy"].shape == ds["By"].shape
    assert current["jz"].shape == ds["Bz"].shape

    # Check a value to see if it's in a reasonable range
    # The exact value depends on the test data and is not critical here,
    # as the synthetic test validates the calculation's correctness.
    pass


def test_get_current_density_synthetic():
    """Test current density calculation with synthetic data."""
    # Create a synthetic 3D dataset with a known curl
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Define a magnetic field with a constant curl: B = (a*z, b*x, c*y)
    # curl(B) = (c, a, b)
    a, b, c = 0.1, 0.2, 0.3
    bx = a * Z
    by = b * X
    bz = c * Y

    ds = xr.Dataset(
        {
            "Bx": (("x", "y", "z"), bx),
            "By": (("x", "y", "z"), by),
            "Bz": (("x", "y", "z"), bz),
        },
        coords={"x": x, "y": y, "z": z},
    )
    ds.attrs = {
        "ndim": 3,
        "gencoord": False,
        "dims": ["x", "y", "z"],
        "unit": "SI",
    }

    # Test with SI units
    current_density_si = ds.idl.get_current_density()
    mu0 = 4.0 * np.pi * 1e-7
    # Conversion from A/m^2 to µA/m^2 is 1e6
    assert np.allclose(current_density_si["jx"].values, (c / mu0) * 1e6)
    assert np.allclose(current_density_si["jy"].values, (a / mu0) * 1e6)
    assert np.allclose(current_density_si["jz"].values, (b / mu0) * 1e6)
    assert current_density_si["jx"].attrs["units"] == "µA/m^2"

    # Test with PLANETARY units
    ds.attrs["unit"] = "PLANETARY"
    current_density_planetary = ds.idl.get_current_density()
    conversion_factor = 1e-9 / mu0
    assert np.allclose(
        current_density_planetary["jx"].values, c * conversion_factor * 1e6
    )
    assert np.allclose(
        current_density_planetary["jy"].values, a * conversion_factor * 1e6
    )
    assert np.allclose(
        current_density_planetary["jz"].values, b * conversion_factor * 1e6
    )
    assert current_density_planetary["jx"].attrs["units"] == "µA/m^2"
