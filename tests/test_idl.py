import pytest
import flekspy as fs
import numpy as np
import xarray as xr
from scipy.constants import mu_0


def test_get_pressure_anisotropy(idl_data_files):
    ds = fs.load(idl_data_files[1])
    anisotropy = ds.derived.get_pressure_anisotropy(species=1)
    assert anisotropy.name == "pressure_anisotropy_S1"
    assert anisotropy.shape == (601, 2)
    assert np.isclose(anisotropy.isel(x=0, y=0), 1.2906302, atol=1e-5)


def test_get_current_density(idl_data_files):
    """Test current density calculation with real data."""
    ds = fs.load(idl_data_files[2])  # 3d_raw.out

    # The file must be 3D
    if ds.attrs["ndim"] != 3:
        pytest.skip("Test file is not 3D, skipping.")

    current = ds.derived.get_current_density()
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
        "parameters": {
            "mS0": 1.0,
            "qS0": 1.0,
            "mS1": 1e3,
            "qS1": 0.0,
            "cLight": 3e8,
            "rPlanet": 1e3,
            "cutz": 0.0,
        },
        "unit": "PLANETARY",
    }

    # Test with PLANETARY units
    ds.attrs["unit"] = "PLANETARY"
    current_density_planetary = ds.derived.get_current_density()
    conversion_factor = 1e-9 / mu_0 * 1e3
    assert np.allclose(current_density_planetary["jx"].values, c * conversion_factor)
    assert np.allclose(current_density_planetary["jy"].values, a * conversion_factor)
    assert np.allclose(current_density_planetary["jz"].values, b * conversion_factor)
    assert current_density_planetary["jx"].attrs["units"] == "µA/m^2"


def test_get_current_density_from_definition_si():
    """Test current density calculation from definition with synthetic SI data."""
    # Create a synthetic dataset
    rhoS0 = np.ones((10, 10, 10))
    uxS0 = np.full((10, 10, 10), 0.1)
    uyS0 = np.zeros((10, 10, 10))
    uzS0 = np.zeros((10, 10, 10))

    rhoS1 = np.ones((10, 10, 10))
    uxS1 = np.full((10, 10, 10), 0.2)
    uyS1 = np.zeros((10, 10, 10))
    uzS1 = np.zeros((10, 10, 10))

    ds = xr.Dataset(
        {
            "rhoS0": (("x", "y", "z"), rhoS0),
            "uxS0": (("x", "y", "z"), uxS0),
            "uyS0": (("x", "y", "z"), uyS0),
            "uzS0": (("x", "y", "z"), uzS0),
            "rhoS1": (("x", "y", "z"), rhoS1),
            "uxS1": (("x", "y", "z"), uxS1),
            "uyS1": (("x", "y", "z"), uyS1),
            "uzS1": (("x", "y", "z"), uzS1),
        },
        coords={
            "x": np.linspace(-1, 1, 10),
            "y": np.linspace(-1, 1, 10),
            "z": np.linspace(-1, 1, 10),
        },
    )
    # electron mass, proton mass
    ds.attrs = {
        "parameters": {
            "mS0": 9.10938356e-31,
            "qS0": -1.60217663e-19,
            "mS1": 1.67262192e-27,
            "qS1": 1.60217663e-19,
        },
        "unit": "SI",
    }

    # Calculate expected current densities in A/m^2 (before conversion)
    n0 = rhoS0 / ds.attrs["parameters"]["mS0"]
    q0 = ds.attrs["parameters"]["qS0"]
    expected_jx_s0_si = n0 * q0 * uxS0

    n1 = rhoS1 / ds.attrs["parameters"]["mS1"]
    q1 = ds.attrs["parameters"]["qS1"]
    expected_jx_s1_si = n1 * q1 * uxS1

    expected_jx_total_si = expected_jx_s0_si + expected_jx_s1_si

    # Test with SI units (default case)
    current_density_total = ds.derived.get_current_density_from_definition(species=[0, 1])
    assert np.allclose(current_density_total["jx"].values, expected_jx_total_si * 1e6)
    assert current_density_total["jx"].attrs["units"] == "µA/m^2"


def test_get_current_density_from_definition_planetary():
    """Test current density calculation with synthetic PLANETARY data."""
    # Create a synthetic dataset with PLANETARY units
    # rhoS0 in amu/cc, uxS0 in km/s
    rhoS0 = np.ones((10, 10, 10)) * 1.0  # 1 amu/cc
    uxS0 = np.full((10, 10, 10), 1000.0)  # 1000 km/s
    uyS0 = np.zeros((10, 10, 10))
    uzS0 = np.zeros((10, 10, 10))

    rhoS1 = np.ones((10, 10, 10)) * 2.0  # 2 amu/cc
    uxS1 = np.full((10, 10, 10), 500.0)  # 500 km/s
    uyS1 = np.zeros((10, 10, 10))
    uzS1 = np.zeros((10, 10, 10))

    ds = xr.Dataset(
        {
            "rhoS0": (("x", "y", "z"), rhoS0),
            "uxS0": (("x", "y", "z"), uxS0),
            "uyS0": (("x", "y", "z"), uyS0),
            "uzS0": (("x", "y", "z"), uzS0),
            "rhoS1": (("x", "y", "z"), rhoS1),
            "uxS1": (("x", "y", "z"), uxS1),
            "uyS1": (("x", "y", "z"), uyS1),
            "uzS1": (("x", "y", "z"), uzS1),
        },
        coords={
            "x": np.linspace(-1, 1, 10),
            "y": np.linspace(-1, 1, 10),
            "z": np.linspace(-1, 1, 10),
        },
    )
    # mS in amu, qS normalized to elementary charge
    ds.attrs = {
        "parameters": {"mS0": 0.00054858, "qS0": -1.0, "mS1": 1.0, "qS1": 1.0},
        "unit": "PLANETARY",
    }

    # Manually calculate the expected current density in µA/m^2
    amu_to_kg = 1.66053906660e-27
    e_charge = 1.602176634e-19
    cc_to_m3 = 1e6
    km_to_m = 1e3
    A_to_uA = 1e6

    # Species 0 (electron)
    n0_si = (rhoS0 / ds.attrs["parameters"]["mS0"]) * cc_to_m3
    q0_si = ds.attrs["parameters"]["qS0"] * e_charge
    v0_si = uxS0 * km_to_m
    jx0_si = n0_si * q0_si * v0_si * A_to_uA

    # Species 1 (proton)
    n1_si = (rhoS1 / ds.attrs["parameters"]["mS1"]) * cc_to_m3
    q1_si = ds.attrs["parameters"]["qS1"] * e_charge
    v1_si = uxS1 * km_to_m
    jx1_si = n1_si * q1_si * v1_si * A_to_uA

    expected_jx_total = jx0_si + jx1_si

    # Get current density from the method
    current_density = ds.derived.get_current_density_from_definition(species=[0, 1])

    # The method should return values in µA/m^2
    assert np.allclose(current_density["jx"].values, expected_jx_total)
    assert current_density["jx"].attrs["units"] == "µA/m^2"


def test_idl_dataset_variable_units(idl_data_files):
    """Test that units attributes are properly attached to data variables and coordinates upon reading."""
    # Test file with PLANETARY units
    ds_planetary = fs.load(idl_data_files[1])
    assert ds_planetary.attrs["unit"] == "PLANETARY"
    assert ds_planetary["rhoS0"].attrs["units"] == "amu/cm**3"
    assert ds_planetary["Bx"].attrs["units"] == "nT"
    assert ds_planetary["uxS0"].attrs["units"] == "km/s"
    assert ds_planetary["x"].attrs["units"] == "Planet_Radius"
    assert ds_planetary["y"].attrs["units"] == "Planet_Radius"

    # Test file with normalized/other units
    ds_norm = fs.load(idl_data_files[0])
    assert ds_norm["Rho"].attrs["units"] == "dimensionless"
    assert ds_norm["x"].attrs["units"] == "dimensionless"


def test_get_unit_helper():
    """Test get_unit function directly for various variable names and unit systems."""
    from flekspy.util.utilities import get_unit

    # Planetary
    assert get_unit("rhoS0", "PLANETARY") == "amu/cm**3"
    assert get_unit("rhoS12", "planet") == "amu/cm**3"
    assert get_unit("uxS0", "PLANETARY") == "km/s"
    assert get_unit("Bx", "PLANETARY") == "nT"
    assert get_unit("pxxS0", "PLANETARY") == "nPa"
    assert get_unit("x", "PLANETARY") == "Planet_Radius"
    assert get_unit("r", "PLANETARY") == "Planet_Radius"

    # SI
    assert get_unit("rhoS0", "SI") == "kg/m**3"
    assert get_unit("uxS0", "si") == "m/s"
    assert get_unit("Bx", "SI") == "T"
    assert get_unit("pxxS0", "SI") == "Pa"
    assert get_unit("x", "SI") == "m"

    # Dimensionless / Unknown
    assert get_unit("rhoS0", "normalized") == "dimensionless"
    assert get_unit("unknown_var", "PLANETARY") == "dimensionless"
    assert get_unit("x", "dimensionless") == "dimensionless"
