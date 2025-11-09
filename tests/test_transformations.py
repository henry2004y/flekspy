import numpy as np
import pytest
from flekspy.util.transformations import create_field_transform

@pytest.fixture
def sample_data():
    """Provides sample data for testing."""
    return np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ])

@pytest.fixture
def component_names():
    """Provides component names for sample data."""
    return ["x", "y", "velocity_x", "velocity_y", "velocity_z"]

def test_create_field_transform_b_only(sample_data, component_names):
    """Tests the transformation with only the B-field."""
    b_field = np.array([1, 0, 0])
    transform_func = create_field_transform(b_field, component_names)
    transformed_data, new_names = transform_func(sample_data)

    assert new_names == ["x", "y", "v_parallel", "v_perp"]
    assert transformed_data.shape == (3, 4)

    # Expected values
    v_parallel_expected = sample_data[:, 2]
    v_perp_expected = np.sqrt(sample_data[:, 3]**2 + sample_data[:, 4]**2)

    assert np.allclose(transformed_data[:, 2], v_parallel_expected)
    assert np.allclose(transformed_data[:, 3], v_perp_expected)

def test_create_field_transform_b_and_e(sample_data, component_names):
    """Tests the transformation with both B and E-fields."""
    b_field = np.array([0, 1, 0])
    e_field = np.array([1, 0, 0])
    transform_func = create_field_transform(b_field, component_names, e_field=e_field)
    transformed_data, new_names = transform_func(sample_data)

    assert new_names == ["x", "y", "v_B", "v_E_perp", "v_BxE_perp"]
    assert transformed_data.shape == (3, 5)

    # Expected values
    v_B_expected = sample_data[:, 3]
    v_E_perp_expected = sample_data[:, 2]
    v_BxE_perp_expected = -sample_data[:, 4]

    assert np.allclose(transformed_data[:, 2], v_B_expected)
    assert np.allclose(transformed_data[:, 3], v_E_perp_expected)
    assert np.allclose(transformed_data[:, 4], v_BxE_perp_expected)

def test_create_field_transform_e_parallel_to_b(sample_data, component_names):
    """Tests that a ValueError is raised when E is parallel to B."""
    b_field = np.array([1, 0, 0])
    e_field = np.array([2, 0, 0])
    transform_func = create_field_transform(b_field, component_names, e_field=e_field)

    with pytest.raises(ValueError, match="E field is parallel to B field"):
        transform_func(sample_data)

def test_create_field_transform_missing_velocity_components(sample_data):
    """Tests that a ValueError is raised if velocity components are missing."""
    b_field = np.array([1, 0, 0])
    component_names = ["x", "y", "z"]
    transform_func = create_field_transform(b_field, component_names)

    with pytest.raises(ValueError, match="Velocity components.*not found"):
        transform_func(sample_data)
