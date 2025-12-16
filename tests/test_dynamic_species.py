
import unittest
from unittest.mock import MagicMock
from flekspy.yt.yt import FLEKSFieldInfo

class TestDynamicFields(unittest.TestCase):
    def test_dynamic_species_registration(self):
        import yt.geometry.geometry_enum as geometry_enum

        # Mock the dataset
        mock_ds = MagicMock()
        # Set geometry to CARTESIAN to avoid assert_never(geometry) in yt
        mock_ds.geometry = geometry_enum.Geometry.CARTESIAN

        # Create a fake list of raw fields that includes a high index species
        mock_ds.index.raw_fields = ["Bx", "By", "Bz", "rhos0", "rhos99", "uxs99"]
        mock_ds.nodal_flags = {f: 0 for f in mock_ds.index.raw_fields}

        # Configure field_units on the mock dataset to return string units
        mock_ds.field_units.get.return_value = "code_length" # Default for unknown

        # Initialize FieldInfo
        # Pass the raw fields as field_list to simulate what yt does
        field_list = [("raw", f) for f in mock_ds.index.raw_fields]
        fi = FLEKSFieldInfo(mock_ds, field_list)

        # Setup fluid fields
        fi.setup_fluid_fields()

        # Verify that dynamic aliasing logic was executed and created expected fields
        # Note: in mocked environment without a real index hierarchy,
        # 'alias' might not behave exactly as in real yt unless perfectly mocked.
        # But we can check if self.alias was called or if the fields are in 'fi'.

        # If alias failed because original_name not in self (which alias() checks),
        # we need to make sure original_name IS in self.

        # Since we passed field_list to __init__, ('raw', 'rhos0') etc should be in fi.
        self.assertIn(("raw", "rhos0"), fi)
        self.assertIn(("raw", "rhos99"), fi)

        # Check if mesh aliases exist
        self.assertIn(("mesh", "rhos0"), fi)
        self.assertEqual(fi[("mesh", "rhos0")].units, "code_density")

        self.assertIn(("mesh", "rhos99"), fi)
        self.assertEqual(fi[("mesh", "rhos99")].units, "code_density")
        self.assertIn("_{s99}", fi[("mesh", "rhos99")].display_name)

        self.assertIn(("mesh", "uxs99"), fi)
        self.assertEqual(fi[("mesh", "uxs99")].units, "code_velocity")
        self.assertIn("_{s99}", fi[("mesh", "uxs99")].display_name)

        # Check that unknown patterns are not registered blindly (though regular fields loop handles aliases for known_other_fields)
        # Bx is in known_other_fields
        self.assertIn(("mesh", "Bx"), fi)

if __name__ == "__main__":
    unittest.main()
