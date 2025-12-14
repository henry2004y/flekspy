import pytest
import matplotlib.pyplot as plt
import flekspy as fs
import os


def test_unstructured_plots(idl_data_files):
    """Test plotting functions with unstructured IDL data."""
    # The 4th file (index 3) is the unstructured one: bx0_mhd_6_t00000100_n00000352.out
    filename = idl_data_files[3]

    # Ensure the file exists (it should, via the fixture)
    if not os.path.exists(filename):
        pytest.skip(f"Test file {filename} not found.")

    ds = fs.load(filename)

    # Check that it is indeed unstructured
    if not ds.attrs.get("gencoord", False):
        pytest.fail("Test data is not unstructured (gencoord is not True).")

    # Use non-interactive backend for testing
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()

    # 1. Test basic plot
    try:
        # Plot 'Rho' which is a scalar
        ds.fleks.plot("Rho", axes=[ax], f=fig)
        # Verify that a collection was added (tripcolor/tricontourf creates collections)
        assert (
            len(ax.collections) > 0
        ), "plot() did not create any collections for unstructured data."
    except Exception as e:
        pytest.fail(f"ds.fleks.plot failed: {e}")
    finally:
        ax.clear()

    # 2. Test add_contour
    try:
        ds.fleks.add_contour(ax, "Rho")
        # Contour also adds collections
        assert len(ax.collections) > 0, "add_contour() did not create any collections."
    except Exception as e:
        pytest.fail(f"ds.fleks.add_contour failed: {e}")
    finally:
        ax.clear()

    # 3. Test add_stream
    try:
        # Streamplot requires vector components. Use Ux and Uy.
        # Note: The file has Ux, Uy, Uz.
        ds.fleks.add_stream(ax, "Ux", "Uy", nx=20, ny=20)

        # Streamplot adds a PatchCollection (arrows) and LineCollection (streamlines)
        # So we expect both collections and patches or just collections depending on implementation.
        # streamplot returns a StreamplotSet which holds .lines and .arrows
        # The axes should have these added.
        # In matplotlib, LineCollection is in ax.collections
        # PatchCollection is also in ax.collections

        has_collections = len(ax.collections) > 0
        has_patches = len(ax.patches) > 0

        assert (
            has_collections or has_patches
        ), "add_stream() did not add collections or patches."

    except Exception as e:
        pytest.fail(f"ds.fleks.add_stream failed: {e}")
    finally:
        plt.close(fig)
