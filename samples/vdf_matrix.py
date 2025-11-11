import flekspy
import flekspy.amrex
import glob
import os  # Added for creating output directory
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def generate_regions_grid(
    x_start, x_end, x_step, y_start, y_end, y_step, box_dx_half, box_dy_half
):
    """
    Generates a grid of dictionary-defined regions.
    """
    x_centers = np.arange(x_start, x_end + x_step, x_step)
    y_centers = np.arange(y_start, y_end + y_step, y_step)

    regions_to_plot = []
    for xc in x_centers:
        for yc in y_centers:
            region = {
                "x_range": (xc - box_dx_half, xc + box_dx_half),
                "y_range": (yc - box_dy_half, yc + box_dy_half),
            }
            regions_to_plot.append(region)

    print(f"Generated {len(regions_to_plot)} regions for plotting.")
    return regions_to_plot


def calculate_histograms(particle_file, regions, var_x, var_y, hist_range, hist_bins):
    """
    Loads a single particle file and calculates histograms for all defined regions.

    Returns:
        - A new list of region dicts, now including the 'hist_data'.
        - The global maximum value found across all histograms for normalization.
    """
    print(f"Loading particle data from: {particle_file}")
    try:
        pd = flekspy.amrex.AMReXParticleData(particle_file)
    except Exception as e:
        print(f"Error loading particle data: {e}")
        return [], 0.0  # Return empty list and zero max

    print("Pre-calculating histograms for all regions...")
    global_vmax = 0.0
    processed_regions = []

    for i, region in enumerate(regions):
        new_region = region.copy()
        try:
            density_data = pd.get_phase_space_density(
                x_variable=var_x,
                y_variable=var_y,
                x_range=new_region["x_range"],
                y_range=new_region["y_range"],
                hist_range=hist_range,
                bins=hist_bins,
                normalize=True,
            )

            if density_data:
                hist_data, _, _, _ = density_data
                new_region["hist_data"] = hist_data
                current_max = hist_data.max()
                if current_max > global_vmax:
                    global_vmax = current_max
            else:
                new_region["hist_data"] = None

            processed_regions.append(new_region)

        except Exception as e:
            print(f"Warning: Could not process region {i}: {e}")
            new_region["hist_data"] = None
            processed_regions.append(new_region)

    print(f"All histograms calculated. Global max count: {global_vmax}")
    return processed_regions, global_vmax


def create_plot(
    field_file,
    regions_with_data,
    global_vmax,
    output_filename,
    hist_range,
    hist_bins,
    inset_cmap,
    line_color,
    dash_length,
):
    """
    Creates and saves the main plot with all insets for a single time step.
    """
    print(f"Creating main field plot for: {field_file}")

    # --- 4. Define Colormap and Normalization for INSETS ---
    # Use max(global_vmax, 1e-10) to avoid errors if vmax is zero
    inset_norm = mcolors.LogNorm(vmin=1e-7, vmax=max(global_vmax, 1e-10))

    # --- 5. Create the Main Background Plot ---
    try:
        ds = flekspy.load(field_file)
    except Exception as e:
        print(f"Error loading field data {field_file}: {e}")
        return  # Skip this plot

    fig, ax = plt.subplots(1, 1, figsize=(12, 15), constrained_layout=True)

    # Plot the background field
    norm = mcolors.SymLogNorm(0.1, linscale=0.2, vmin=-20, vmax=20)
    ds.Bz.plot.imshow(
        ax=ax,
        x="x",
        y="y",
        cmap="PRGn",
        norm=norm,
        cbar_kwargs=dict(label="By [nT]", pad=0.005, aspect=40),
    )
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
    ax.tick_params(
        which="minor",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    ax.set_xlabel(r"x [$R_E$]", fontsize="x-large")
    ax.set_ylabel(r"z [$R_E$]", fontsize="x-large")
    time_val = ds.attrs.get("time", 0.0)  # Use .get for safety
    ax.set_title(f"t = {time_val:.1f}s", fontsize="xx-large")

    ax.set_xlim(ds.coords["x"].values.min(), ds.coords["x"].values.max())
    ax.set_ylim(ds.coords["y"].values.min(), ds.coords["y"].values.max())

    # --- 6. Add Inset Plots to the Main Figure ---

    # Calculate pixel indices for reference lines
    vxmin, vxmax = hist_range[0]
    vzmin, vzmax = hist_range[1]
    bins_x, bins_y = hist_bins

    # Find pixel index for the vertical line (vx)
    vx_zero_idx = int(np.floor((0.0 - vxmin) / (vxmax - vxmin) * bins_x))
    # Find pixel index for the horizontal line (vy)
    vy_zero_idx = int(np.floor((0.0 - vzmin) / (vzmax - vzmin) * bins_y))

    print(
        f"Drawing reference lines at pixel indices: vx={vx_zero_idx}, vy={vy_zero_idx}"
    )
    print("Adding inset plots...")

    for region in regions_with_data:
        if region.get("hist_data") is None:
            continue  # Skip if histogram calculation failed

        hist_data = region["hist_data"]

        # Calculate the center of the region for the anchor point
        x_center = (region["x_range"][0] + region["x_range"][1]) / 2.0
        y_center = (region["y_range"][0] + region["y_range"][1]) / 2.0

        # Convert the 2D data array to an RGBA image
        rgba_image = inset_cmap(inset_norm(hist_data))

        # --- Draw lines manually on the RGBA image ---
        # Draw vertical line at col = vx_zero_idx
        if 0 <= vx_zero_idx < bins_x:
            for i in range(0, bins_y, dash_length * 2):
                rgba_image[i : i + dash_length, vx_zero_idx, :] = line_color

        # Draw horizontal line at row = vy_zero_idx
        if 0 <= vy_zero_idx < bins_y:
            for i in range(0, bins_x, dash_length * 2):
                rgba_image[vy_zero_idx, i : i + dash_length, :] = line_color

        # Create the OffsetImage artist
        image_artist = OffsetImage(rgba_image, zoom=1.0, origin="lower")

        # Create the AnnotationBbox to place the image
        ann_box = AnnotationBbox(
            image_artist,
            (x_center, y_center),
            frameon=False,
            pad=0.1,
            boxcoords="data",
        )
        ax.add_artist(ann_box)

    print(f"Saving final plot to {output_filename}...")
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)  # IMPORTANT: Close the figure to free memory


# --- Main execution block ---
def main():
    # --- 1. Define Data Paths and Plotting Regions ---
    parser = argparse.ArgumentParser(description="Generate VDF matrix plots.")
    parser.add_argument("datapath", help="Path to the simulation output directory (e.g., PC/).")
    args = parser.parse_args()
    topdir = args.datapath

    output_dir = "./frames"  # Directory to save output images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of FIELD files
    # We will use these as the "driver" to find matching particle files
    field_search = topdir + "z=0_fluid_region0_0_t*.out"
    field_files = sorted(glob.glob(field_search))

    if not field_files:
        print(f"No field files found matching: {field_search}")
        print("Exiting.")
        return

    print(f"Found {len(field_files)} field files to process.")

    # --- Region Definitions ---
    regions = generate_regions_grid(
        x_start=9.5,
        x_end=29.5,
        x_step=5.0,
        y_start=-29.5,
        y_end=29.5,
        y_step=5.0,
        box_dx_half=0.5,
        box_dy_half=0.5,
    )

    # --- 2. Define Phase Space Histogram Parameters (Static) ---
    var_x = "velocity_x"
    var_y = "velocity_y"
    vxmin, vxmax, vzmin, vzmax = -2500, 2500, -2500, 2500
    hist_range = [[vxmin, vxmax], [vzmin, vzmax]]
    hist_bins = (64, 64)
    line_color = mcolors.to_rgba("k", alpha=0.7)
    dash_length = 4
    inset_cmap = plt.get_cmap("turbo")

    # --- 3. Main Processing Loop ---
    # Loop over the field files
    for i, field_file in enumerate(field_files):
        print(f"\n--- Processing Frame {i} ---")

        # --- Parse field file to get time and iteration ---
        field_time_str = ""
        field_iter_num = -1
        try:
            f_filename = os.path.basename(field_file)
            pattern = re.compile(r"_t(\d+)_n(\d+)\.out")
            match = pattern.search(f_filename)
            if match:
                field_time_str = "t" + match.group(1)
                field_iter_num = int(match.group(2))
            else:
                # handle error
                print(f"Warning: Could not parse field file {f_filename}. Skipping.")
                continue
            print(f"  Field:    {f_filename}")

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse field file {f_filename}: {e}. Skipping.")
            continue
        # --- End of field file parsing ---

        # --- Find all particle files for this time stamp ---
        particle_pattern = os.path.join(
            topdir, f"cut_particle_region0_4_{field_time_str}_n*_amrex"
        )
        candidate_particle_files = glob.glob(particle_pattern)

        if not candidate_particle_files:
            print(
                f"  Warning: No particle files found for time {field_time_str}. Skipping."
            )
            continue

        # --- Find the closest iteration number ---
        closest_particle_file = None
        min_iter_diff = float("inf")

        for p_file in candidate_particle_files:
            try:
                p_filename = os.path.basename(p_file)
                p_parts = p_filename.split("_")
                p_iter_str = p_parts[5]  # e.g., n00042888
                p_iter_num = int(p_iter_str[1:])  # e.g., 42888

                iter_diff = abs(p_iter_num - field_iter_num)

                if iter_diff < min_iter_diff:
                    min_iter_diff = iter_diff
                    closest_particle_file = p_file

            except (IndexError, ValueError):
                print(
                    f"  Warning: Could not parse candidate particle file {p_filename}. Skipping this candidate."
                )
                continue

        # --- Check if we found a valid match ---
        if closest_particle_file is None:
            print(
                f"  Warning: Found particle files for {field_time_str}, but could not parse any. Skipping frame."
            )
            continue

        particle_file = closest_particle_file
        print(
            f"  Particle: {os.path.basename(particle_file)} (closest iter, diff={min_iter_diff})"
        )

        # --- Both files are found, proceed with plotting ---
        output_filename = os.path.join(output_dir, f"frame_e_{i:04d}.png")

        try:
            # Step 1: Calculate all histograms for this timestep
            regions_with_data, global_vmax = calculate_histograms(
                particle_file, regions, var_x, var_y, hist_range, hist_bins
            )

            # Step 2: Create the plot if we have valid data
            if global_vmax > 0:
                create_plot(
                    field_file,
                    regions_with_data,
                    global_vmax,
                    output_filename,
                    hist_range,
                    hist_bins,
                    inset_cmap,
                    line_color,
                    dash_length,
                )
            else:
                print("Skipping frame: No particle data found (global_vmax is 0).")

        except Exception as e:
            print(f"!!-- FATAL ERROR on frame {i} --!!")
            print(f"Field: {field_file}, Particle: {particle_file}")
            print(f"Error: {e}")
            # Continue to the next frame
            pass

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
