import numpy as np
from flekspy import FLEKSTP
import multiprocessing
import time
import os
import argparse  # Import argparse for command-line arguments
from scipy.constants import proton_mass, elementary_charge


# --- Worker Function ---
# This function is run by each parallel process on a single node.
def save_chunk_worker(chunk_data):
    """
    Worker function to save a single chunk of particle IDs/indexes.
    'chunk_data' is a tuple containing:
    (particle_index_chunk, global_chunk_id, dataPath, iSpecies)
    """
    # This 'chunk' is a list of global particle indexes, e.g., [10500, 10501, ...]
    particle_indexes, global_chunk_id, data_path_local, ispecies_local = chunk_data

    # Use a padded-zero format for easier file sorting
    output_filename = f"test/trajectories_chunk_{global_chunk_id:05d}.h5"

    try:
        # Each process creates its own tp object
        # We print the global chunk ID for better logging
        print(f"[Chunk {global_chunk_id}]: Initializing FLEKSTP...")
        if ispecies_local == 1:
            tp_local = FLEKSTP(data_path_local, iSpecies=ispecies_local)
        elif ispecies_local == 0:
            mi2me = 25
            tp_local = FLEKSTP(
                data_path_local,
                iSpecies=ispecies_local,
                mass=proton_mass / mi2me,
                charge=-elementary_charge,
            )
        else:
            print(
                f"[Chunk {global_chunk_id}]: Error: Unknown iSpecies {ispecies_local}"
            )
            return

        print(
            f"[Chunk {global_chunk_id}]: Saving {len(particle_indexes)} particles (by index) to {output_filename}..."
        )

        # We pass the list of global indexes directly
        tp_local.save_trajectories(particle_indexes.tolist(), filename=output_filename)

        print(f"[Chunk {global_chunk_id}]: Finished saving {output_filename}.")
        return f"Chunk {global_chunk_id} successful"

    except Exception as e:
        print(f"[Chunk {global_chunk_id}]: FAILED with error: {e}")
        return f"Chunk {global_chunk_id} FAILED"


# --- Main script ---
# This guard is still essential
if __name__ == "__main__":
    main_start_time = time.time()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Process a range of particles in parallel."
    )
    parser.add_argument(
        "--total-particles",
        type=int,
        required=True,
        help="Total number of particles to process.",
    )
    parser.add_argument(
        "--datapath", type=str, required=True, help="Path to the particle data."
    )
    parser.add_argument(
        "--ispecies",
        type=int,
        required=True,
        help="Species index (0 for electron, 1 for proton).",
    )
    args = parser.parse_args()

    # --- Get SLURM Configuration ---
    # These variables are set by srun
    # Fallback to 1 (for serial testing) if not in a SLURM job
    total_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    # Get the number of CPUs allocated for this task, default to 10 if not set
    n_chunks = int(os.environ.get("SLURM_CPUS_PER_TASK", 10))

    # Ensure the output directory exists
    os.makedirs("test", exist_ok=True)

    print(
        f"[Task {task_id}]: Started. Total tasks = {total_tasks}, My CPUs = {n_chunks}"
    )

    # --- Chunking Logic (Distributed) ---
    # Calculate the particle range for THIS task (node)
    particles_per_task = args.total_particles // total_tasks
    start_index = task_id * particles_per_task
    end_index = (task_id + 1) * particles_per_task

    # The last task (task_id == total_tasks - 1) gets the remainder
    if task_id == total_tasks - 1:
        end_index = args.total_particles

    nparticles_this_task = end_index - start_index

    print(
        f"[Task {task_id}]: Responsible for {nparticles_this_task} particles: global indexes {start_index} to {end_index-1}"
    )

    # --- Create chunks of indexes for this task ---
    # These are the actual global indexes this task will process
    particle_indexes = np.arange(start_index, end_index, dtype=np.uint64)

    # Split this task's work into 'n_chunks' for its local multiprocessing pool
    index_chunks = np.array_split(particle_indexes, n_chunks)

    print(
        f"[Task {task_id}]: Splitting its {nparticles_this_task} particles into {n_chunks} local chunks."
    )

    # --- Prepare Tasks for the local Pool ---
    tasks = []
    for i, local_chunk in enumerate(index_chunks):
        # Calculate a globally unique chunk ID
        # This prevents file collisions (e.g., node0_chunk0.h5 and node1_chunk0.h5)
        global_chunk_id = (task_id * n_chunks) + i
        tasks.append((local_chunk, global_chunk_id, args.datapath, args.ispecies))

    # --- Run in Parallel (on this node) ---
    print(
        f"[Task {task_id}]: Starting a local processing pool with {n_chunks} workers."
    )
    with multiprocessing.Pool(processes=n_chunks) as pool:
        results = pool.map(save_chunk_worker, tasks)

    print(f"\n[Task {task_id}]: --- All local processes finished ---")
    for res in results:
        print(f"[Task {task_id}]: {res}")

    main_end_time = time.time()
    print(
        f"\n[Task {task_id}]: Total time taken: {main_end_time - main_start_time:.2f} seconds."
    )
