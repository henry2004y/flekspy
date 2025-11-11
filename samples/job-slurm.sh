#!/bin/bash
#SBATCH -q regular           # Quality of Service/Queue (e.g., debug, regular, shared)
#SBATCH -C cpu               # Node constraint (e.g., cpu, gpu)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # We want to run 1 task per node (11 tasks total)
#SBATCH --cpus-per-task=10   # We want each of those tasks to have 10 CPUs available for its multiprocessing pool
#SBATCH -t 01:00:00          # Wall-clock time limit (HH:MM:SS)
###SBATCH -A m9999           # Your NERSC allocation project
#SBATCH -J hdf               # Job name
#SBATCH -o %x-%j.out         # Standard output file (name-jobid.out)
#SBATCH -e %x-%j.err         # Standard error file (name-jobid.err)

# --- Job Setup ---
echo "Starting particle processing job..."
echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "Total tasks (nodes): $SLURM_NTASKS"
echo "CPUs per task (cores per node): $SLURM_CPUS_PER_TASK"

# --- Define Your Parameters ---
TOTAL_PARTICLES=100
DATA_PATH="run/PC/test_particles"
SPECIES=0 # 0 for electron

# --- Load Modules ---
module load python
conda activate fleks

# --- Run the parallel job ---
# srun will launch SLURM_NTASKS (11) copies of the command.
# Each copy will run on a different node (due to --ntasks-per-node=1).
# Each copy will automatically get the SLURM environment variables
# (like SLURM_PROCID) it needs to find its correct particle range.
srun python convert_testparticle_parallel.py \
    --total-particles "$TOTAL_PARTICLES" \
    --datapath "$DATA_PATH" \
    --ispecies "$SPECIES"

echo "Particle processing job complete."