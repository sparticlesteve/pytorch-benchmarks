#!/bin/bash

# Launch all benchmark runs for this version

# Scaling on Haswell
sbatch -N 1 -q debug scripts/run_hsw.sh
sbatch -N 2 -q regular scripts/run_hsw.sh
sbatch -N 4 -q regular scripts/run_hsw.sh
sbatch -N 8 -q debug scripts/run_hsw.sh
sbatch -N 16 -q debug scripts/run_hsw.sh
sbatch -N 32 -q debug scripts/run_hsw.sh
sbatch -N 64 -q debug scripts/run_hsw.sh

# Scaling on KNL
sbatch -N 1 scripts/run_knl.sh
sbatch -N 2 scripts/run_knl.sh
sbatch -N 4 scripts/run_knl.sh
sbatch -N 8 scripts/run_knl.sh
sbatch -N 16 scripts/run_knl.sh
sbatch -N 32 scripts/run_knl.sh
sbatch -N 64 scripts/run_knl.sh

# Scaling on Cori-GPU
module purge
module load esslurm
sbatch -n 1 scripts/run_cgpu.sh
sbatch -n 2 scripts/run_cgpu.sh
sbatch -n 4 scripts/run_cgpu.sh
sbatch -n 8 scripts/run_cgpu.sh
sbatch -n 16 scripts/run_cgpu.sh
sbatch -n 32 -t 5:00:00 scripts/run_cgpu.sh
sbatch -n 64 -t 5:00:00 scripts/run_cgpu.sh
