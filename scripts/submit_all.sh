#!/bin/bash

# Launch all benchmark runs for this version

# Scaling on Haswell
sbatch -N 1 scripts/run_hsw.sh
sbatch -N 2 scripts/run_hsw.sh
sbatch -N 4 scripts/run_hsw.sh
sbatch -N 8 scripts/run_hsw.sh
sbatch -N 16 scripts/run_hsw.sh
sbatch -N 32 scripts/run_hsw.sh
sbatch -N 64 scripts/run_hsw.sh

# Scaling on KNL
sbatch -N 1 scripts/run_knl.sh
sbatch -N 2 scripts/run_knl.sh
sbatch -N 4 scripts/run_knl.sh
sbatch -N 8 scripts/run_knl.sh
sbatch -N 16 scripts/run_knl.sh
sbatch -N 32 scripts/run_knl.sh
sbatch -N 64 scripts/run_knl.sh

# Scaling on Haswell with Gloo
sbatch -N 1 scripts/run_hsw.sh --backend gloo
sbatch -N 2 scripts/run_hsw.sh --backend gloo
sbatch -N 4 scripts/run_hsw.sh --backend gloo
sbatch -N 8 scripts/run_hsw.sh --backend gloo
sbatch -N 16 scripts/run_hsw.sh --backend gloo
sbatch -N 32 scripts/run_hsw.sh --backend gloo
sbatch -N 64 scripts/run_hsw.sh --backend gloo

# Software version comparisons - cpu
sbatch scripts/run_hsw.sh --version v1.3.1
sbatch scripts/run_hsw.sh --version v1.2.0
sbatch scripts/run_hsw.sh --version v1.4.0
sbatch scripts/run_hsw.sh --version v1.5.0
sbatch scripts/run_hsw.sh --version v1.6.0
#sbatch scripts/run_hsw.sh --version 1.7.1

# Scaling on Cori-GPU with NCCL
module purge
module load cgpu
sbatch -n 1 scripts/run_cgpu.sh --backend nccl
sbatch -n 2 scripts/run_cgpu.sh --backend nccl
sbatch -n 4 scripts/run_cgpu.sh --backend nccl
sbatch -n 8 scripts/run_cgpu.sh --backend nccl
sbatch -n 16 scripts/run_cgpu.sh --backend nccl
sbatch -n 32 scripts/run_cgpu.sh --backend nccl
sbatch -n 64 scripts/run_cgpu.sh --backend nccl

# Scaling on Cori-GPU with Gloo
sbatch -n 1 scripts/run_cgpu.sh --backend gloo
sbatch -n 2 scripts/run_cgpu.sh --backend gloo
sbatch -n 4 scripts/run_cgpu.sh --backend gloo
sbatch -n 8 scripts/run_cgpu.sh --backend gloo
sbatch -n 16 scripts/run_cgpu.sh --backend gloo
sbatch -n 32 scripts/run_cgpu.sh --backend gloo
sbatch -n 64 scripts/run_cgpu.sh --backend gloo

# Software version comparisons - gpu
sbatch -n 1 scripts/run_cgpu.sh --version v1.2.0
sbatch -n 1 scripts/run_cgpu.sh --version v1.3.1
sbatch -n 1 scripts/run_cgpu.sh --version v1.4.0
sbatch -n 1 scripts/run_cgpu.sh --version v1.5.1
sbatch -n 1 scripts/run_cgpu.sh --version v1.6.0
sbatch -n 1 scripts/run_cgpu.sh --version 1.7.0
sbatch -n 1 scripts/run_cgpu_shifter.sh

# Performance on DGX-A100 with NCCL
module purge
module load dgx
sbatch -n 1 scripts/run_dgx.sh --backend nccl
sbatch -n 2 scripts/run_dgx.sh --backend nccl
sbatch -n 4 scripts/run_dgx.sh --backend nccl
sbatch -n 8 scripts/run_dgx.sh --backend nccl
