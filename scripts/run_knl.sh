#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J pytorch-bm-knl
#SBATCH -o logs/%x-%j.out

# Configuration
export OMP_NUM_THREADS=68
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/cori-knl/N${SLURM_JOB_NUM_NODES}/v1.1.0
module load pytorch/v1.1.0

srun python train.py configs/alexnet.yaml
srun python train.py configs/vgg11.yaml
srun python train.py configs/resnet50.yaml
srun python train.py configs/inceptionV3.yaml
srun python train.py configs/lstm.yaml
srun python train.py configs/cnn3d.yaml
