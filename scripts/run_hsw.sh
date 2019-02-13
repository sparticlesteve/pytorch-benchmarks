#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J hsw-bm
#SBATCH -o logs/%x-%j.out

# Configuration
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/cori-hsw/N${SLURM_JOB_NUM_NODES}/v1.0.0-intel
module load pytorch/v1.0.0-intel

srun python train.py -d configs/alexnet.yaml
srun python train.py -d configs/vgg11.yaml
srun python train.py -d configs/resnet50.yaml
srun python train.py -d configs/inceptionV3.yaml
srun python train.py -d configs/lstm.yaml
srun python train.py -d configs/cnn3d.yaml
