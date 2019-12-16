#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH -t 30
#SBATCH -J pytorch-bm-cgpu
#SBATCH -o logs/%x-%j.out

# Configuration
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/cori-gpu/v1.2.0-gpu
module load pytorch/v1.2.0-gpu

srun python train.py --device cuda configs/alexnet.yaml
srun python train.py --device cuda configs/vgg11.yaml
srun python train.py --device cuda configs/resnet50.yaml
srun python train.py --device cuda configs/inceptionV3.yaml
srun python train.py --device cuda configs/lstm.yaml
srun python train.py --device cuda configs/cnn3d.yaml
