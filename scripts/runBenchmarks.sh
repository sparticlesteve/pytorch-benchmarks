#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

# Single node benchmarking for now.

. setup.sh
module list
which python

srun python ./train.py configs/alexnet.yaml
srun python ./train.py configs/vgg11.yaml
srun python ./train.py configs/resnet50.yaml
srun python ./train.py configs/inceptionV3.yaml
srun python ./train.py configs/lstm.yaml
srun python ./train.py configs/cnn3d.yaml
