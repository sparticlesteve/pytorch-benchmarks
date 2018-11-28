#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

# Single node benchmarking for now.

. setup.sh
module list

python ./train.py configs/benchmark_alexnet.yaml
python ./train.py configs/benchmark_vgg11.yaml
python ./train.py configs/benchmark_resnet50.yaml
python ./train.py configs/benchmark_inceptionV3.yaml
