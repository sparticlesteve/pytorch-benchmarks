#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -d singleton
#SBATCH -J pytorch-bm-hsw
#SBATCH -o logs/%x-%j.out

set -e

# Configuration
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/hsw-v1.2.0-n${SLURM_JOB_NUM_NODES}
[ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
module load pytorch/v1.2.0

srun -l python train.py -d configs/alexnet.yaml
srun -l python train.py -d configs/vgg11.yaml
srun -l python train.py -d configs/resnet50.yaml
srun -l python train.py -d configs/inceptionV3.yaml
srun -l python train.py -d configs/lstm.yaml
srun -l python train.py -d configs/cnn3d.yaml

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
