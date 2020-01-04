#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node 8
#SBATCH -c 10
#SBATCH -t 30
#SBATCH -J pytorch-bm-cgpu
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=v1.3.1
clean=false
backend=gloo
models="alexnet vgg11 resnet50 inceptionV3 lstm cnn3d"
if [ $# -ge 1 ]; then models=$@; fi

# Configuration
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/gpu-$version-n$SLURM_NTASKS
if $clean; then
    [ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
fi

# Load software
module load pytorch/$version-gpu

# Run each model
for m in $models; do
    srun -l python train.py -d $backend --rank-gpu configs/${m}.yaml
done

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
