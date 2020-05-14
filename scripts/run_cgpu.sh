#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -d singleton
#SBATCH -c 10
#SBATCH -t 30
#SBATCH -J pytorch-bm-cgpu
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=v1.5.0
clean=true
backend=nccl
models="alexnet resnet50 lstm cnn3d"
if [ $# -ge 1 ]; then models=$@; fi

# Configuration
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/results/gpu-$version-$backend-n$SLURM_NTASKS
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
