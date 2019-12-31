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
clean=false
models="alexnet vgg11 resnet50 inceptionV3 lstm cnn3d"
if [ $# -ge 1 ]; then models=$@; fi

# Configuration
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/gpu-v1.2.0-n$SLURM_NTASKS
if $clean; then
    [ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
fi

# Load software
module load gcc/7.3.0
module load cuda/10.1.168
module load openmpi/4.0.1-ucx-1.6
module load pytorch/v1.2.0-gpu
# Library fix
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH
export UCX_LOG_LEVEL=error

# Run each model
for m in $models; do
    srun -l python train.py -d gloo --rank-gpu configs/${m}.yaml
done

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
