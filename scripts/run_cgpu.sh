#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -d singleton
#SBATCH -c 10
#SBATCH -t 30
#SBATCH -J pytorch-bm-cgpu
#SBATCH --image registry.services.nersc.gov/wbhimji/nvidia-pytorch:19.12-py3
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=v1.4.0.ngc
clean=true
backend=nccl
models="alexnet resnet50 lstm cnn3d"
if [ $# -ge 1 ]; then models=$@; fi

# Configuration
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/results/gpu-$version-$backend-n$SLURM_NTASKS
if $clean; then
    [ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
fi

# Run each model
for m in $models; do
    srun -l shifter --volume="/dev/infiniband:/sys/class/infiniband_verbs" \
        python train.py -d $backend --rank-gpu configs/${m}.yaml
done

echo "Collecting benchmark results..."
srun shifter python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
