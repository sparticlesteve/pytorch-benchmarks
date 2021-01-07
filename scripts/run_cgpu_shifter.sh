#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -d singleton
#SBATCH -c 10
#SBATCH -t 30
#SBATCH -J pytorch-bm-cgpu
#SBATCH --image nersc/pytorch:ngc-20.10-v0
#SBATCH -o logs/%x-%j.out

set -e

# Options
backend=nccl
models="alexnet resnet50 lstm cnn3d transformer"
clean=false
usage="$0 --backend BACKEND --models \"MODELS ...\" --clean CLEAN"

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --backend)
            backend=$2
            shift 2
            ;;
        --models)
            models=$2
            shift 2
            ;;
        --clean)
            clean=$2
            shift 2
            ;;
        *)
            echo "Usage: $usage"
            exit 1
            ;;
    esac
done

# Configuration
version=$(echo $SLURM_SPANK_SHIFTER_IMAGEREQUEST | tr ':' ' ' | awk '{print $2}')
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/results/gpu-$version-$backend-n$SLURM_NTASKS
if $clean; then
    [ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
fi

# Print settings
echo "Running PyTorch benchmarks with"
echo "version $version"
echo "backend $backend"
echo "models $models"
echo "clean $clean"
echo "writing outputs to $BENCHMARK_RESULTS_PATH"

# Run each model
for model in $models; do
    srun -l shifter --volume="/dev/infiniband:/sys/class/infiniband_verbs" \
        python train.py configs/${model}.yaml -d $backend --rank-gpu \
        --output-dir $BENCHMARK_RESULTS_PATH/$model
done

echo "Collecting benchmark results..."
srun -n 1 -N 1 shifter python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
