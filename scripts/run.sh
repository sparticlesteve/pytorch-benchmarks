#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH -d singleton
#SBATCH -c 32
#SBATCH -t 30
#SBATCH -J pytorch-bm-gpu
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=1.8.0
backend=nccl
models="alexnet resnet50 lstm cnn3d transformer"
clean=false
usage="$0 --version VERSION --backend BACKEND --models \"MODELS ...\" --clean CLEAN"

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --version)
            version=$2
            shift 2
            ;;
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

# Load software
module load pytorch/$version
module list
#export NCCL_DEBUG=INFO

# Run each model
for model in $models; do
    echo "running $model"
    srun -l -u python train.py configs/${model}.yaml -d $backend --rank-gpu \
        --output-dir $BENCHMARK_RESULTS_PATH/$model \
	--ranks-per-node $SLURM_NTASKS_PER_NODE
done

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
