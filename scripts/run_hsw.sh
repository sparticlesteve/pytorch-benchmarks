#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -t 30
#SBATCH -d singleton
#SBATCH -J pytorch-bm-hsw
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=v1.5.0
backend=mpi
models="alexnet resnet50 lstm cnn3d"
clean=true
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
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
# TODO: improve this to avoid using env var in the config file
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/results/hsw-$version-$backend-n${SLURM_JOB_NUM_NODES}
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

module load pytorch/$version
module list

# Run each model
for m in $models; do
    srun -l python train.py -d $backend configs/${m}.yaml
done

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
