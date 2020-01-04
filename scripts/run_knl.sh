#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -d singleton
#SBATCH -J pytorch-bm-knl
#SBATCH -o logs/%x-%j.out

set -e

# Options
version=v1.3.1
clean=false
backend=mpi
models="alexnet vgg11 resnet50 inceptionV3 lstm cnn3d"
if [ $# -ge 1 ]; then models=$@; fi

# Configuration
export OMP_NUM_THREADS=68
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export BENCHMARK_RESULTS_PATH=$SCRATCH/pytorch-benchmarks/knl-$version-n${SLURM_JOB_NUM_NODES}
if $clean; then
    [ -d $BENCHMARK_RESULTS_PATH ] && rm -rf $BENCHMARK_RESULTS_PATH
fi
module load pytorch/$version

# Run each model
for m in $models; do
    srun -l python train.py -d $backend configs/${m}.yaml
done

echo "Collecting benchmark results..."
python parse.py $BENCHMARK_RESULTS_PATH -o $BENCHMARK_RESULTS_PATH/results.txt
