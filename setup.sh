# Example environment setup script for Cori
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export BENCHMARK_VERSION=v1.0-intel

installDir=$SCRATCH/conda/pytorch/v1.0.0-intel
conda activate $installDir
export LD_LIBRARY_PATH=$installDir/lib:$LD_LIBRARY_PATH
