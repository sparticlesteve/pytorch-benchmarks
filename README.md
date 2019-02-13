# NERSC PyTorch benchmarks

This repository contains some PyTorch benchmarks used to check
performance of PyTorch on NERSC systems.

The SLURM batch scripts for running the benchmarks are in the `scripts/` folder.

Benchmark configurations are in the `configs/` folder.

Benchmark results can be found in Jupyter notebooks in the `notebooks/` folder.

## Software analysis

We compare benchmark throughput performance on Haswell for several software
installations of PyTorch. The Intel-optimized PyTorch v1.0.0 tends to be the
fasted on our Intel-based machines.

You can see the comparisons at
[notebooks/SoftwareAnalysis.ipynb](notebooks/SoftwareAnalysis.ipynb)

## Hardware analysis

We compare benchmark throughput on Cori Haswell, KNL, and GPU.

You can see the comparisons at
[notebooks/HardwareAnalysis.ipynb](notebooks/HardwareAnalysis.ipynb)

## Scaling analysis

Here we look at throughput scaling on Cori Haswell with Intel-optimized
PyTorch v1.0.0 using the native PyTorch distributed library and MPI backend.

You can see the results at
[notebooks/ScalingAnalysis.ipynb](notebooks/ScalingAnalysis.ipynb)
