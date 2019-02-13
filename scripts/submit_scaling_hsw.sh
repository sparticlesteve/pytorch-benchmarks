#!/bin/bash

args="-J hsw-scaling -C haswell -d singleton"
sbatch $args -N 1 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 2 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 4 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 8 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 16 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 32 -q debug -t 30 scripts/run_hsw.sh
sbatch $args -N 64 -q debug -t 30 scripts/run_hsw.sh
