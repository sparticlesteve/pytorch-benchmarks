#!/bin/bash

sbatch -d singleton -N 1 scripts/run_hsw.sh
sbatch -d singleton -N 2 scripts/run_hsw.sh
sbatch -d singleton -N 4 scripts/run_hsw.sh
sbatch -d singleton -N 8 scripts/run_hsw.sh
sbatch -d singleton -N 16 scripts/run_hsw.sh

sbatch -d singleton -N 32 scripts/run_hsw.sh
sbatch -d singleton -N 64 scripts/run_hsw.sh

sbatch -d singleton -N 128 -q regular scripts/run_hsw.sh
sbatch -d singleton -N 256 -q regular scripts/run_hsw.sh
