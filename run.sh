#!/bin/bash

#SBATCH --job-name="moisturetest"
#SBATCH --time=00:20:00     # walltime
#SBATCH --cpus-per-task=8  # number of cores
#SBATCH --gres=gpu:p100:1       # 1 GPU
#SBATCH --mem-per-cpu=4G   # memory per CPU core

module purge
module load julia/1.8.5
module load cuda/11.0

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'
julia --project moisture.jl
