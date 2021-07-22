#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o %x-%j.out  # send stdout to outfile
#SBATCH -e %x-%j.err  # send stderr to errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:second
#SBATCH --constraint=cpunode
source activate

echo $1
srun python simple_toml.py -r $1
