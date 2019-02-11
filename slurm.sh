#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 16      # cores requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 10000000:00:00  # time requested in hour:minute:second
source activate
cd $HOME/information_impact
python setup.py build_ext --inplace
srun python run.py
