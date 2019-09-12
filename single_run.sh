#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 16      # cores requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:second
source activate base
cd $HOME/information_impact

echo "starting { $1 }"
srun python3 single_run.py --file $1
echo "job finished"
