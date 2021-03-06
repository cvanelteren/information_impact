#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 16      # cores requested
#SBATCH -o %x-%j.out  # send stdout to outfile
#SBATCH -e %x-%j.err  # send stderr to errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:second
#SBATCH --constraint=cpunode
source activate base
cd $HOME/information_impact
echo "starting { $1 }"
srun python3 new.py --file $1 --id $SLURM_JOB_ID
echo "job finished"
