#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 16      # cores requested
#SBATCH -o %x-%j.out  # send stdout to outfile
#SBATCH -e %x-%j.err  # send stderr to errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:second
#SBATCH --constraint=cpunode

FILE=$1
source activate base
cd $HOME/information_impact
echo "starting $FILE at $SLURM_JOB_ID"

if [ -z $FILE ]
then
   echo "File not found running new"
   srun python new.py --id $SLURM_JOB_ID
else
   echo $FILE
   srun python new.py --file $FILE --id $SLURM_JOB_ID
fi

echo "job finished"
