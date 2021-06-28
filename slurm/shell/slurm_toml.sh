#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o %x-%j.out  # send stdout to outfile
#SBATCH -e %x-%j.err  # send stderr to errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:second
#SBATCH --constraint=cpunode
source activate

nodename=$(uname -n)
output_dir="data/"
if [[ $nodename =~ "ivi-*" ]]; then
   cd ~/information_impact/slurm
   output_dir=~/information_impact/slurm/data
fi

if [[ $nodename =~ "fs2" ]]; then
    cd /var/scratch/cveltere/information_impact/slurm
   output_dir=/var/scratch/cveltere/data
fi

echo $nodename
echo $output_dir
srun python run_toml.py -s "settings.toml"
