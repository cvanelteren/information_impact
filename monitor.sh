echo "Number of simulations running"
squeue | grep cveltere | wc -l
echo "Disk space in scratch" && du -sh /var/scratch/cveltere
squeue | grep cveltere
squeue | grep -u cveltere | akw "$0=$1" | cat *"$(xargs)""
