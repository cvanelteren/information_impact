echo "Number of simulations running"
squeue | grep cveltere | wc -l
echo "Disk space in scratch" && du -sh /var/scratch/cveltere
squeue | grep cveltere
cat *"$(squeue -u cveltere -h | awk '$0=$1')"*
