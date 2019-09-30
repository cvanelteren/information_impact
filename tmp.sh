echo "number of simulations running:" 
squeue | grep cveltere | wc -l
echo "Number of simulations to do " 
cat simulations.txt | wc -l 

squeue | grep cveltere

