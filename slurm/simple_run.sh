#!/bin/sh
# setup experiments
source activate base
python run -c "settings.toml"

file="./tasks.txt"
while IFS= read -r line;
do
    sbatch -W ./single_task.sh "$line"
done < "$file"
