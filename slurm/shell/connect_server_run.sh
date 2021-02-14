git add settings.toml
git commit -m 'running on server'
git push
ssh cveltere@fs2.das5.science.uva.nl 'cd /var/scratch/cveltere/information_impact && git pull --force&& cd slurm && sbatch slurm_toml.sh'
