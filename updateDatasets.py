import os, subprocess

root = '/var/scratch/cveltere/tester'
# root = 'Data/tester/'
# root = 'Data/2019-09-29T14:33:58.112218'

if __name__ == '__main__':
    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        subprocess.Popen(f'sbatch new_run.sh {path}'.split())
        # subprocess.Popen(f'python new.py --file {path}'.split())
