import os, subprocess

root = '/var/scratch/cveltere/stupid/'

if __name__ == '__main__':
    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        subprocess.Popen(f'sbatch new_run.sh {path}'.split())
        #if 'settings' not in path and path.endswith('.bk'):
        #    print(path)

        # subprocess.Popen(f'python new.py --file {path}'.split())
