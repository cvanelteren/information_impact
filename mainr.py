import os, datetime
with open('simulations.txt', 'r') as f:
    files = list(f.readlines())

nodename = os.uname().nodename
if any([nodename in i for i in \
'fs4 node'.split()]):
    now = datetime.datetime.now().isoformat()
    runCommand = 'sbatch single_run.sh'
else:
    runCommand = 'python single_run.py --file'

import subprocess
import os
import time
print(os.getpid())
import sys, time

start = time.time()
threshold = 1 * 60  + start
# set limit of jobs 
LIMIT = 10

counts = int(\
subprocess.run('squeue | grep cveltere | wc -l' , shell = 1, \
capture_output = 1).stdout.strip()\
)

REST = LIMIT - counts 
if REST < 0:
    REST = 0
for count in range(REST):
    runFile = files[0].strip('\n')
    files.pop(0)
    subprocess.Popen([*runCommand.split(), runFile])
    time.sleep(0.1)
    if time.time() > threshold:
        break

with open('simulations.txt', 'w') as f:
    f.writelines([i for i in files])
    # call itself
# time.sleep(30)
if files:
    subprocess.Popen('python mainr.py'.split(), \
            stdin = None, \
            stdout = None, \
            stderr = None)

print('Done')
