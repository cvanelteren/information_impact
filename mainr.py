import os, datetime
with open('simulations.txt', 'r') as f:
    files = list(f.readlines())

nodename = os.uname().nodename
if any([nodename in i for i in \
'fs4 node'.split()]):
    now = datetime.datetime.now().isoformat()
    runCommand = 'sbatch single_run.sh'
else:
    runCommand = 'python3 single_run.py --file'

import subprocess
import os
import time
print(os.getpid())
import sys, time

start = time.time()
threshold = 1 * 60  + start
while files and time.time() < threshold:
    runFile = files[0].strip('\n')
    files.pop(0)
    subprocess.Popen([*runCommand.split(), runFile])
    time.sleep(.5)
with open('simulations.txt', 'w') as f:
    f.writelines([i + '\n' for i in files])
    # call itself
if files:
    subprocess.Popen('python mainr.py'.split(), \
            stdin = None, \
            stdout = None, \
            stderr = None)
print('Done')
