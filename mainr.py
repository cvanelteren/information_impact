import os, datetime
with open('simulations.txt') as f:
    files = list(f.readlines())

nodename = os.uname().nodename
if any([nodename in i for i in \
'fs4 node'.split()]):
    now = datetime.datetime.now().isoformat()
    runCommand = 'sbatch single_run.sh'
else:
    runCommand = 'python3 single_run.py --file'

from subprocess import Popen
if files:
    runFile = files[0].strip('\n')
    files.pop(0)
    Popen([*runCommand.split(), runFile])
    with open('simulations.txt') as f:
        f.writelines([i + '\n' for i in files])
    # call itself
    call('python mainr.py'.split())
