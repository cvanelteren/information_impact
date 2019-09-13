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
if files:
    a = len(files)
    runFile = files[0].strip('\n')
    files.pop(0)

    # Popen([*runCommand.split(), runFile],\
            #stdin = None, stdout = None,\
            # stderr = None)
    with open('simulations.txt', 'w') as f:
        f.writelines([i + '\n' for i in files])
    # call itself
    print('calling myself')
    subprocess.Popen('python mainr.py'.split())
    print('new')
    assert len(files) < a
