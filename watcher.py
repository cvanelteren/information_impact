__author__ = "Casper van Elteren"
"""
Idiot script to fool the fileserver
"""
# compile
from subprocess import call, Popen
import time, os, re, sys
call('python compile.py build_ext --inplace'.split())
TIMELIMIT = time.time() + 120

nodename = os.uname().nodename
for i in 'fs4 node'.split():
    runCommand = 'python new.py --file'
    if nodename in i:
        runCommand = 'sbatch new_single.sh'
        break
SCRIPT = os.path.realpath(__file__)
ROOT   = os.path.dirname(SCRIPT)
if __name__ == '__main__':

    # poll and call scripts
    print('Starting polling', end = '')
    while True:
        for file in os.listdir(ROOT):
            pattern = re.findall('sim-\d+.pickle', file)
            # print(pattern)
            if pattern:
                Popen(f'{runCommand} {os.path.join(ROOT, file)}'.split())
        # kill and restart
        if time.time() > TIMELIMIT:
            # path = 'watcher.py'
            Popen(f'python {SCRIPT}'.split())
            sys.exit()
        print('.', end = '')
        time.sleep(10)
