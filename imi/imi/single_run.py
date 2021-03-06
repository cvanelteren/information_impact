import argparse, datetime, os


from Utils import IO
from Toolbox import infcy
# init arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--file')
import time
if __name__ == '__main__':
    # load settings
    # parse input
    args = parser.parse_args()
    runFile = args.file

    # load data to global
    print("IM AM IN RUNFILE", runFile)
    settings = IO.loadPickle(runFile)
    model    = settings.get('model')


    # init model
     # run experiment
    if not settings.get('pulse'):
        # run snapshots (cheap enough)
        snaps = {k : settings.get(k) for\
                k in 'nSamples burninSamples steps'.split()\
                }
        snapshots = infcy.getSnapShots(
                            model, **snaps
                            )
    # load nudges
    else:

        # think of something to extract the control
        trial = settings.get('trial')
        mag   = settings.get('ratio')[0]
        root = os.path.join(*tuple(i for i in runFile.split('/')[:-1]))
        root = '/' + root
        control = os.path.join(root, \
                    f"trial={trial}_r={mag}_{{}}.pickle"\
                    )
        snapshots = {}
        while not snapshots:
            try:
                snapshots = IO.loadPickle(control).snapshots
            except:
                time.sleep(1)
    deltas, repeats = [settings.get(k) for k in 'deltas repeats'.split()]
    conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)

    # store the results
    store = dict(\
    mi          = mi, \
    conditional = conditional,\
    px          = px,\
    snapshots   = snapshots,\
    )
    # reduce dataset size
    # TODO: make nicer
    if settings.get('pulse'):
        store['mi']          = []
        store['conditional'] = []
        store['snapshots']   = []
    sr = IO.SimulationResult(**store)

    # TODO: remove

    IO.savePickle(runFile, sr)
