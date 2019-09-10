from Toolbox import infcy
from Utils import IO
import argparse

def equilbriate(model: Model, \
                nSamples: int,\
                burnin : int,\
                step: int,\
                ) -> list:
    return temperatures
# init arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--file')

if __name__ == '__main__':
    # load settings
    # parse input
    args = parser.parse_args()
    runFile = args.file

    # load data to global
    for k, v in IO.loadPickle(runFile).items():
        print(f'Loaded {k}')
        globals()[k] = v

    # init model
    model = model(**modelSettings)
    if equilbriate:
        pass
    else:
         # run experiment
        control = 1 if not pulse else 0
        if control:
            # run snapshots (cheap enough)
            snapshots = infcy.getSnapShots(
                                model, nSamples,\
                                burninSamples = burninSamples,\
                                steps = step\
                                )
            model.nudges = pulse
        # load nudges
        else:
            snapshots = IO.loadSnapshots(fileName)
            model.nudges = pulse

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
        if not control:
            store['mi']          = []
            store['conditional'] = []
            store['snapshots']   = []
        sr = IO.SimulationResult(**store)
        IO.savePickle(fileName, sr)
    os.remove(fileName)
