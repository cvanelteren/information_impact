from Utils import IO
import numpy as np, scipy, os, matplotlib.pyplot as plt
from Toolbox import infcy
from Models import FastIsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file')
if __name__ == "__main__":
    args = parser.parse_args()
    print(args, args.file)
    runFile = args.file
    for k, v in IO.loadPickle(runFile).items():
        globals()[k] = v

    modelSettings = dict(\
                             graph       = graph,\
                             temperature = 0,\
                             updateType  = updateType,\
                             magSide     = magSide,\
                             nudgeType   = nudgeType)
    model = FastIsing.Ising(**modelSettings)
    magRange = np.array([CHECK]).ravel()

    # magRange = array([.9, .2])
    fitTemps = np.linspace(0, graph.number_of_nodes() / 2, tempres)
    mag, sus = model.matchMagnetization(temps = fitTemps,\
     n = int(1e3), burninSamples = 0)

    func = lambda x, a, b, c, d :  a / (1 + np.exp(b * (x - c))) + d # tanh(-a * x)* b + c
    # func = lambda x, a, b, c : a + b*exp(-c * x)
    fmag = scipy.ndimage.gaussian_filter1d(mag, .2)
    a, b = scipy.optimize.curve_fit(func, fitTemps, fmag.squeeze(), maxfev = 10000)

    matchedTemps = np.array([])
    f_root = lambda x,  c: func(x, *a) - c
    magnetizations = max(fmag) * magRange
    for m in magnetizations:
        r = scipy.optimize.root(f_root, 0, args = (m), method = 'linearmixing')#, method = 'linearmixing')
        rot = r.x if r.x > 0 else 0
        matchedTemps = np.hstack((matchedTemps, rot))

    fig, ax = plt.subplots()
    xx = np.linspace(0, max(fitTemps), 1000)
    ax.plot(xx, func(xx, *a))
    ax.scatter(matchedTemps, func(matchedTemps, *a), c ='red')
    ax.scatter(fitTemps, mag, alpha = .2)
    ax.scatter(fitTemps, fmag, alpha = .2)
    plt.setp(ax, **dict(xlabel = 'Temperature', ylabel = '<M>'))
    #            savefig(f'{targetDirectory}/temp vs mag.png')
    # show()

    # TODO: combine these? > don't as it is model specific imo
    tmp = dict(\
               fitTemps     = fitTemps, \
               matchedTemps = matchedTemps, \
               magRange     = magRange, \
               mag          = mag,\
               fmag         = fmag,\
               )

    settings = dict(
                repeats       = repeats,\
                deltas        = deltas,\
                nSamples      = nSamples,\
                step          = step,\
                burninSamples = burninSamples,\
                pulseSizes    = pulseSizes,\
                updateType    = updateType,\
                nNodes        = graph.number_of_nodes(),\
                nTrials       = nTrials,\
                # this is added
                graph         = nx.readwrite.json_graph.node_link_data(graph),\
                mapping       = model.mapping,\
                rmapping      = model.rmapping,\
                model         = type(model).__name__,\
                directory     = targetDirectory,\
                nudgeType     = nudgeType,\
                )
    settingsObject = IO.Settings(settings)
    settingsObject.save(targetDirectory)
    IO.savePickle(f'{targetDirectory}/mags.pickle', tmp)

    for t, mag in zip(matchedTemps, magRange):
        print(f'{datetime.datetime.now().isoformat()} Setting {t}')
        model.t = t # update beta
        tempDir = f'{targetDirectory}/{mag}'
        if not os.path.exists(tempDir):
            print('making directory')
            os.mkdir(tempDir)

        for trial in range(nTrials):
            from multiprocessing import cpu_count
            # st = [random.choice(model.agentStates, size = model.nNodes) for i in range(nSamples)]
            print(f'{datetime.datetime.now().isoformat()} Getting snapshots')
            # enforce no external influence
            pulse        = {}
            model.nudges = pulse
            snapshots    = infcy.getSnapShots(model, nSamples, \
                                           burninSamples = burninSamples, \
                                           steps         = step)
            # TODO: uggly, against DRY
            # always perform control
            conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)
            print(f'{datetime.datetime.now().isoformat()} Computing MI')
            # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
            if not os.path.exists(f'{tempDir}/control/'):
                os.mkdir(f'{tempDir}/control')

            props = "nSamples deltas repeats updateType".split()
            fileName = f"{tempDir}/control/{datetime.datetime.now().isoformat()}"
            fileName += "".join(f"_{key}={settings.get(key, '')}" for key in props)
            fileName += f'_pulse={pulse}'
            sr       = SimulationResult(\
                                    mi          = mi,\
                                    conditional = conditional,\
        #                                        graph       = model.graph,\
                                    px          = px,\
                                    snapshots   = snapshots)
            IO.savePickle(fileName, sr)
            from Utils.stats import KL
            for pulseSize in pulseSizes:
                pulseDir = f'{tempDir}/{pulseSize}'
                if not os.path.exists(pulseDir):
                    os.mkdir(pulseDir)
                for n in model.graph.nodes():
                    pulse        = {n : pulseSize}
                    model.nudges = pulse
                    conditional, px, mi = infcy.runMC(model, snapshots, deltas, repeats)

                    print(f'{datetime.datetime.now().isoformat()} Computing MI')

                    # snapshots, conditional, mi = infcy.reverseCalculation(nSamples, model, deltas, pulse)[-3:]
                    fileName = f"{pulseDir}/{datetime.datetime.now().isoformat()}"
                    fileName += "".join(f"_{key}={settings.get(key, '')}" for key in props)
                    fileName += f'_pulse={pulse}'
                    sr       = SimulationResult(\
                                            mi          = mi,\
                                            conditional = conditional,\
        #                                                graph       = model.graph,\
                                            px          = px,\
                                            snapshots   = snapshots)
                    IO.savePickle(fileName, sr)
    os.remove(runFile) # remove th temporary file
