import numpy as np, os, re
from Utils import stats, IO

ROOT = '/var/scratch/cveltere/2019-09-22T12:04:44.952513'


def loadSettings(root):
    settings = {}
    for (r, dirs, files) in os.walk(root):
        for file in files:
            if 'settings' in  file:
                path = os.path.join(r, file)
                settings[r] = IO.loadPickle(path)
    return settings


# setup data
settings = loadSettings(ROOT)

data = {}
for k, v in settings.items():
    print(v)
    eq      = v.get('equilibrium')
    nTemps  = len(eq.get('ratios'))
    nTrials = v.get('nTrials')
    nPulse  = len(v.get('pulseSizes'))# v.get('graph').number_of_nodes() + 1
    deltas  = v.get('deltas')
    nNodes  = v.get('graph').number_of_nodes()

    s =  (nNodes, nTrials, nPulse, nTemps, deltas // 2 - 1)
    data[k] = np.zeros(s)


from tqdm import tqdm_notebook as tqdm
from Utils import stats
import ast

def loadDataFilesSingle(fileName, **kwargs):
    fileName, setting = fileName
    # model        = setting.get('model')
    repeats      = setting.get('repeats')
    pulses       = setting.get('pulseSizes')
    temperatures = setting.get('equilibrium').get('ratios')

    # extract a mapper to be consistent [is this across datasets as well]
    # add 1 if zero is in the setting file, zero corresponds to a control
    tmpPulses = {i: idx + 1 if 0 not in pulses else idx for idx, i in enumerate(pulses)}
    # print(pulses, 0 in pulses); assert 0
    tmpTemps  = {i: idx for idx, i in enumerate(temperatures)}

    deltas = setting.get('deltas')

    # load data
    lData = IO.loadPickle(os.path.join(fileName))

    # load file
    file = fileName.strip('.pickle')
    file = file.split('/')[-1]
    trial, mag, pulse = file.split('_')

    trialidx = int(trial.split('=')[-1])
    tempidx  = tmpTemps[float(mag.split('=')[-1])]

    intervention = ast.literal_eval(pulse)
    # control
    if not intervention:
        nodeidx = np.arange(setting.get('graph').number_of_nodes())
        simData = lData.mi.T[:, :deltas // 2 - 1]

        bias = stats.panzeriTrevesCorrection(lData.px, lData.conditional, repeats)
        simData -= bias[:deltas // 2 - 1, :].T
        pulseidx = 0
    # intervention
    else:
        try:
            node = next(iter(intervention))
            nodeidx = int(node)

            pulseSize = intervention[node]

            pulseidx = tmpPulses[pulseSize]
            # assert pulseidx != 0
            # load corresponding control
            controlFile = '_'.join(i for i in [trial, mag, '{}'])
            path = os.path.join(*fileName.split('/')[:-1])
            # print('>', path)
            # print('>>', controlFile)
            cpx = IO.loadPickle(os.path.join(path, controlFile)).px

            simData = np.nansum(stats.KL2(lData.px, cpx), axis = -1)[deltas // 2 + 1:]
            # print(simData)
        # missing data
        except Exception as e:
            print(e, lData.px, IO.loadPickle(os.path.join(path, controlFile)))
            simData = np.NaN
    s = (nodeidx, trialidx, pulseidx, tempidx)
    return (fileName, s, simData)


def loadDataFiles(fileName):

    # fileName.split('/')[]
    datsetname, setting = x

    path         = os.path.join(datasetname, 'data')

    # model        = setting.get('model')
    repeats      = setting.get('repeats')
    pulses       = setting.get('pulseSizes')
    temperatures = setting.get('equilibrium').get('ratios')

    # extract a mapper to be consistent [is this across datasets as well]
    tmpPulses = {i: idx for idx, i in enumerate(pulses)}
    tmpTemps  = {i: idx for idx, i in enumerate(temperatures)}

    deltas = setting.get('deltas')


    eq      = setting.get('equilibrium')
    nTemps  = len(eq.get('ratios'))
    nTrials = setting.get('nTrials')
    nPulse  = len(tmpPulses)
    deltas  = setting.get('deltas')
    nNodes  = setting.get('graph').number_of_nodes()

    s =  (nNodes, nTrials, nPulse, nTemps, deltas // 2 - 1)
    data = np.zeros(s, dtype = float)
    for file in os.listdir(path):

        # load data
        lData = IO.loadPickle(os.path.join(path, file))

        # load file
        file = file.strip('.pickle')
        trial, mag, pulse = file.split('_')

        trialidx = int(trial.split('=')[-1])
        tempidx  = tmpTemps[float(mag.split('=')[-1])]

        intervention = ast.literal_eval(pulse)
        # control
        if not intervention:
            nodeidx = np.arange(setting.get('graph').number_of_nodes())
            simData = lData.mi.T[:, :deltas // 2 - 1]

            bias = stats.panzeriTrevesCorrection(lData.px, lData.conditional, repeats)
            simData -= bias[:deltas // 2 - 1, :].T
            pulseidx = 0
        # intervention
        else:
            try:
                node = next(iter(intervention))
                nodeidx = int(node)

                pulseSize = intervention[node]

                pulseidx = tmpPulses[pulseSize]
                assert pulseidx != 0
                # load corresponding control
                controlFile = '_'.join(i for i in [trial, mag, '{}'])
                cpx = IO.loadPickle(os.path.join(path, controlFile)).px
                simData = np.nansum(stats.KL2(lData.px, cpx), axis = -1)[deltas // 2 + 1:]
            # missing data
            except:
                simData = np.NaN
        data[nodeidx, trialidx, pulseidx, tempidx, :] = simData
    return datasetname, data
if __name__ == "__main__":
    import multiprocessing as mp
    from functools import partial
    func  = partial(loadDataFiles, settings  = settings)
    with mp.Pool(mp.cpu_count()) as p:
        p.map(func, data.items())


    # normalize data

    def norm(x):
        s = x.shape
        x = x.reshape(-1)
        x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        return x.reshape(s)

    import itertools
    for k, v in data.items():
        nodes, trials, pulses, temps, deltas = v.shape
        for (i, j, k) in itertools.product(*[range(i) for i in v.shape[1:-1]]):
            v[:, i, j, k] = norm(v[:, i, j,k])



    def worker(sample):
        # tmp workaround
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1).T
        auc = np.zeros((len(sample), 2))
        x = np.arange(sample.shape[-1])

        tmp = np.arange(x.max() * 100, x.max() * 150, 3)

        x = np.concatenate((x, tmp))
        padded = np.zeros(tmp.size)

        sample = np.array([\
                           np.concatenate((i, padded)) for i in sample\
                          ])
        coeffs, errors = plotz.fit(sample, func, x = x, params = fitParam)
        for nodei, c in enumerate(coeffs):
            tmp = 0
            F      = lambda x: func(x, *c)
            tmp, _ = scipy.integrate.quad(F, 0, LIMIT)
            auc[nodei, 0] = tmp
            auc[nodei, 1] = errors[nodei]
            if errors[nodei] > .1:
                print('error large')
        auc[auc < np.finfo(auc.dtype).eps ] = 0
        return auc[:, 0]
    from tqdm import tqdm

    double = lambda x, b, c, d, e, f, g: b * np.exp(-c*(x - g)) + d * np.exp(- e * (x-f))

    double_= lambda x, b, c, d, e, f, g:b * np.exp(-c*(x - g)) + d * np.exp(- e * (x - f ))
    single = lambda x, a, b, c : a + b * np.exp(-c * x)
    single_= lambda x, a, b, c : a + b * np.exp(-c * x)
    special= lambda x, a, b, c, d: a  + b * np.exp(- (x)**c - d)

    func        = double
    p0          = np.ones((func.__code__.co_argcount - 1)); # p0[0] = 0
    fitParam    = dict(\
                       maxfev = int(1e6), \
                       bounds = (0, np.inf), p0 = p0,\
                       jac = 'cs', \
                      )
    aucs = {}
    for k, v in data.items():
        setting = settings[k]

        LIMIT = np.inf
        s = v.shape
        v = v.reshape(-1, s[-1])

        with mp.Pool(mp.cpu_count()) as p:
            auc = np.zeros(len(v))
            for idx, i in enumerate(p.imap(worker, tqdm(v))):
                auc[idx] = i
        auc = auc.reshape(s[:-1])
        aucs[k] = auc

    IO.savePickle('all_data', dict(aucs = aucs, data = data, settings = settings))
