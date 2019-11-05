import numpy as np, os, re
import itertools, scipy, multiprocessing as mp
from tqdm import tqdm
from Utils import plotting as plotz
ROOT = '/var/scratch/cveltere/2019-10-31T14:33:34.823116/'
ROOT = 'Data/tester'
def worker(sample):

    sidx, sample, func = sample
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
    return (sidx, auc[:, 0]), coeffs

from Utils import stats, IO

def loadSettings(root):
    settings = {}
    for (r, dirs, files) in os.walk(root):
        for file in files:
            if 'settings' in  file:
                path = os.path.join(r, file)
                settings[r] = IO.loadPickle(path)
    return settings

def norm(x):
    s = x.shape
    x = x.reshape(-1)
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    return x.reshape(s)


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

    # work around for the inf index
    pulse = pulse.replace('inf', 'None')

    # literal_eval will crash with inf
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

            model   = setting.get('model')
            try:
                nodeidx = model.mapping[node]
            except Exception as e:
                # print(e)
                g     = setting.get('graph')
                model = model(g)
                nodeidx = model.mapping[node]

            # nodeidx = int(node)
            pulseSize = intervention[node]
            if pulseSize == None:
                pulseSize = np.inf

            pulseidx = tmpPulses[pulseSize]
            # assert pulseidx != 0
            # load corresponding control
            controlFile = '_'.join(i for i in [trial, mag, '{}'])
            path = os.path.join(*fileName.split('/')[:-1])
            if fileName.startswith('/'):
                 path = '/' + path 
            # print('>', path)
            # print('>>', controlFile)
            cpx = IO.loadPickle(os.path.join(path, controlFile)).px

            simData = np.nansum(stats.KL2(lData.px, cpx), axis = -1)[deltas // 2 + 1:]
            # print(simData)
        # missing data
        except Exception as e:
            print('>', e)
            simData = np.NaN
    s = (nodeidx, trialidx, pulseidx, tempidx)
    return (fileName, s, simData)

def func(x, a, b, c, d, e, f):
    return a * np.exp(-b * ( x - c ) ) + d * np.exp(-e * ( x - f )) 
p0          = np.ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(\
                       maxfev = int(1e6), \
                       bounds = (0, np.inf), p0 = p0,\
                       jac = 'cs', \
                      )
LIMIT = np.inf



def main():
    # setup files to read
    settings = {}
    fileNames = []

    print("Finding files and settings")
    for (root, dirs, files) in os.walk(ROOT):
        for file in files:
            if file.endswith('.pickle') and 'setting' not in file:
                settingRoot = '/'.join(i for i in root.split('/')[:-1])
                setting = settings.get(settingRoot, {})
                if not setting:
                    print(settingRoot)
                    path = os.path.join(settingRoot, 'settings.pickle')
                    setting = IO.loadPickle(path)
                    m = setting.get('model')
                    g = setting.get('graph')
                    try:
                        m.graph
                    except:
                        m = m(g)
                    setting['model'] = m
                
                settings[settingRoot] = setting
                tmp = (os.path.join(root, file), setting)
                # print('>>', tmp[0])
                fileNames.append(tmp)
    # setup data vector
    data = {}
    print("setting up data")
    for root, setting in settings.items():
        eq = setting.get('equilibrium')
        nTemps = len(eq.get('ratios'))
        nTrials = setting.get('nTrials')
        nPulse = len(setting.get('pulseSizes'))
        nPulse = nPulse + 1 if 0 not in setting.get('pulseSizes') else nPulse

        deltas = setting.get('deltas')
        nNodes = setting.get('model').nNodes
        s = (nNodes, nTrials, nPulse, nTemps,\
                    deltas // 2 - 1)
        data[root] = np.zeros(s, dtype = float)
    print("Loading data") 
    pbar = tqdm(total = len(fileNames))
    with mp.Pool(mp.cpu_count()) as p:
        for (path, s, d) in p.imap(loadDataFilesSingle, fileNames):
            fn = '/'.join(i for i in path.split('/')[:-2])
            if np.all(np.isnan(d)):
                print(path)
                assert False
            data[fn][s] = d
            pbar.update(1)
    pbar.close()

   # normalize data
    rdata = {}
    l = list(data.keys())
    print("Normalizing")
    for key in l:
        v = data[key]
        nodes, trials, pulses, temps, deltas = v.shape
        # print(key)
        for (i, j, k) in itertools.product(*[range(i) for i in v.shape[1:-1]]):
            tmp = norm(v[:, i, j,k])
            if np.all(np.isnan(tmp)):
                print("Nan found at", i, j, k)
            v[:, i, j, k] = tmp
        rdata[key] = v 
    aucs = {}
    coeffs = {}
    print("Computing area under the curve")
    for k, v in rdata.items():
        setting = settings[k]
        
        s = v.shape
        v = v.reshape(-1, s[-1])
        
        with mp.Pool(mp.cpu_count()) as p:
            auc = np.zeros(len(v))
            v = [(idx, i, func) for idx, i in enumerate(v)]
            pbar = tqdm(total = len(v)) 
            
            tmp = []
            for (idx, i), coeff in p.imap( worker, v):
                auc[idx] = i
                tmp.append(coeff)
                pbar.update(1)
        tmp = np.array(tmp).squeeze()
        
        auc = auc.reshape(s[:-1])
        aucs[k] = auc
        coeffs[k] =  tmp.reshape(*s[:-1], tmp.shape[-1])

    IO.savePickle('tester.pickle', dict(aucs = aucs, data = data, \
                                              rata = rdata, settings = settings,\
                                   coeffs = coeffs))
if __name__ == "__main__":
    main()
