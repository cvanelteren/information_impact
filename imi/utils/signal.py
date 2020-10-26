import numpy as np, multiprocessing as mp, os
from pyprind import prog_bar
from scipy import ndimage

def _find_peaks(dummy, settings):
    # reset state
    import copy
    m           = settings.get('m')
    # print(f"{os.getpid()} \t {id(m)}\n")
    buffer_size = settings.get('buffer_size')
    sigma       = settings.get('sigma')
    surround    = settings.get('surround')
    start       = settings.get('start')
    target      = settings.get('target')
    bins        = settings.get("bins")

    dist    =  {}
    dist[0] = {}
    for b in bins:
        dist[b] = {}

    if settings.get('reset'):
        m.states = m.agentStates[0]
    # simulate filter and get tipping values
    sim = m.simulate(buffer_size)
    if target == None:
        filtered = ndimage.gaussian_filter(sim.mean(1), sigma = sigma)
        scores = np.abs(np.gradient(np.sign(filtered * 2 - 1)))
        idx = np.where(scores)[0]
    else:
        s = sim[:, target]
        # remove spurious connections
        s = ndimage.gaussian_filter(s, sigma)
        scores = np.abs(np.gradient(np.sign(s * 2 - 1)))
        idx = np.where(scores)[0]


    # for every tipping point...
    if len(idx):
        if idx.max() + surround > sim.shape[0]:
            sim = np.concatenate((sim, m.simulate(surround)), axis = 0)

    for number, i in enumerate(idx):
        window = np.asarray([*np.arange(i - surround, i - start), *np.arange(i + start, i + surround)])
        # np.delete(window, i)
        # convert for key
        state = tuple(sim[i])
        # get tippign value
        tipp  = np.mean(state)
        # setup the target
        dist[0][state] = dist[0].get(state, 0 ) + 1

        # get distance around window
        # bin window as distance to window
        mag      = sim[window].mean(1)
        distance = mag - tipp
        jdx = np.digitize(distance, bins[:-1], right = 1)
        for idx, s in enumerate(sim[window]):
            b = bins[jdx[idx]]
            # sample only on one side of the tipping point
            # if s.mean() > .5:
                # s = s.astype(int)^1
            s = tuple(s)
            dist[b][s]= dist[b].get(s, 0) + 1
    return dist
def find_peaks(m,
               n_samples,
               buffer_size,
               surround,
               bins = np.linspace(0, .5, 10),
               sigma = 500,\
               target = None,
               start = 1,
               reset = False
               ):
    """
    Will attempt to find peaks for the Potts model with states 0, 1
    #TODO: make this more general function
    """
    from pyprind import prog_bar as pb
    # setup output dicts



    # setup state
    assert surround > start
    start = m.nNodes * start
    surround = surround * m.nNodes
    from itertools import repeat
    settings = dict(
        m           = m,
        buffer_size = buffer_size,
        surround    = surround,
        bins        = bins,
        sigma       = sigma,
        target      = target,
        start       = start,
        reset       = reset
    )
    from functools import partial
    f = partial(_find_peaks, settings = settings)
    with mp.Manager() as man:

        cpus = mp.cpu_count()
        with mp.Pool(cpus) as p:
            output = p.map(f, range(n_samples))

        dist = {}
        for o in output:
            for k, v in o.items():
                dist[k] = dist.get(k, {})
                for kk, vv in v.items():
                    dist[k][kk] = dist[k].get(kk, 0) + vv

    # normalize counts
    for case, num in dist.items():
        z = sum(num.values())
        for k, v in num.items():
            num[k] = v/z
    dist = dict(sorted(dist.items(), key = lambda x : x[0]))
    return dist

def find_peaks_single(m,
               n_samples,
               buffer_size,
               surround,
               bins = np.linspace(0, .5, 10),
               sigma = 500,\
               target = None,
               start = 1,
               reset = False,
               tol = .1,
               spacing = 1,
               ):
    """
    Will attempt to find peaks for the Potts model with states 0, 1
    #TODO: make this more general function
    """
    from pyprind import prog_bar as pb
    # setup output dicts
    dist = {0: {}}
    for b in bins:
        dist[b] = {}

    # setup state
    assert surround > start
    start = m.nNodes * start
    surround = surround * m.nNodes
    for ni in pb(range(n_samples)):
        # reset state
        if reset:
            m.states = m.agentStates[0]
        # simulate filter and get tipping values
        sim = m.simulate(buffer_size)
        if target == None:
            # filtered = ndimage.gaussian_filter(sim.mean(1), sigma = sigma)
            idx = np.where(np.isclose(sim.mean(1), 0.5, atol = 0, rtol = tol))[0]
            idx =  idx[np.argwhere(np.diff(idx) > spacing)]
            # scores = np.abs(np.gradient(np.sign(filtered * 2 - 1)))
            # idx = np.where(scores)[0]
        else:
            s = sim[:, target]
            # remove spurious connections
            s = ndimage.gaussian_filter(s, sigma)
            # scores = np.abs(np.gradient(np.sign(s * 2 - 1)))
            # idx = np.where(scores)[0]
            idx = np.where(np.isclose(s, 0.5, atol = tol, rtol = 0))[0]


        # for every tipping point...
        if len(idx):
            if idx.max() + surround > sim.shape[0]:
                sim = np.concatenate((sim, m.simulate(surround)), axis = 0)

        for number, i in enumerate(idx):
            window = np.asarray([*np.arange(i - surround, i - start), *np.arange(i + start, i + surround)])
            # np.delete(window, i)
            # convert for key
            state = tuple(sim[i])
            # get tippign value
            tipp  = np.mean(state)
            # setup the target
            dist[0][state] = dist[0].get(state, 0 ) + 1

            # get distance around window
            # bin window as distance to window
            mag      = sim[window].mean(1)
            distance = mag - tipp
            jdx = np.digitize(distance, bins[:-1])
            for idx, s in enumerate(sim[window]):
                b = bins[jdx[idx]]
                # sample only on one side of the tipping point
                # if s.mean() > .5:
                    # s = s.astype(int)^1
                s = tuple(s)
                dist[b][s]= dist[b].get(s, 0) + 1
    # normalize counts
    for case, num in dist.items():
        z = sum(num.values())
        for k, v in num.items():
            num[k] = v/z
    dist = dict(sorted(dist.items(), key = lambda x : x[0]))
    return dist
def find_peaks_dumb(m, n_samples,
                   bins):
    dist = {}
    for b in bins:
        dist[b] = {}

    sim = m.simulate(n_samples)
    target = .5
    for idx, i in enumerate(sim):
        d = np.mean(i) - target
        idx = np.digitize(d, bins, right = 1)
        idx = bins[idx]
        i = tuple(i)
        dist[idx][i] = dist[idx].get(i, 0) + 1
    for k, v in dist.items():
        z = sum(v.values())
        for kk, vv in v.items():
            vv /= z
    return dist

def running(m, n_samples, buffer_size = int(1e5),
           window = 100, sigma = 1000,
            tol = .1,
           ):

    snaps = {}
    conds = {}
    for sample in prog_bar(range(n_samples)):
        sim = m.simulate(buffer_size)

#         filtered = ndimage.gaussian_filter(sim.mean(1), sigma = sigma)
#         idx = np.where(np.abs(np.gradient(np.sign(filtered * 2 - 1))))[0]
        idx = np.where(np.isclose(sim.mean(1), 0.5, atol = 0, rtol = .1))[0]
        if len(idx):
            if idx.max() + window > sim.shape[0]:
                sim = np.concatenate((sim, m.simulate(window)))

        for jdx in idx:
            if jdx - window  > 0:
                # get center
                target = tuple(sim[jdx])
                snaps[target] = snaps.get(target, 0) + 1
                # get windowed data
                state = sim[jdx - window : jdx + window]
                binned = np.digitize(state, m.agentStates, right = 1)
                buff   = conds.get(target, np.zeros((state.shape[0], m.nNodes, m.nStates)))
                s    = buff.shape
                buff = buff.reshape(-1, m.nStates)
                for kdx, j in enumerate(binned.flat):
                    buff[kdx, j] += 1
                conds[target] = buff.reshape(s)

    z = sum(snaps.values())
    for k, v in snaps.items():
        conds[k] /= v
        snaps[k] = v / z
    return snaps, conds
