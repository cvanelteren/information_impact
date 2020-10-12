import numpy as np
from scipy import ndimage
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
