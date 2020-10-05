import numpy as np
from scipy import ndimage
def find_peaks(m,
               n_samples,
               buffer_size,
               surround,
               bins = np.linspace(0, .5, 10),
               sigma = 500,\
               target = None,
               ):
    """
    Will attempt to find peaks for the Potts model with states 0, 1
    #TODO: make this more general function
    """
    from pyprind import prog_bar as pb
    # setup output dicts
    dist = {'tipping': {}}
    for b in bins:
        dist[b] = {}

    # setup state
    for ni in pb(range(n_samples)):
        # reset state
        m.states = m.agentStates[0]
        # simulate filter and get tipping values
        sim = m.simulate(buffer_size)
        if target == None:
            filtered = ndimage.gaussian_filter(sim.mean(1), sigma = sigma)
            scores = np.abs(np.gradient(np.sign(filtered * 2 - 1)))
            idx = np.where(scores)[0]
        else:
            scores = np.abs(np.gradient(np.sign(sim[:, target] * 2 - 1)))
            idx = np.where(scores)[0]

        # for every tipping point...
        for number, i in enumerate(idx):
            # convert for key
            state = tuple(sim[i])
            # get tippign value
            tipp  = np.mean(state)
            # setup the target
            dist["tipping"][state] = dist["tipping"].get(state, 0 ) + 1

            # get distance around window
            # bin window as distance to window
            mag = sim[i - surround : i].mean(1)
            distance = abs(mag - tipp)
            jdx = np.digitize(distance, bins[:-1])
            for idx, s in enumerate(sim[i - surround:i, :]):
                b = bins[jdx[idx]]
                # sample only on one side of the tipping point
                if s.mean() > .5:
                    s = s.astype(int)^1
                s = tuple(s)
                dist[b][s]= dist[b].get(s, 0) + 1
    # normalize counts
    for case, num in dist.items():
        z = sum(num.values())
        for k, v in num.items():
            num[k] = v/z
    return dist
