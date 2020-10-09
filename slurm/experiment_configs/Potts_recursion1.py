from scipy import optimize
import numpy as np, copy

import sys; sys.path.insert(0, '../../')
from plexsim.models import *

from Toolbox import infcy
# TODO: write general setup step for model
#
MODEL = Ising

def setup_model(model) -> list :
    run_settings = [] # list of dicts


    # setup settings
    temps = np.linspace(0, 10, 100); #temps[-1] = 10000
    out   = model.magnetize(temps, n = int(1e5))

    from scipy import optimize
    def sig(x, a, b, c, d):
        return a / (1 + b * np.exp(c * (x - d)))
    opts, cov = optimize.curve_fit(sig, xdata = temps, ydata = out[0],\
                      maxfev = 100_000)

    thetas = [.9, .5, .1] # match_temperatures

    # bounds = optimize.Bounds(0, np.inf)
    for theta in thetas:
        res = optimize.minimize(lambda x: abs(sig(x, *opts) - theta), \
                                x0 = .1,\
                                method = 'TNC',\
                                )
        model.t = res.x
        n = copy.deepcopy(model)
        settings = dict(model = n, t = n.t, mag = theta)
        run_settings.append(settings)
    return run_settings

def run_experiment(model, settings = {}) -> dict:
    from Utils.signal import find_peaks
    peak_settings =  settings['tipping']

    # find peaks
    #
    bins = np.linspace(0.01, 1, 20)
    bins = np.asarray([*-bins[::-1], *bins])
    snapshots = find_peaks(model,
                           bins = bins,
                           **peak_settings)
    sim = infcy.Simulator(model)

    conditional = settings.get("conditional")
    mis = {}
    pxs = {}
    cs  = {}
    for k, v in snapshots.items():
        try:
            s, c = sim.forward(v, **conditional).values()
            px, mi = infcy.mutualInformation(c, s)
            mis[k] = mi
            cs[k]  = c
            pxs[k] = px
        # no states found
        except Exception as e :
            print(e)
    print("-" * 13 + "Tipping points" + "-" * 13)
    for k, v in snapshots.items():
        print(f"{len(v)} \t at {k}")
    # print(f"Found {len(snapshots)}")
    return dict(snapshots = snapshots, mi = mis, px = pxs)
