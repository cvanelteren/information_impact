
from scipy import optimize
import numpy as np, copy

from imi.utils import signal
from plexsim.models import *
from imi.signal import find_tipping
from imi.utils.stats import check_allocation, resample

from imi import infcy
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

    thetas = [0.3] # match_temperatures

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
    peak_settings =  settings['tipping']

    # find peaks
    #
    # bins = np.linspace(0, 1, 20)
    bins = np.linspace(-1, 1, 21)
    # bins = np.asarray([*-bins[::-1], *bins])

    # hacked in parameter settings
    peak_settings['rtol'] = 2/model.nNodes
    peak_settings['sigma'] = model.nNodes * 10

    # snapshots here is bins, states
    snapshots, isi = find_tipping(model,
                           bins = bins,
                           **peak_settings)
    sim = infcy.Simulator(model)

    conditional = settings.get("conditional")
    mis = {}
    pxs = {}
    cs  = {}
    nStates = sum([len(i) for i in snapshots.values()])
    if nStates == 0:
        nStates = 1
    t = conditional.get("time_steps")

    # check for allocation of results
    bits = t * model.nNodes * nStates * np.float64().itemsize
    bits =  check_allocation(bits)
    if bits < 1:
        bins = np.array([i for i in snapshots.keys()])
        counts = np.asarray([len(i) for i in snapshots.values()])
        n = int(bits * counts.sum())
        print(f"resampling to {n}")
        resampled = resample(counts, bins, snapshots, n)
    else:
        resampled = snapshots



    time_steps = conditional.get("time_steps", 1000)
    time       = np.arange(0, time_steps // 2)
    try:
        del conditional['time_step']
        logtime    = np.geomspace(min(isi), max(isi), time_steps // 2)
        time = np.concatenate((time, logtime))
    except KeyError:
        print("key not found")

    conditional['time'] = time
    for k, v in resampled.items():
        try:
            s, c = sim.forward(v, **conditional).values()
            c = { k: np.float32(vv) for k, vv in c.items() }
            px, mi = infcy.mutualInformation(c, s)
            mis[k] = np.float32(mi)
            cs[k]  = c
            pxs[k] = px
        # no states found
        except Exception as e :
            print(e)
    print("-" * 13 + "Tipping points" + "-" * 13)
    for k, v in snapshots.items():
        print(f"{len(v)} \t at {k}")
    # print(f"Found {len(snapshots)}")
    print("done")
    return dict(snapshots = snapshots, mi = mis, px = pxs, resampled =
                resampled, isi = isi, time = time, cpx = cs)
