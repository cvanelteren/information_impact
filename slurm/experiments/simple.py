from slurm.Task import Task
from plexsim import models
from datetime import datetime
from imi import infcy

import numpy as np, itertools

# helper function for phase transition
def sig(x, a, b, c, d):
        return a / (1 + b * np.exp(c * (x - d)))

def setup():
   # define type of model + settings
    model_t = Potts
    model_settings = dict(
       agentStates = np.arange(2)
    )

    # output directory
    output_directory = f"{__file__}:{datetime.now().isoformat()}"
    # magnetization params
    n_magnetize = int(1e5)
    temps = np.linspace(0, 10, 100);

    # fitting params
    maxfev = 100_000

    from scipy import optimize

    thetas = [.5] # match_temperatures

    # TODO : implement
    graphs = []

    # memoize phase transition results
    phase_transition = {}

    # hold tasks
    experiments = []

    combinations = itertools.product(thetas, graphs)
    for comb in combinations:
       # unpack
       theta, graph = comb
        # get params fit
       if phase_transition := phase_transitions.get(graph):
           opts, cov = phase_transition.get('fit')
       else:
            # create model
            model = model_t(**model_settings)
            magnetization, susceptibility   = model.magnetize(temps, n = n_magnetize)

            # fit phase transition
            opts, cov = optimize.curve_fit(sig, xdata = temps,
                                           ydata = magnetization,
                                           maxfev = maxfev)

            # store function fit and output
            phase_transitions[graph] = dict(fit = (opts, cov),
                                            magnetization  = magnetization,
                                            susceptibility = susceptibility
                                            )
      
       res = optimize.minimize(lambda x: abs(sig(x, *opts) - theta), \
                                x0 = .1,\
                                method = 'TNC',\
                                )
        # setup model
       model_settings['graph'] = graph
       model_settings['temperature'] = res.x
   
       # reinit model
       model = model_t(**model_settings)

       settings = dict(model = model,
                        t = temperature,
                        magnetisation = theta)

       task = Experiment(settings, output_directory = output_directory)
       experiments.append(task)
    return experiments

class Experiment(Task):
   def __init__(self, settings, *args, **kwargs): 
      super(self).__init__(self, settings)
      # do stuff

   def run(self):
       # setup simulatator object
       sim = infcy.Simulator(model)

       # obtain system snapshots
       snapshots = sim.snapshots(**snapshots_settings)

       # conditional sampling
       conditional = sim.forward(snapshots, **conditional_settings)

       # store results
       results = dict(snapshots = snapshots, conditional = conditional)

       return results

      
