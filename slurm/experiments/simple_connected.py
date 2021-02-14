from slurm.Task import Task
from plexsim import models
from datetime import datetime
from imi import infcy

from scipy import optimize
import numpy as np, itertools, copy

# helper function for phase transition
def sig(x, a, b, c, d):
        return a / (1 + b * np.exp(c * (x - d)))

from imi.utils.graph import ConnectedSimpleGraphs as CSG
def setup(config : dict):
   # define type of model + settings
    model_t = models.Potts
    model_settings = dict(
       agentStates = np.arange(2)
    )


    # output directory
    output_directory = f"{__file__}:{datetime.now().isoformat()}"
    # magnetization params
    n_magnetize = int(1e5)
    temps = np.linspace(0, 10, 40);

    # fitting params
    maxfev = 100_000


    thetas = [.7] # match_temperatures

    # TODO : implement
    gen = CSG()
    # graphs = [gen.generate(5)]
    graphs = [j for i in gen.generate(5).values() for j in i]

    # memoize phase transition results
    phase_transitions = {}

    # hold tasks
    experiments = []

    combinations = itertools.product(thetas, graphs)
    for comb in combinations:
       # unpack

       instance_settings = model_settings.copy()
       theta, graph = comb

       instance_settings['graph'] = graph
       instance_settings['sampleSize'] = graph.number_of_nodes() 

        # get params fit
       if phase_transition := phase_transitions.get(graph):
           opts, cov = phase_transition.get('fit')
       else:
            # create model
            model = model_t(**instance_settings)
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
       temperature = res.x
       instance_settings['temperature'] = temperature
   
       # reinit model
       model = model_t(**instance_settings)

       
       settings = dict(model = model,
                       instance_settings = instance_settings, 
                       t = temperature,
                       config = config,
                magnetisation = theta)

       task = Experiment(settings, output_directory = output_directory)
       experiments.append(task)
    return experiments

class Experiment(Task):
   def __init__(self, settings, *args, **kwargs): 
      super(Experiment, self).__init__(settings, output_directory = "connected_simple")
      # do stuff

   def run(self):
       model = self.settings.get('model')

       config = self.settings.get('config').copy()
       print(config)
       snapshot_settings = config.get("snapshots").copy()
       conditional_settings = config.get("conditional").copy()
       # setup simulatator object
       sim = infcy.Simulator(model)

       # obtain system snapshots
       snapshots = sim.snapshots(**snapshot_settings)

       same_side = True
       if same_side:
            tmp = dict()
            for k, v in snapshots.items():
                if np.mean(v) > .5:
                    k = np.abs(np.array(k)  - 1)
                    k = tuple(k)
                tmp[k] = tmp.get(k, 0) + v
            z = sum(tmp.values())
            tmp = {k : v / z for k, v in tmp.items()}
            snapshots = tmp
       backup = copy.deepcopy(snapshots)

       # conditional sampling
       time = conditional_settings.get('time')
       time = np.arange(time)
       
       conditional_settings['time'] = time
       output  = sim.forward(snapshots, **conditional_settings)

       snapshots = output['snapshots']
       conditional = output['conditional']
       px, mi = infcy.mutualInformation(conditional, snapshots)


       # store results
       results = dict(snapshots = snapshots,
                      conditional = conditional, 
                      backup_snapshots = backup,
                      graph = model.graph,
                      config = config,
                      mi = mi,
                      px = px)

       return results
   def gen_id(self):
        N = self.settings.get('model').nNodes
        M = self.settings.get('magnetisation')

        import time
        return f"{time.time()}_nNodes={N}_mag={M}"

      
