from slurm.Task import Task
from plexsim import models
from datetime import datetime
from imi import infcy

from scipy import optimize
import numpy as np, itertools, copy, networkx as nx
from imi.utils.graph import recursive_tree

# helper function for phase transition
def sig(x, a, b, c, d):
        return a / (1 + b * np.exp(c * (x - d)))

from imi.utils.graph import ConnectedSimpleGraphs as CSG
from imi.utils.rules import create_rule_full
def setup(config : dict):
   # define type of model + settings
    model_t = models.Potts
    model_settings = dict(
       agentStates = np.arange(2)
    )
    # output directory
    output_directory = f"{__file__}:{datetime.now().isoformat()}"
    # magnetization params
    n_magnetize = int(1e3)
    temps = np.linspace(0, 3, 40);

    # fitting params
    maxfev = 100_000


    thetas = [.7] # match_temperatures

    # TODO : implement
    gen = CSG()
    #graphs = [gen.generate(5)]
    graphs = [nx.grid_graph((10, 10))]
    rules = [create_rule_full(j).copy() for i in gen.generate(5).values() for j in i]

    # memoize phase transition results
    phase_transitions = {}

    # hold tasks
    experiments = []

    combinations = itertools.product(thetas, graphs, rules)
    for comb in combinations:
       # unpack
       instance_settings = model_settings.copy()
       theta, graph, rule = comb

       instance_settings['graph'] = graph
       instance_settings['rules'] = rule.copy()
       instance_settings['sampleSize'] = graph.number_of_nodes() 
       instance_settings['agentStates'] = np.arange(len(rule))

       # get params fit
       if phase_transition := phase_transitions.get((graph, rule)):
           opts, cov = phase_transition.get('fit')
       else:
            # create model
            model = model_t(**instance_settings)
            magnetization, susceptibility   = model.magnetize(temps, n = n_magnetize)

            # fit phase transition
            opts, cov = optimize.curve_fit(sig, xdata = temps,
                                           ydata = magnetization,
                                           maxfev = maxfev, bounds = (0, np.inf))
      
       res = optimize.minimize(lambda x: abs(sig(x, *opts) - theta), \
                                x0 = .1,\
                                method = 'COBYLA',\
                                )
        # setup model
       temperature = res.x
       if temperature < 0:
           assert 0

       instance_settings['temperature'] = temperature
   
       # reinit model
       model = model_t(**instance_settings)

       
       settings = dict(model = model,
                       instance_settings = instance_settings, 
                       t = temperature,
                       config = config,
                magnetisation = theta)

       task = Experiment(settings.copy(),
                         output_directory = output_directory)
       experiments.append(task)
    return experiments

class Experiment(Task):
   def __init__(self, settings, *args, **kwargs): 
      super(Experiment, self).__init__(settings, output_directory = "connected_simple")
      # do stuff

   def run(self):
       model = self.settings.get('model')

       N = 100000 
       res =  model.simulate(N)

       satisfaction = np.zeros((N, model.nNodes))
       for idx, r in enumerate(res):
           satisfaction[idx, :] = np.asarray(model.siteEnergy(r)).flatten()
    
       # store results
       results = dict(results = res,
                      satisfaction = satisfaction,
                      graph = model.graph,
                      config = self.settings.copy()
                      )

       return results
   def gen_id(self):
        N = self.settings.get('model').nNodes
        M = self.settings.get('magnetisation')

        import time
        return f"{time.time()}_nNodes={N}_mag={M}"

      
