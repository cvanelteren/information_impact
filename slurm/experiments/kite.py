from slurm.Task import Task
from plexsim import models
from datetime import datetime
from imi import infcy

from scipy import optimize
import numpy as np, itertools, copy, networkx as nx

# helper function for phase transition
def sig(x, a, b, c, d):
        return a / (1 + b * np.exp(c * (x - d)))

def setup(config : dict):
   # define type of model + settings
    model_t = models.Potts
    model_settings = dict(
       agentStates = np.arange(2)
    )


    # output directory
    output_directory = f"{__file__}:{datetime.now().isoformat()}"
    # magnetization params
    n_magnetize = int(1e4)
    temps = np.linspace(0, 5, 50);

    # fitting params
    maxfev = 100_000


    thetas = [.9] # match_temperatures


    # TODO : implement
    # graphs = [gen.generate(5)]
    graphs = [nx.krackhardt_kite_graph()]
    # graphs = [nx.florentine_families_graph()]
    # graphs = [nx.star_graph(5)]
    # graphs = [nx.path_graph(5)]
    # 

    # memoize phase transition results
    phase_transitions = {}

    # hold tasks
    experiments = []

    impacts = np.array([-1, -np.inf])
    interventions = []
    for g in graphs:
        # unperturbed
        interventions.append({})
        for node in g.nodes():
            for impact in impacts:
                tmp = {node : impact}
                # nodal intervention
                interventions.append(tmp)

    print(f"int: {len(interventions)},g: {len(graphs)},theta: {len(thetas)}")
    combinations = itertools.product(thetas, graphs, interventions)
    
    snaps = {}
    for comb in combinations:
       # unpack

       instance_settings = model_settings.copy()
       theta, graph, intervention = comb
       instance_settings['sampleSize'] = 1 #len(graph)
       instance_settings['graph'] = graph
       # instance_settings['sampleSize'] = graph.number_of_nodes() 

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
                                x0 = 0,\
                                method = 'COBYLA',\
                                )

        # setup model
       t = res.x
       print(f"setting temperature to {t}")
       instance_settings['t'] = t
   
       # reinit model
       model = model_t(**instance_settings)

       # s = snaps.get(graph, {})
       # if len(s) == 0:
       #      sim = infcy.Simulator(model)
       #      ss = config['snapshots']
       #      s = sim.snapshots(**ss)

       #      print(s)
       #      snaps[graph] = s
       #      print(f"snaps:{len(s)}")
       config['intervention'] = intervention
       import copy
       settings = dict(
                       model             = copy.deepcopy(model),
                       instance_settings = instance_settings, 
                       t                 = t,
                       config            = config.copy(),
                       magnetisation     = theta,
                       # snapshots         = s
                       )

       task = Experiment(settings, output_directory = output_directory)
       experiments.append(task)
    return experiments

class Experiment(Task):
   def __init__(self, settings, output_directory, *args, **kwargs): 
      super(Experiment, self).__init__(settings, output_directory)
      # do stuff

   def run(self):
       model = self.settings.get('model')


       # print("here", model.sampleSize)

       config = self.settings.get('config').copy()


       intervention = config.get('intervention', {})
       snapshot_settings = config.get("snapshots").copy()

       conditional_settings = config.get("conditional").copy()
       # setup simulatator object
       sim = infcy.Simulator(model)

       # obtain system snapshots
       snapshots = sim.snapshots(**snapshot_settings)
       # snapshots = self.settings['snapshots']


       same_side = True
       if same_side:
            tmp = dict()
            mapper = {i: j for i, j in zip(model.agentStates, model.agentStates[::-1])}
            # print(mapper)
            for k, v in snapshots.items():
                if np.mean(k) < np.mean(model.agentStates):
                    k = tuple(mapper[ki] for ki in k)
                tmp[k] = tmp.get(k, 0) + v
                # print(k, v)
            z = sum(tmp.values())
            tmp = {k : v / z for k, v in tmp.items()}
            snapshots = tmp
       backup = copy.deepcopy(snapshots)

       # conditional sampling
       time = conditional_settings.get('time')
       time = np.arange(time)
       
       conditional_settings['time'] = time

       
       model.nudges = intervention
       print(model.nudges)

       # # model.nudges = intervention
       # for k, v in intervention.items():
       #      idx = model.adj.mapping[str(k)]
       #      model.H[idx] = v
       # print(model.H)

       sim = infcy.Simulator(model)
       output  = sim.forward(snapshots, **conditional_settings)

       # snapshots = output['snapshots']
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

      
