from imi.utils.graph import ConnectedSimpleGraphs as CSG
from imi.utils.rules import create_rule_full
from slurm.Task import Task
from plexsim import models
from datetime import datetime
from imi import infcy

from scipy import optimize
import numpy as np
import itertools
import copy
import networkx as nx
from imi.utils.graph import recursive_tree

# helper function for phase transition


def sig(x, a, b, c, d):
    return a / (1 + b * np.exp(c * (x - d)))


def setup(config: dict):
   # define type of model + settings
    model_t = models.ValueNetwork
    model_settings = dict(
        agentStates=np.arange(2),
    )
    # output directory
    output_directory = f"{__file__}:{datetime.now().isoformat()}"
    # magnetization params
    n_magnetize = int(1e3)
    temps = np.linspace(0, 10, 40)

    # fitting param10, 10s
    maxfev = 100_000

    thetas = [.5]  # match_temperatures

    # TODO : implement
    gen = CSG()
    #graphs = [gen.generate(5)]
    graphs = [nx.grid_graph((20, 20), periodic=1)]
    rules = [create_rule_full(j).copy()
             for i in gen.generate(5).values() for j in i]

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
        instance_settings['bounded_rational'] = 2

        for bounded_rational in range(1, 3):
            instance_settings['bounded_rational'] = bounded_rational

            model = model_t(**instance_settings)

            settings = dict(model=model.__deepcopy__(),
                            instance_settings=instance_settings.copy(),
                            #t = temperature,
                            config=config.copy(),
                            magnetisation=theta)
            task = Experiment(settings.copy(),
                              output_directory=output_directory)
            experiments.append(task)

    return experiments


class Experiment(Task):
    def __init__(self, settings, *args, **kwargs):
        super(Experiment, self).__init__(
            settings, output_directory="connected_simple")
        # do stuff

    def run(self):
        model = self.settings.get('model')
        theta = .75
        temps = np.linspace(0, 5, 20)
        res = model.magnetize(temps, n=1e3)
        # fit phase transition
        maxfev = int(1e4)
        opts, cov = optimize.curve_fit(sig, xdata=temps,
                                       ydata=res[0],
                                       maxfev=maxfev)

        res_opt = optimize.minimize(lambda x: abs(sig(x, *opts) - theta),
                                x0=.1,
                                method='TNC',
                                )

        model.t = res_opt.x
        model.reset()
        time_res = model.simulate(1e3)
        val_res = np.array([model.check_vn(i) for i in time_res])
        # store results
        results = dict(
            results=res,
            res_opt = res_opt,
            val_res=val_res,
            time_res=time_res,
            #satisfaction = satisfaction,
            graph=model.graph,
            config=self.settings.get("instance_settings"),
            temps=temps
        )

        return results

    def gen_id(self):
        N = self.settings.get('model').nNodes
        M = self.settings.get('magnetisation')

        import time
        return f"{time.time()}_nNodes={N}_mag={M}"
