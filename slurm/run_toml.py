from plexsim.models import *
import networkx as nx
import toml, pickle, os
from imi.utils import graph
from imi import infcy
import importlib

"""
loads :settings.toml: and runs simulation that are in configs/.
The idea is to setup an experiment with an 'easy to use' toml file which is then passed to the corresponding
run_file in configs. This takes away the complexity of designing a 'one does all' class for running experiments.

The config files have two functions
:setup_model: which setups the model and
:run_model: which runs the model

N.B. there is currently no standard what the toml should contain..
"""
class toml_reader:
    def __init__(self, fn):
        self.settings = toml.load(fn)
        print(self.settings)
        self.sim_settings = self.settings.get('simulation')

    def experiment_run(self, f, opt_settings = {}):
        """
        calls the run_model method from the experimental script
        """
        fn = self.settings.get("experiment_run").replace("/", ".").replace(".py", '')
        print("-" * 13 + f"{f}" + "-" * 13)
        run = importlib.import_module(fn)
        return getattr(run, f)(**opt_settings)
   
    def load_model(self):
        """
        load the model
        - primarily used to set up the graph structure, the actual model settings need to be defined in setup_model
        """
        model_settings = self.settings.get('model')
        name           = model_settings.get('name', '')
        settings       = model_settings.get('settings', {})
        
        g_name     = model_settings.get('graph').get('name')
        g_settings = model_settings.get('graph').get('settings', {})
        dotted = g_name.split(".")
        # TODO: clean up
        if len(dotted) > 1:
            mod = globals()[dotted[0]]  # assure package networkx is imported as nx
            for comp in dotted[1:]:
                mod = getattr(mod, comp)
            g = mod(**g_settings)
        else:
            g = globals()[g_name](**g_settings)
        m   = globals()[name](g, **settings)

        return m

import datetime
if __name__ == "__main__":

    # load toml settings
    reader = toml_reader('settings.toml')
    # make local directory if exists
    date = datetime.datetime.now().isoformat()

    # get the run script
    run_name = os.path.basename(reader.settings.get("experiment_run", "experiment")) # remove path
    run_name = os.path.splitext(run_name)[0] # remove extensions
    run_name += date

    # create the output directory

    if "fs2" in os.uname().nodename:
        from getpass import getuser
        output_directory = os.path.join("/var/scratch/data", getuser())
    else:
        output_directory = 'data/'

    output_directory += run_name

    os.makedirs(output_directory, exist_ok = True)
    print(f"created {output_directory}")
    # run experiments
    from pyprind import prog_bar
    trials = reader.settings.get("trials", 1)
    for trial in prog_bar(range(trials)):
        for idx, experiment in enumerate(reader.experiment_run("setup_model",
                                    dict(model = reader.load_model()))):
            # run the actual experiment
            fn       = f'{run_name}-{idx}-{trial}.pickle'
            settings = dict(
                model    = experiment["model"],
                settings = reader.settings.get("simulation", {})
            )
            results  = reader.experiment_run("run_experiment", settings)
            print("-" * 13 + "Done" + "-" * 13)
            #write data
            path = os.path.join(output_directory, fn)
            print(f"Saving {path}")
            with open(path, "wb") as f:
                o = dict(results = results,
                         settings = experiment)
                pickle.dump(o, f)
    print('exited')
