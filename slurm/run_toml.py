from plexsim.models import *
import toml, pickle, os
import sys; sys.path.insert(0, '../')
from Utils.graph import *
from Toolbox import infcy
import importlib

class toml_reader:
    def __init__(self, fn):
        self.settings = toml.load(fn)
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
        name = model_settings.get('name', '')
        settings = model_settings.get('settings', {})
        
        g_name = model_settings.get('graph').get('name')
        g_settings = model_settings.get('graph').get('settings', {})
        g = globals()[g_name](**g_settings)
        m = globals()[name](g, **settings)
        return m




import datetime
# TODO: iterate over those
if __name__ == "__main__":

    # load toml settings
    reader = toml_reader('settings.toml')
    # make local directory if exists
    date = datetime.datetime.now().isoformat()

    trials = reader.settings.get("trial_runs", 1)
    run_name = reader.settings.get('name', 'experiment') + date

    output_directory = reader.settings.get("output_directory") + "_" + run_name

    os.makedirs(output_directory, exist_ok = True)
    # run experiments
    from pyprind import prog_bar
    for trial in prog_bar(range(trials)):
        for idx, experiment in enumerate(reader.experiment_run("setup_model", dict(model = reader.load_model()))):
            # run the actual experiment
            assert experiment['model'].sampleSize == 1
            fn       = f'{run_name}-{idx}-{trial}.pickle'
            settings = dict(
                model    = experiment["model"],
                settings = reader.settings.get("simulation", {})
            )
            results  = reader.experiment_run("run_experiment", settings)


            #write data
            path = os.path.join(output_directory, fn)
            with open(path, "wb") as f:
                o = dict(results = results,
                         settings = experiment)
                pickle.dump(o, f)
    print('exited')
