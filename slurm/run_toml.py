import toml, pickle, os, subprocess, datetime, importlib
from Task import Worker
"""
loads :settings.toml: and runs simulation that are in configs/.
The idea is to setup an experiment with an 'easy to use' toml file which is then passed to the corresponding
run_file in configs. This takes away the complexity of designing a 'one does all' class for running experiments.

The config files have two functions
:setup_model: which setups the model and
:run_model: which runs the model

N.B. there is currently no standard what the toml should contain..
"""
class sim_settings:
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                v = toml_dotted(v)
            self.__setitem__(k, v)

    def __setitem__(self, key, value):
        self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__.get(k)




class ExperimentManager:
    def __init__(self, settings_file):
        self.reader = sim_settings(settings_file)

        # max jobs on the server
        #
        # assume running on the server
        # TODO deal with local processing of experiments
        self.delegate = False
        if os.uname().nodename != "g14":
            self.max_jobs = 5
            self.delegate = True

    def run(self):
        date = datetime.datetime.now().isoformat()
        os.makedirs(base, exists_ok = True)

        for idx, experiment in enumerate(self.reader.experiments):
            # run experiment
            self.run_experiment(experiment, idx)

            
    def run_experiment(self, experiment : str, idx : int) -> None:
        if run_file := experiments.get(experiment):
            # create tasks
            tasks = run_file.setup()

            # extract deadline in case it overflows server deadline
            # convert time to int
            hours, minutes, seconds = [int(i) for i in reader.deadline.split(":")]
            deadline = (date + datetime.timedelta(hours = hours, minutes = minutes,
                                                seconds = seconds)).toordinal()

            worker = Worker(tasks, id = id, deadline = deadline)

            fp = f"worker_{worker.id}.pickle"
            with open(fp, "wb") as f:
                pickle.dump(worker, fp)
            if self.delegate:
                subprocess.call(f"sbatch run_worker.sh worker_{worker.id}".split())

                # release only when jobs can be queued
                while self.get_jobs() < self.max_jobs:
                    time.sleep(1)

        else:
            print(f"!Warning! {experiment} not found")

    def get_jobs(self):
       # count output squeue
       call = "squeue -u {echo $USER} -h -t pending,running -r | wc -l"
       jobs = subprocess.check_output(call, shell = True)
       return int(jobs)
       
import experiments # contains the experiments
SETTINGS = "settings.toml"
if __name__ == "__main__":
    manager = ExperimentManager(SETTINGS)
    manager.run()

    
         

