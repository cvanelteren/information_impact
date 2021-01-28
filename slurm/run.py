import toml, pickle, os, subprocess, datetime, importlib, re, time
from Task import Worker
import experiments # contains the experiments


# TODO  general settings file?
class sim_settings:
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                v = sim_settings(v)
            print(k)
            self.__setitem__(k, v)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__.get(key)

# this class setups the the experiments and delegates it to workers
# TODO add unique identifier to the manager class
class ExperimentManager:
    def __init__(self, settings_file):
        try:
            self.reader = toml.load(settings_file)
        except Exception as e:
            print(e)
            exit()

        # max jobs on the server
        # assume running on the server
        # TODO deal with local processing of experiments
        self.delegate = False
        # TODO: REMOVE THIS assume server
        if os.uname().nodename != "g14":
            self.max_jobs = 5
            self.delegate = True

        # counter for reset to prevent overwriting workers
        # unique id base on time ensures uniqueness
        self.pid = time.time()

        self.deadline = self.set_deadline()
        self.threshold = .8

        # create output folder
        date = datetime.datetime.now().isoformat()

        # group simulations in the same folder
        base = f"data/{__file__}-{date}"
        os.makedirs(base, exist_ok = True)
        self.base = base

        self.name = f"{self.__class__.__name__}-{self.pid}"

        self.workers = []

        self.setup_workers()

    def set_deadline(self):
        current_time = time.time()
        timeout_server = 60 * 15 # //minutes
        self.deadline = current_time + timeout_server

    def clean_up(self):
        # if resumed cleanup file
        if os.path.exists(self.name + ".pickle"):
            os.remove(fp)

    def run(self):
        # cleanup if resumed
        self.clean_up()
        self._running = True
        self.run_workers()
        # keep running
        experiments = list(self.reader.get("experiments").items())
        
            
    def override_config(self, config: dict) -> dict:
        overriden = self.reader.get("general", {}).copy()
        for k, v in config.items():
            overriden[k] = v
        return overriden


    def setup_workers(self, experiments):
        while experiments and self._running:
            experiment = experiments.pop(0)
            worker = self.setup_experiment(experiment, self.pid)
            self.workers.append(worker)
            self.check_deadline()

    def run_worker(self, worker):
        if self.delegate:
            self.send_worker(worker)
        else:
            worker.run()

    def run_workers(self):
        for worker in self.workers:
            self.run_worker(worker)
        
    def create_worker(self, tasks):
        # extract deadline in case it overflows server deadline
        # convert time to int
        date = datetime.datetime.now()
        deadline = self.reader.get('sbatch', {} ).get("deadline", None)
        hours, minutes, seconds = [int(i) for i in deadline.split(":")]
        deadline = (date + datetime.timedelta(hours = hours,
                                              minutes = minutes,
                                              seconds = seconds)).toordinal()

        worker = Worker(tasks, id = f"{self.pid}-{idx}",
                        deadline = deadline, base = self.base)
        return worker

    def send_worker(self, worker):
        fp = f"{worker.name}.pickle"
        with open(fp, "wb") as f:
                pickle.dump(worker, fp)
        subprocess.call(f"sbatch run_worker.sh {worker.name}".split())

    def setup_experiment(self, experiment: tuple, idx : int) -> None:
        name, config = experiment
        if run_file := experiments.get(name):
            # create tasks
            config = self.override_config(config)
            tasks = run_file.setup(config) 
            worker = self.create_worker(tasks)
        else:
            print(f"!Warning! {experiment} not found")

    def dump_to_disk(self):
        fp = f"{self.name}.pickle" 
        with open(fp, "wb") as f:
            pickle.dump(self)


    def exit(self):
        self._running = False
        subprocess.call("python {__file__}")

    # TODO: write test case dump file to disk and call itself
    def restart(self):
        self.dump_to_disk()
        #update deadline
        self.set_deadline()
        self.exit()
        

    def check_deadline(self):
        # only use when on the server
        while self.get_jobs() < self.max_jobs:
            time.sleep(1)
            # restart process
            if time.time() >= self.deadline * self.threshold:
                self.restart()
                
                
    def get_jobs(self):
       # count output squeue
       call = "squeue -u {echo $USER} -h -t pending,running -r | wc -l"
       jobs = subprocess.check_output(call, shell = True) # return string
       return int(jobs)
       
SETTINGS = "settings.toml"
if __name__ == "__main__":

    # check if manager exists
    pattern = f"{ExperimentManager.__name__}-d+.pickle"
    # load it
    for file in os.listdir():
        # found file
        if re.match(pattern, file):
            with open(file, 'rb') as f:
                manager = pickle.load(f)
            break
    # create one and run it
    else:
        manager = ExperimentManager(SETTINGS)
    manager.run()

    
         

