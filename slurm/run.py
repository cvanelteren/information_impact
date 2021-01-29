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
        self.reader = toml.load(settings_file)

        # max jobs on the server
        # assume running on the server
        # TODO deal with local processing of experiments
        self.delegate = False
        # TODO: REMOVE THIS assume server
        if os.uname().nodename.endswith("uva.nl"):
            self.max_jobs = 5
            self.delegate = True

        # counter for reset to prevent overwriting workers
        # unique id base on time ensures uniqueness
        self.pid = time.time()

        self.is_running = False # running overwrites
        self.deadline = self.set_deadline()
        self.threshold = .8

        # create output folder
        date = datetime.datetime.now().isoformat()

        # group simulations in the same folder
        base = f"data/{__file__}-{date}"
        os.makedirs(base, exist_ok = True)
        self.base = base

        self.name = f"{self.__class__.__name__}-{self.pid}"

        
        # holds workers name and whether they are done
        self.setup_workers(self.reader.get("experiments"))

    def run(self):
        # cleanup if resumed
        self.clean_up()
        self.is_running = True
        while self.is_running:
            # are all workers done?
            if all([w for w in self.workers.values()]):
                self.is_running = False
            for worker_file, is_done in self.workers.items():
                is_done = self.check_worker(worker_file)
                if not is_done:
                    if self.delegate:
                        self.delegate_worker(worker_file)
                    else:
                        #run it
                        worker = Worker.load_from(worker_file)
                        worker.run()
            self.check_deadline()
        
    def check_worker(self, worker_file):
        """Check whether the worker is working"""
        # check error logs
        
        # assume it is not done
        self.workers[worker_file] = False
        if self.delegate:
            fp = worker_file.replace(".pickle", ".out")
            with open(fp, 'rb') as f:
                for line in f.readlines():
                    # assume still running
                    if "CANCELLED" in line:
                        self.workers[worker_file] = True
                    if "exited" in line:
                        self.workers[worker_file] = True
                    
        # load file
        else:
            worker = Worker.load_from(worker_file)
            if worker.task_done:
                self.workers[worker_file] = True


    def check_pending_workers(self):
        pending = []
        for file in os.listdir():
            if file in self.workers:
                pending.append(file)

    def override_config(self, config: dict) -> dict:
        overriden = self.reader.get("general", {}).copy()
        for k, v in config.items():
            overriden[k] = v
        return overriden

    def setup_workers(self, experiments):
        self.workers = {}
        for idx, experiment in enumerate(experiments):
            self.setup_worker(experiment, idx)

    def setup_worker(self, experiment, idx: int):
        name, config = experiment
        if experiment_module := experiments.get(name):
            # create tasks
            config = self.override_config(config)
            # extract deadline in case it overflows server deadline
            # convert time to int
            date = datetime.datetime.now()

            # default to 24 hours
            job_time = self.reader.get('sbatch', {} ).get("job_time", "24:00:0")
            hours, minutes, seconds = [int(i) for i in job_time.split(":")]
            deadline = (date + datetime.timedelta(
                                                hours = hours,
                                                minutes = minutes,
                                                seconds = seconds)).toordinal()
            # setup worker settings
            worker_settings = dict(
                id        = f"{self.pid}-{idx}",
                deadline  = deadline,
                base      = self.base,
                job_time  = job_time,
                autostart = False
            )

            # bind configs
            config = dict(experiment_settings = config,
                        worker_settings = worker_settings)

            # setup worker / delegate?
            worker = Worker(experiment_module, config)
            fp = worker.dump_to_disk()
            self.workers[fp] = worker.tasks_done
        else:
            print(f"!Warning! {experiment} not found")
       

    def delegate_worker(self, worker_file):
        command = "srun " # mind the space

        for k, v in sbatch.items():
            command += "--{k}={v} "
        
        command += f"--output {worker_file.split()}.out "
        command += f"python do_task.py --worker_file {worker_file}" 
        subprocess.call(command.split())


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
        self.update_deadline()

        self.is_running = False

    def check_deadline(self):
        # only use when on the server
        while self.get_jobs() < self.max_jobs:
            time.sleep(1)
            # restart process
            if time.time() >= self.deadline * self.threshold:
                self.restart()
                
    def update_deadline(self):
        current_time = time.time()
        timeout_server = 60 * 15 # //minutes
        self.deadline = current_time + timeout_server

    def clean_up(self):
        # if resumed cleanup file
        if os.path.exists(self.name + ".pickle"):
            os.remove(fp)

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

    
         

