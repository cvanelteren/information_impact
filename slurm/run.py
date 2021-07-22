import toml, pickle, os, subprocess, datetime, importlib, re, time
from Task import Worker
import experiments, click  # contains the experiments


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
    def __init__(self, settings_file, timeout=600 * 5):
        self.reader = toml.load(settings_file)

        # assume running on the server
        # TODO deal with local processing of experiments
        self.delegate = False
        # TODO: REMOVE THIS assume server

        self.timeout_server = timeout
        if os.uname().nodename.endswith("uva.nl_"):
            self.max_jobs = 5
            # max jobs on the server
            self.delegate = True

        # counter for reset to prevent overwriting workers
        # unique id base on time ensures uniqueness
        self.pid = time.time()

        self.is_running = False  # running overwrites
        self.shutdown = False
        self.threshold = 0.95

        # create output folder
        date = datetime.datetime.now().isoformat()

        # group simulations in the same folder
        base = f"data/{self.__class__.__name__}-{date}"
        os.makedirs(base, exist_ok=True)
        self.base = base
        self.task_dir = "./tasks"
        os.makedirs("./tasks", exists_ok=True)

        self.name = f"{self.__class__.__name__}-{self.pid}"

        # holds workers name and whether they are done
        self.workers = {}  # maps filepath to bool
        self.setup_workers(self.reader.get("experiments", {}))

    def run(self):
        # cleanup if resumed
        self.update_deadline()
        self.clean_up()
        self.is_running = True
        while self.is_running:
            # are all workers done?
            if all([w for w in self.workers.values()]):
                self.is_running = False
                self.shutdown = True
                break
            for worker_file, is_done in self.workers.items():
                is_done = self.check_worker(worker_file)
                if not is_done:
                    if self.delegate and self.get_jobs() < self.max_jobs:
                        self.delegate_worker(worker_file)
                        self.check_deadline()
                    else:
                        # run it
                        worker = Worker.load_from(worker_file)
                        worker.run()
                        self.workers[worker_file] = worker.tasks_done

                if self.is_running == False:
                    break

        if self.shutdown:
            self.clean_up()

    def check_worker(self, worker_file):
        """Check whether the worker is working"""
        # check error logs

        # assume it is not done
        self.workers[worker_file] = False
        if self.delegate:
            fp = worker_file.replace(".pickle", ".out")
            with open(fp, "rb") as f:
                for line in f.readlines():
                    line = line.lower()
                    # assume still running

                    if "cancelled" in line:
                        self.workers[worker_file] = True
                    if "exited" in line:
                        self.workers[worker_file] = True
                    if "termination" in line:
                        self.workers[worker_file] = True

        # load file
        else:
            worker = Worker.load_from(worker_file)
            self.workers[worker_file] = worker.tasks_done

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
        for idx, experiment in enumerate(experiments.items()):
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
            job_time = self.reader.get("sbatch", {}).get("deadline", "24:00:0")
            hours, minutes, seconds = [int(i) for i in job_time.split(":")]
            td = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
            deadline = (date + td).toordinal()
            # setup worker settings
            worker_settings = dict(
                id=f"{self.pid}-{idx}",
                deadline=deadline,
                base=self.base,
                job_time=td.total_seconds(),
                autostart=False,
            )

            # bind configs
            config = dict(experiment_settings=config, worker_settings=worker_settings)

            # setup worker / delegate?
            worker = Worker(experiment_module, config)
            fp = worker.dump_to_disk(self.task_dir)
            self.workers[fp] = worker.tasks_done
        else:
            print(f"!Warning! {experiment} not found")

    def delegate_worker(self, worker_file):
        command = "srun "  # mind the space

        for k, v in sbatch.items():
            command += "--{k}={v} "

        command += f"--output {worker_file.split()}.out "
        command += f"python do_task.py --worker_file {worker_file}"
        subprocess.call(command.split())

    def dump_to_disk(self):
        fp = f"{self.name}.pickle"

        print(f"dumping to {fp}")
        with open(fp, "wb") as f:
            pickle.dump(self, f)

        print(fp)

    def exit(self):
        self._running = False
        command = f"python {__file__}"
        # os.system(command)
        subprocess.Popen(command.split())
        # os.system(command)
        print("exiting")

    # TODO: write test case dump file to disk and call itself
    def restart(self):
        self.dump_to_disk()
        # update deadline
        self.exit()

    def check_deadline(self):
        # only use when on the server
        if time.time() >= self.deadline * self.threshold:
            print("deadline reached; restarting")
            # restart process
            self.restart()

    def update_deadline(self):
        current_time = time.time()
        self.deadline = current_time + self.timeout_server

    def clean_up(self):
        # if resumed cleanup file
        fp = self.name + ".pickle"
        if os.path.exists(fp):
            os.remove(fp)

        # remove worker files
        if self.shutdown:
            for k, v in self.workers.items():
                try:
                    os.remove(k)
                except:
                    continue

    def get_jobs(self):
        # count output squeue
        call = "squeue -u {echo $USER} -h -t pending,running -r | wc -l"
        jobs = subprocess.check_output(call, shell=True)  # return string
        return int(jobs)


SETTINGS = "settings.toml"


@click.command()
@click.option("-s", default="", type=str)
def run_main(s: str) -> None:
    """
    Script that can be envoked from cli

    Attempt to load the string @s which points to a config file,
    else it attempts to load the workers from tasks
    """
    # if config file is empty, attempt to load worker pickle
    os.makedirs("./tasks", exists_ok=True)
    if s == "":
        pattern = f"{ExperimentManager.__name__}.+\.pickle"
        for file in os.listdir("./tasks"):
            # found file
            if re.match(pattern, file):
                print("Found a match")
                fi = os.path.join("./tasks/", file)
                with open(fi, "rb") as f:
                    manager = pickle.load(f)
                    if not manager.is_running:
                        manager.run()
                    print("resuming")
                break
    else:
        manager = ExperimentManager(s)
        manager.run()


if __name__ == "__main__":
    run_main()
