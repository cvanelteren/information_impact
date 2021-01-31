import time, os, pickle, re, subprocess

# TODO: write nosetests check the deadline for the worker
class Task:
    def __init__(self, settings : dict, output_directory : str,
                *args, **kwargs):
        self.settings = settings
        self.output_directory = output_directory

    def run(self):
        assert False, "Needs to be implemented"

class Worker:
    def __init__(self,
                 experiment : object,
                 config : dict,
                 ):

        # setup variables
        self.experiment_setup = experiment.setup
        self.worker_settings = config.get('worker_settings', {})
        self.experiment_settings = config.get('experiment_settings', {})

        # setup worker
        self.setup_worker(self.worker_settings)

        self.tasks_done = False
        self._restart = False
        self.tasks = []
        # start worker
        if self.autostart:
            self.run()

    def setup_experiment(self):
        self.tasks = self.experiment_setup(self.experiment_settings)

    def setup_worker(self, settings: dict ):
        # TODO set better defaults
        defaults = dict(deadline = 1,
                        threshold = 1,
                        id = '',
                        autostart = False,
                        base = '',
                        job_time = 0)

        # assign worker properties
        for k,v in defaults.items(): 
            self.__dict__[k] = settings.get(k, v)

        self.name = f"worker_{self.id}"

    def create_output_file(self, task, dire = ""):
        fp = os.path.join(dire, f"{task.settings}.pickle")
        return fp


    def run(self):
        """
        setup experiments and start running
        """

        # only when first init
        if self.tasks_done == False ^  len(self.tasks) == 0 :
            self.setup_experiment()

        self._restart = False
        while self.tasks_done == False and self._restart == False:
            self.log(f"{len(self.tasks)} tasks left")
            task = self.tasks.pop(0)
            results = task.run()

            dire = os.path.join(self.base, task.output_directory)

            os.makedirs(dire, exist_ok = True)

            fp = os.path.join(dire, self.create_output_file(task))
            print(fp)
            with open(fp, 'wb') as f:
                self.log(f"Saving results to {dire}")
                pickle.dump(results, f)
                self.log("Done with task")

            # resuming if deadline
            self.check_deadline()

            # termination case: no more tasks but still running
            if len(self.tasks) == 0:
                self.tasks_done = True
                self.log("Done with all tasks")
                self.clean_up()

    def update_deadline(self):
        current_time = time.time()
        self.deadline = current_time + self.job_time
        
    def clean_up(self):
        fp =  self.create_dump_name()
        if os.path.exists(fp):
            os.remove(fp)

    def create_dump_name(self):
        return self.name + ".pickle"

    def dump_to_disk(self):
        fp = self.create_dump_name()
        with open(fp, 'wb') as f:
            pickle.dump(self, f)
        
        # with open(fp, 'rb') as f:
            # print(">", pickle.load(f))
        self.log("Dumping to disk")
        return fp

    def restart(self):
        self.dump_to_disk()
        self.update_deadline()
        self._restart = True
        # exit compute node

    def check_deadline(self):
        # dump object and resume
        if time.time() >= self.deadline * self.threshold:
            self.log("Deadline reached")
            self.restart()

    def log(self, message : str):
        print(f"Worker {self.id}: {message}")

    @staticmethod
    def load_from(file):
        # if re.match("worker*+.pickle", file):
            # print("->", file)
        with open(file, 'rb') as f:
            return pickle.load(f)

    


