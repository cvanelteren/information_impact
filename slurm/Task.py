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
    def __init__(self, tasks: list,
                 deadline: int,
                 id:  str,
                 threshold: float = .8,
                 autostart = False,
                 base = '', 
                 *args, **kwargs):

        self.tasks = tasks
        self.deadline = deadline # in uints

        # be within threshold of deadline
        self.threshold = threshold

        # give it a personality
        self.id = id

        self.name = f"worker_{self.id}"

        self.base = base

        # start worker
        if autostart:
            self.run()

    def create_output_file(self, task):
        fp = os.path.join(directory,
                        f"{task.settings}.pickle")
        return fp


    def run(self):
        self._running = True
        while self.tasks and self._running:
            self.log(f"{len(self.tasks)} tasks left")
            task = self.tasks.pop(0)
            results = task.run()

            directory = os.path.join(self.base, task.output_directory)
            fp = self.create_output_file(task)

            os.makedirs(directory, exist_ok = True)

            with open(fp, 'wb') as f:
                self.log(f"Saving results to {directory}")
                pickle.dump(results, f)
                self.log("Done with task")

            # resuming if deadline
            self.check_deadline()

        # termination case: no more tasks but still running
        if self._running and len(self.tasks) == 0:
            self.log("Done with all tasks")
            self.clean_up()

    def clean_up(self):
        fp =  self.create_dump_name()
        if os.path.exists(fp):
            os.remove(fp)

    def create_dump_name(self):
        return self.name + ".pickle", 'wb'

    def dump_to_disk(self):
        fp = self.create_dump_name()
        with open(fp, 'wb') as f:
            pickle.dump(self, f)

        self.log("Dumping to disk")

    def restart(self):
        self.dump_to_disk()
        self._running = False

    def check_deadline(self):
        # dump object and resume
        if time.time() >= self.deadline * self.threshold:
            self.log("Deadline reached")
            self.restart()

    def log(self, message : str):
        print(f"Worker {self.id}: {message}")

    


