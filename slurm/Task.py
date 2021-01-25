import time, os, pickle, re, subprocess

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

    def run(self):
        while self.tasks:
            self.log(f"{len(self.tasks)} tasks left")
            task = self.tasks.pop(0)
            results = task.run()

            directory = os.path.join(self.base, task.output_directory)
            os.makedirs(directory, exist_ok = True)
            fp = os.path.join(directory,
                              f"{task.settings}.pickle")

            with open(fp, 'wb') as f:
                self.log(f"Saving results to {directory}")
                pickle.dump(results, f)
                self.log("Done with task")

            # resuming if deadline
            self.check_deadline()

    def check_deadline(self):
        # dump object and resume
        if time.time() >= self.deadline * self.threshold:
            with open(self.name + ".pickle", 'wb') as f:
                pickle.dump(self, f)
            self.log("Deadline reached")
            self.log("Stashing and resuming")
            exit()

    def log(self, message : str):
        print(f"Worker {self.id}: {message}")

    


