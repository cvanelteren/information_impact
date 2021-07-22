import click


# contains folder with python files as "experments"
import experiments as EXPERIMENTS
from datetime import datetime
from Task import Worker, Task
import os, pickle, toml, time
from datetime import datetime


class SimpleExperimentManager:
    """
    Controls the file scheduled to slurm.
    The main purpose is to
    1. Read the config file
    2. Queue up the experiments achieved by running setup_experiment
    3. Decompose the tasks to separate pickle files
    """

    task_dir = "./tasks"
    data_dir = "./data"

    def __init__(self, config: str):
        # put workers here
        os.makedirs(SimpleExperimentManager.task_dir, exist_ok=True)
        os.makedirs(SimpleExperimentManager.data_dir, exist_ok=True)

        self.pid = time.time()
        self.tasks = []  # holds task to be called

        # setup workers
        self.reader = toml.load(config)
        self.setup_workers(self.reader.get("experiments", {}))

        self.create_task_file()

    def setup_workers(self, experiments: dict) -> None:
        for idx, experiment in enumerate(experiments.items()):
            self.setup_worker(experiment, idx)

    def setup_worker(self, experiment: tuple, idx: int) -> None:
        name, config = experiment
        # setup the worker
        # do not start the job yet
        # tell it the output dir
        worker_settings = dict(
            id=f"{self.pid}-{idx}",
            base=SimpleExperimentManager.data_dir,
            autostart=False,
        )
        # fetch the run file
        if module := EXPERIMENTS.get(name):

            print(f"Loading {module}")
            config = dict(experiment_settings=config, worker_settings=worker_settings)
            worker = Worker(module, config)
            self.create_tasks(worker)

        else:
            print(f"Warning {experiment} not found!")

            def create_tasks(self, worker: Worker) -> None:
        """
        Puts all tasks of the worker into the task directory
        """
        worker.setup_experiment()
        for task_idx, task in enumerate(worker.tasks):
            fp = os.path.join(
                SimpleExperimentManager.task_dir, f"{worker.id}-{task_idx}.pkl"
            )
            with open(fp, "wb") as f:
                pickle.dump(task, f)
            self.tasks.append(fp)

    def create_task_file(self) -> None:
        """
        Dumps txt containing all the jobs to be run
        """
        now = datetime.now()
        fp = "./tasks.txt"
        print("Writing tasks to file")
        tmp = [f"{i}\n" for i in self.tasks]
        with open(fp, "a") as f:
            f.writelines(tmp)
        print("Done!")

    @staticmethod
    def run_task(task_name: str) -> None:
        with open(task_name, "rb") as task_file:
            task = pickle.load(task_file)

        output_file = SimpleExperimentManager.create_task_output_file(task)
        results = task.run()
        with open(output_file, "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def create_task_output_file(task: Task) -> str:
        fp = os.path.join(SimpleExperimentManager.data_dir, f"{task.gen_id()}.pkl")
        print(f"Created output file: \t{fp}")
        return fp


@click.command()
@click.option("-c", "--config", default="", type=str)
@click.option("-r", "--run", default="", type=str)
def setup_experiments(config: str, run: str) -> None:
    """
    Decomposes task  for  each worker  in separate  job
    file. The  task will  be called from  a bash  script and
    sent to slurm
    """

    # sets up workers
    if config != "":
        manager = SimpleExperimentManager(config)

    # run a worker task
    if run != "":
        SimpleExperimentManager.run_task(run)


if __name__ == "__main__":
    setup_experiments()
